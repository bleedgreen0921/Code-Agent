"""Static HTTP-only MCP declarations; local catalogs never connect at API startup."""

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.tools import StructuredTool
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_service.schema import local_schema

VERSION = "mcp-http-v1"
MAX_MESSAGE_BYTES = 1024 * 1024
MAX_CATALOG_BYTES = 32768


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str = Field(min_length=1, max_length=200)
    alias: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    access: Literal["read", "write"]
    execution: Literal["sync", "background"] = "sync"
    cancel_retry_safe: bool = False

    @model_validator(mode="after")
    def cancel_authority(self):
        if self.cancel_retry_safe and self.execution != "background":
            raise ValueError("cancel_retry_safe requires a background MCP tool")
        return self


class Server(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    transport: Literal["http"] = "http"
    url: str
    account_id: str = Field(min_length=1, max_length=200)
    catalog: str
    tools: list[Selection] = []
    bearer_env: str | None = None
    headers_env: dict[str, str] = {}
    allow_insecure_loopback: bool = False

    @model_validator(mode="after")
    def validate_endpoint(self):
        url = urlsplit(self.url)
        local = url.hostname == "localhost"
        try:
            local = local or ipaddress.ip_address(url.hostname or "").is_loopback
        except ValueError:
            pass
        if (
            (
                url.scheme != "https"
                and not (
                    url.scheme == "http" and local and self.allow_insecure_loopback
                )
            )
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "MCP requires HTTPS or explicitly allowed loopback HTTP, without URL credentials/query/fragment"
            )
        _ = url.port
        for key, value in self.headers_env.items():
            if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) or key.lower() in {
                "host",
                "content-type",
                "content-length",
                "accept",
                "mcp-session-id",
                "mcp-protocol-version",
            }:
                raise ValueError("Invalid credential header")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
                raise ValueError(
                    "Credentials must reference environment variable names"
                )
        if self.bearer_env is not None and (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.bearer_env)
            or any(k.lower() == "authorization" for k in self.headers_env)
        ):
            raise ValueError("Invalid or duplicate bearer reference")
        return self


class MCPSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    servers: list[Server] = []
    init_timeout: float = Field(default=10, gt=0, le=10)
    call_timeout: float = Field(default=30, gt=0, le=30)
    cleanup_timeout: float = Field(default=10, gt=0, le=10)


def contract(tool):
    return tool.model_dump(mode="json", by_alias=True, exclude_none=True)


def read_catalog(path):
    file = Path(path)
    if (
        not file.is_absolute()
        or not file.is_file()
        or file.stat().st_size > MAX_MESSAGE_BYTES
    ):
        raise ValueError("MCP catalog must be an absolute file of at most 1 MiB")
    raw = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("tools"), list):
        raise ValueError("Expected a standard tools/list catalog")
    tools = [Tool.model_validate(t) for t in raw["tools"]]
    result = {t.name: contract(t) for t in tools}
    if len(result) != len(tools):
        raise ValueError("Duplicate MCP tool names in catalog")
    return result


@dataclass(frozen=True)
class CatalogTool:
    server: Server
    selection: Selection
    serialized: str

    @property
    def definition(self):
        return json.loads(self.serialized)

    def manifest(self):
        return {
            "version": VERSION,
            "server_id": self.server.id,
            "account_id": self.server.account_id,
            "tool": self.selection.model_dump(exclude={"cancel_retry_safe"})
            | ({"cancel_retry_safe": True} if self.selection.cancel_retry_safe else {}),
            "catalog_hash": digest(self.definition),
            "connection_hash": digest(
                {
                    "url": self.server.url,
                    "headers_env": self.server.headers_env,
                    "bearer_env": self.server.bearer_env,
                }
            ),
        }

    def declaration(self):
        # Match the native adapter's provider normalization without replacing its
        # eventual runtime tool. This helper is pinned by contract tests.
        from langchain.mcp.tools import _normalize_mcp_schema

        async def unavailable(**args):
            raise RuntimeError("MCP tool has no Run-owned connection")

        return StructuredTool(
            name=self.selection.alias,
            description=self.definition.get("description", ""),
            args_schema=_normalize_mcp_schema(self.definition["inputSchema"]),
            coroutine=unavailable,
        )


def load_catalogs(settings):
    selected, servers = {}, set()
    for server in settings.servers:
        if server.id in servers:
            raise ValueError("Duplicate MCP server ID")
        servers.add(server.id)
        catalog = read_catalog(server.catalog)
        remote_names = set()
        for tool in server.tools:
            if tool.alias in selected or tool.name in remote_names:
                raise ValueError("Duplicate MCP alias or server tool")
            if tool.name not in catalog:
                raise ValueError("Selected MCP tool missing from local catalog")
            definition = catalog[tool.name]
            local_schema(definition["inputSchema"])
            if definition.get("outputSchema") is not None:
                local_schema(definition["outputSchema"])
            selected[tool.alias] = CatalogTool(
                server, tool, json.dumps(definition, ensure_ascii=False, sort_keys=True)
            )
            remote_names.add(tool.name)
    return MappingProxyType(selected)
