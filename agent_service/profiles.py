"""Startup-only declarations and trusted Python extension points."""

import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from agent_service.execution_config import (
    ExecutionSettings,
    WORKSPACE_TOOLS,
    IMPLEMENTATION_VERSION,
)
from agent_service.workspace_tools import builtin_tools


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def tool_schema(tool):
    schema = tool.tool_call_schema
    return schema if isinstance(schema, dict) else schema.model_json_schema()


class Declaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Workspace(Declaration):
    id: str = Field(min_length=1)
    root: str


class Profile(Declaration):
    id: str = Field(min_length=1)
    system_prompt: str | None = None
    tools: tuple[str, ...] = ()
    requires_workspace: bool = False


class RuntimeFile(Declaration):
    version: Literal[1]
    workspaces: list[Workspace] = []
    profiles: list[Profile] = []
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)


@dataclass(frozen=True)
class Extension:
    """Only trusted code constructs extensions; version changes with their behavior."""

    version: str = "1"
    middleware_factory: Callable = lambda: ()
    state_schema: Any = None
    response_format: Any = None


class ConfigurationChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class PermissionPolicy:
    tools: Mapping[str, BaseTool]
    validators: Mapping[str, Callable] = field(default_factory=dict)
    binding_fingerprint: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))
        object.__setattr__(self, "validators", MappingProxyType(dict(self.validators)))


class Registry:
    def __init__(self, config=None, *, tools=(), validators=None, extensions=None):
        tools = (*builtin_tools(), *tools)
        self.tools = MappingProxyType({t.name: t for t in tools})
        if len(self.tools) != len(tools):
            raise ValueError("Duplicate tool names")
        self.validators = MappingProxyType(dict(validators or {}))
        self.extensions = MappingProxyType(dict(extensions or {}))
        if "chat" in self.extensions:
            raise ValueError("chat is reserved")
        config = config or RuntimeFile(version=1)
        self.execution = config.execution
        profiles = {
            "chat": Profile(id="chat"),
            "workspace_read": Profile(
                id="workspace_read",
                tools=("glob_search", "grep_search", "read_file"),
                requires_workspace=True,
            ),
            "workspace_write": Profile(
                id="workspace_write",
                tools=("glob_search", "grep_search", "read_file", "edit_file", "shell"),
                requires_workspace=True,
            ),
        }
        workspaces = {}
        for workspace in config.workspaces:
            root = Path(workspace.root)
            if not root.is_absolute() or not root.is_dir():
                raise ValueError(
                    "Workspace root must be an existing absolute directory"
                )
            if workspace.id in workspaces:
                raise ValueError("Duplicate workspace ID")
            workspaces[workspace.id] = str(root.resolve(strict=True))
        for profile in config.profiles:
            if profile.id in profiles:
                raise ValueError("Duplicate or reserved profile ID")
            if (
                len(set(profile.tools)) != len(profile.tools)
                or set(profile.tools) - self.tools.keys()
            ):
                raise ValueError("Unknown or duplicate tools")
            profiles[profile.id] = profile
        if (
            self.extensions.keys() - profiles.keys()
            or self.validators.keys() - self.tools.keys()
        ):
            raise ValueError("Unknown extension profile or validator tool")
        self.profiles = MappingProxyType(profiles)
        self.workspaces = MappingProxyType(workspaces)

    @classmethod
    def from_env(cls, **trusted):
        path = os.environ.get("AGENT_RUNTIME_CONFIG")
        if not path:
            return cls(**trusted)
        with open(path, "rb") as file:
            raw = tomllib.load(file)
        # TOML arrays are lists; the immutable profile stores tool names as tuples.
        for profile in raw.get("profiles", []):
            if "tools" in profile and isinstance(profile["tools"], list):
                profile["tools"] = tuple(profile["tools"])
        return cls(RuntimeFile.model_validate(raw), **trusted)

    def bind(self, profile="chat", workspace_id=None):
        if profile not in self.profiles:
            raise ValueError("Unknown profile")
        declaration = self.profiles[profile]
        if workspace_id is not None and workspace_id not in self.workspaces:
            raise ValueError("Unknown workspace")
        if (
            declaration.requires_workspace or set(declaration.tools) & WORKSPACE_TOOLS
        ) and workspace_id is None:
            raise ValueError("Profile requires a workspace")
        root = self.workspaces.get(workspace_id)
        tool_schemas = {
            name: fingerprint(tool_schema(self.tools[name]))
            for name in sorted(declaration.tools)
        }
        payload = {
            "version": 1,
            "profile": profile,
            "profile_fingerprint": fingerprint(
                {
                    **declaration.model_dump(),
                    "tools": sorted(declaration.tools),
                    "extension_version": self.extensions.get(
                        profile, Extension()
                    ).version,
                    "tool_schemas": tool_schemas,
                    **(
                        {
                            "execution": self.execution.model_dump(),
                            "tool_implementation": IMPLEMENTATION_VERSION,
                        }
                        if set(declaration.tools) & WORKSPACE_TOOLS
                        else {}
                    ),
                }
            ),
            "workspace_id": workspace_id,
            "workspace_identity": fingerprint(root) if root else None,
        }
        return {**payload, "fingerprint": fingerprint(payload)}

    def resolve(self, binding):
        try:
            current = self.bind(binding["profile"], binding["workspace_id"])
        except (ValueError, KeyError) as exc:
            raise ConfigurationChanged(
                "Submitted configuration is unavailable"
            ) from exc
        if current != binding:
            raise ConfigurationChanged("Submitted configuration changed")
        return self.profiles[current["profile"]]

    def policy(self, binding, *, actual_tools=None):
        profile = self.resolve(binding)
        return PermissionPolicy(
            {name: (actual_tools or self.tools)[name] for name in profile.tools},
            {
                name: self.validators[name]
                for name in profile.tools
                if name in self.validators
            },
            binding["fingerprint"],
        )


DEFAULT_BINDING = Registry().bind()
