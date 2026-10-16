"""Startup-only declarations and trusted Python extension points."""

import hashlib
import inspect
import json
import math
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
from agent_service.reliability import ReliabilitySettings
from agent_service.mcp_config import MCPSettings, load_catalogs, MAX_CATALOG_BYTES
from agent_service.workspace_tools import builtin_tools
from agent_service.knowledge import (
    KnowledgeSettings,
    KNOWLEDGE_TOOLS,
    knowledge_tools,
    manifest as knowledge_manifest,
)
from agent_service.delegation import (
    delegate_tool,
    manifest as delegation_manifest,
    READ_TOOLS,
)
from agent_service.skills import (
    load_skills,
    read_skill_tool,
    catalog,
    SKILL_VERSION,
    CATALOG_BYTES,
)
from agent_service.context import (
    validate_todos,
    TODO_LIMITS,
    TODO_VERSION,
    CONTEXT_VERSION,
)


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
    skills: tuple[str, ...] = ()


class SkillsConfig(Declaration):
    directories: tuple[str, ...] = ()


class RuntimeFile(Declaration):
    version: Literal[1]
    workspaces: list[Workspace] = []
    profiles: list[Profile] = []
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)


@dataclass(frozen=True)
class Extension:
    """Only trusted code constructs extensions; version changes with their behavior."""

    version: str = "1"
    middleware_factory: Callable = lambda: ()
    state_schema: Any = None
    response_format: Any = None
    verification_mode: Literal["report", "required"] | None = None
    verifier: Callable | None = None
    verification_timeout: float = 10.0
    no_progress_enabled: bool = False

    def __post_init__(self):
        if (
            type(self.version) is not str
            or not self.version
            or "\x00" in self.version
            or len(self.version.encode()) > 128
        ):
            raise ValueError("Extension version must be a bounded nonempty string")
        if self.verification_mode not in (None, "report", "required"):
            raise ValueError("Invalid verification mode")
        if (
            type(self.verification_timeout) not in (int, float)
            or not math.isfinite(self.verification_timeout)
            or self.verification_timeout <= 0
        ):
            raise ValueError("Verification timeout must be finite and positive")
        if type(self.no_progress_enabled) is not bool:
            raise ValueError("no_progress_enabled must be a boolean")
        if self.verifier is not None:
            target = self.verifier
            if not callable(target) or not (
                inspect.iscoroutinefunction(target)
                or inspect.iscoroutinefunction(getattr(target, "__call__", None))
            ):
                raise ValueError("Verifier must be an async callable")
            try:
                inspect.signature(target).bind(None, "", None)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Verifier must accept context, answer, checkpoint"
                ) from exc
        if self.verification_mode == "required" and self.verifier is None:
            raise ValueError("required verification needs a registered verifier")

    def manifest(self):
        from agent_service.no_progress import manifest

        return {
            **(
                {
                    "verification": {
                        "version": 1,
                        "mode": self.verification_mode,
                        "timeout": self.verification_timeout,
                        "verifier_registered": self.verifier is not None,
                        "extension_version": self.version,
                    }
                }
                if self.verification_mode
                else {}
            ),
            **({"no_progress": manifest()} if self.no_progress_enabled else {}),
        }


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
        from langchain.agents.middleware import TodoListMiddleware

        config = config or RuntimeFile(version=1)
        self.skills = load_skills(
            config.skills.directories,
            config.execution.output_bytes,
            config.execution.output_lines,
        )
        self.reliability = config.reliability
        self.mcp = config.mcp
        self.mcp_tools = load_catalogs(config.mcp)
        from agent_service.background import background_tools

        tools = (
            *background_tools(),
            *builtin_tools(),
            *knowledge_tools(),
            *(t.declaration() for t in self.mcp_tools.values()),
            delegate_tool(),
            *TodoListMiddleware().tools,
            read_skill_tool({}),
            *tools,
        )
        self.tools = MappingProxyType({t.name: t for t in tools})
        if len(self.tools) != len(tools):
            raise ValueError("Duplicate tool names")
        custom_validators = dict(validators or {})
        extra_todo_validator = custom_validators.get("write_todos")
        custom_validators["write_todos"] = lambda args, ctx: (
            validate_todos(args, ctx)
            and (
                extra_todo_validator is None
                or extra_todo_validator(args, ctx) is not False
            )
        )
        self.validators = MappingProxyType(custom_validators)
        self.extensions = MappingProxyType(dict(extensions or {}))
        if "chat" in self.extensions:
            raise ValueError("chat is reserved")
        self.execution = config.execution
        self.knowledge = config.knowledge
        profiles = {
            "chat": Profile(id="chat"),
            "workspace_read": Profile(
                id="workspace_read",
                tools=(
                    "glob_search",
                    "grep_search",
                    "read_file",
                    "write_todos",
                    "delegate_read",
                    *(sorted(KNOWLEDGE_TOOLS) if self.knowledge.enabled else ()),
                    *(("read_skill",) if self.skills else ()),
                ),
                skills=tuple(sorted(self.skills)),
                requires_workspace=True,
            ),
            "workspace_write": Profile(
                id="workspace_write",
                tools=(
                    "glob_search",
                    "grep_search",
                    "read_file",
                    "edit_file",
                    "shell",
                    "write_todos",
                    "delegate_read",
                    *(sorted(KNOWLEDGE_TOOLS) if self.knowledge.enabled else ()),
                    *(("read_skill",) if self.skills else ()),
                ),
                skills=tuple(sorted(self.skills)),
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
        for profile in profiles.values():
            selected = [
                self.mcp_tools[n].definition
                for n in profile.tools
                if n in self.mcp_tools
            ]
            if (
                len(json.dumps(selected, ensure_ascii=False).encode())
                > MAX_CATALOG_BYTES
            ):
                raise ValueError("Profile MCP schema and description exceed 32 KiB")
            if set(profile.tools) & KNOWLEDGE_TOOLS and not self.knowledge.enabled:
                raise ValueError("Knowledge tools require knowledge.enabled")
            if "delegate_read" in profile.tools and not set(profile.tools) & READ_TOOLS:
                raise ValueError("delegate_read requires a workspace read/search tool")
            if (
                len(profile.skills) != len(set(profile.skills))
                or set(profile.skills) - self.skills.keys()
            ):
                raise ValueError("Unknown or duplicate profile skills")
            if ("read_skill" in profile.tools) != bool(profile.skills):
                raise ValueError(
                    "read_skill requires an explicit nonempty skills allowlist"
                )
            if (
                len(catalog([self.skills[n] for n in sorted(profile.skills)]).encode())
                > CATALOG_BYTES
            ):
                raise ValueError("Profile skill catalog exceeds 8 KiB")
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
            for key in ("tools", "skills"):
                if isinstance(profile.get(key), list):
                    profile[key] = tuple(profile[key])
        if isinstance(raw.get("skills", {}).get("directories"), list):
            raw["skills"]["directories"] = tuple(raw["skills"]["directories"])
        return cls(RuntimeFile.model_validate(raw), **trusted)

    def bind(self, profile="chat", workspace_id=None):
        if profile not in self.profiles:
            raise ValueError("Unknown profile")
        declaration = self.profiles[profile]
        if workspace_id is not None and workspace_id not in self.workspaces:
            raise ValueError("Unknown workspace")
        if (
            declaration.requires_workspace
            or set(declaration.tools)
            & (WORKSPACE_TOOLS | KNOWLEDGE_TOOLS | {"start_background_shell"})
        ) and workspace_id is None:
            raise ValueError("Profile requires a workspace")
        root = self.workspaces.get(workspace_id)
        tool_schemas = {
            name: fingerprint(tool_schema(self.tools[name]))
            for name in sorted(declaration.tools)
        }
        payload = {
            "version": 1,
            **self.extensions.get(profile, Extension()).manifest(),
            "profile": profile,
            "profile_fingerprint": fingerprint(
                {
                    **declaration.model_dump(exclude={"skills"}),
                    **(
                        {
                            "todo": {"version": TODO_VERSION, "limits": TODO_LIMITS},
                            "context_policy": CONTEXT_VERSION,
                        }
                        if "write_todos" in declaration.tools
                        else {}
                    ),
                    **(
                        {
                            "skills": [
                                self.skills[n].manifest()
                                for n in sorted(declaration.skills)
                            ],
                            "skill_implementation": SKILL_VERSION,
                            "context_policy": CONTEXT_VERSION,
                        }
                        if declaration.skills
                        else {}
                    ),
                    **(
                        {
                            "tool_output_limits": {
                                "bytes": self.execution.output_bytes,
                                "lines": self.execution.output_lines,
                            }
                        }
                        if set(declaration.tools)
                        & (
                            {"write_todos", "read_skill"}
                            | KNOWLEDGE_TOOLS
                            | self.mcp_tools.keys()
                        )
                        else {}
                    ),
                    **(
                        {"delegation": delegation_manifest()}
                        if "delegate_read" in declaration.tools
                        else {}
                    ),
                    **(
                        {"knowledge": knowledge_manifest()}
                        if set(declaration.tools) & KNOWLEDGE_TOOLS
                        else {}
                    ),
                    **(
                        {
                            "mcp": {
                                "limits": {
                                    "init": self.mcp.init_timeout,
                                    "call": self.mcp.call_timeout,
                                    "cleanup": self.mcp.cleanup_timeout,
                                },
                                "tools": [
                                    self.mcp_tools[n].manifest()
                                    for n in sorted(declaration.tools)
                                    if n in self.mcp_tools
                                ],
                            }
                        }
                        if set(declaration.tools) & self.mcp_tools.keys()
                        else {}
                    ),
                    "reliability": self.reliability.model_dump(),
                    **self.extensions.get(profile, Extension()).manifest(),
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
                        if set(declaration.tools)
                        & (WORKSPACE_TOOLS | {"start_background_shell"})
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
