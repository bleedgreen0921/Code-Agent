"""Trusted, immutable Docker configuration; parsed by API without contacting Docker."""

import math
import os
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

IMPLEMENTATION_VERSION = "workspace-v2"
WORKSPACE_TOOLS = frozenset(
    {"glob_search", "grep_search", "read_file", "edit_file", "shell", "git_status", "git_log", "git_diff", "git_read_file"}
)
WRITE_TOOLS = frozenset({"edit_file", "shell"})


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    image: str | None = None
    user: str | None = None
    cpus: float = Field(default=1.0, gt=0)
    memory_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    pids: int = Field(default=128, gt=0)
    tmp_bytes: int = Field(default=64 * 1024 * 1024, gt=0)
    command_timeout: float = Field(default=30.0, gt=0)
    cleanup_timeout: float = Field(default=10.0, gt=0)
    output_bytes: int = Field(default=16 * 1024, ge=512, le=16 * 1024)
    output_lines: int = Field(default=200, ge=4, le=200)
    file_bytes: int = Field(default=10 * 1024 * 1024, gt=0, le=10 * 1024 * 1024)

    @field_validator("cpus", "command_timeout", "cleanup_timeout")
    @classmethod
    def finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("Execution limits must be finite")
        return value

    @field_validator("image")
    @classmethod
    def immutable_image(cls, value):
        if value is not None and not re.fullmatch(
            r"(?:sha256:[a-f0-9]{64}|[^\s@]+@sha256:[a-f0-9]{64})", value
        ):
            raise ValueError("Execution image must be an immutable image ID or digest")
        return value

    @field_validator("user")
    @classmethod
    def non_root(cls, value):
        if value is not None and not re.fullmatch(r"[1-9][0-9]*:[0-9]+", value):
            raise ValueError("Execution user must be a non-root numeric UID:GID")
        return value

    def effective_user(self):
        if self.user:
            return self.user
        if os.getuid() == 0:
            raise ValueError("Root Worker requires an explicit non-root execution user")
        return f"{os.getuid()}:{os.getgid()}"
