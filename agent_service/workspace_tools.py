"""Startup schemas and per-Run tool factories; native messages remain the only protocol."""

from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_core.tools import StructuredTool, ToolException
from agent_service.failure import failure_info
from agent_service.policy import active_call, mark_correctable
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Literal

from agent_service.execution_adapter import ControlledShellMiddleware
from agent_service.output import limit_output


class ReadFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=200)


class EditFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    new_text: str
    create: bool = False
    expected_hash: str | None = None
    old_text: str | None = None


class GitInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GitStatusInput(GitInput):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class GitLogInput(GitInput):
    revision: str = Field(default="HEAD", min_length=1, max_length=1024)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=10, ge=1, le=50)


class GitDiffInput(GitInput):
    source: Literal["staged", "unstaged", "commit"]
    commit: str | None = Field(default=None, min_length=1, max_length=1024)
    view: Literal["files", "patch"] = "files"
    path: str | None = Field(default=None, min_length=1)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=200)

    @field_validator("path")
    @classmethod
    def literal_path(cls, value):
        if value is not None and (
            value.startswith(("/", ":"))
            or "\x00" in value
            or any(p in ("", ".", "..") for p in value.split("/"))
        ):
            raise ValueError("Expected a literal workspace-relative file path")
        return value

    @model_validator(mode="after")
    def selected_source(self):
        if (self.source == "commit") != (self.commit is not None):
            raise ValueError("commit is required only for source=commit")
        if self.view == "patch" and self.path is None:
            raise ValueError("Patch view requires path; list files first")
        return self


class GitReadFileInput(GitInput):
    source: Literal["commit", "index"]
    path: str = Field(min_length=1)
    commit: str | None = Field(default=None, min_length=1, max_length=1024)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=200)

    _literal_path = field_validator("path")(GitDiffInput.literal_path.__func__)

    @model_validator(mode="after")
    def selected_source(self):
        if (self.source == "commit") != (self.commit is not None):
            raise ValueError("commit is required only for source=commit")
        return self


GIT_DECLARATIONS = [
    (
        "git_status",
        "Read live Git HEAD/branch and per-file staged, unstaged, untracked and conflict status. Paginate with next_offset. No atomic snapshot across calls.",
        GitStatusInput,
    ),
    (
        "git_log",
        "List local reachable commit IDs, parents, author/time and title. Use returned full revision for subsequent pages; no ranges or fetching.",
        GitLogInput,
    ),
    (
        "git_diff",
        "List changed files first (view=files), then read one file patch. staged compares HEAD to index; unstaged compares index to worktree; commit compares first parent to fixed commit (native root comparison). next_line indexes output, NOT source lines; use patch hunks for issue locations. No submodule interiors.",
        GitDiffInput,
    ),
    (
        "git_read_file",
        "Read a UTF-8 regular blob from a fixed commit or stage 0 index, with original line numbers and object ID. No checkout/filters. Use index for staged context, read_file for worktree/untracked. Conflicts and unsupported content report errors.",
        GitReadFileInput,
    ),
]


def file_tools(scope=None):
    native = FilesystemFileSearchMiddleware(root_path="/workspace")
    declarations = [(t.name, t.description, t.args_schema) for t in native.tools]
    declarations += [
        (
            "read_file",
            "Read a UTF-8 regular file in the workspace with line numbers and SHA-256. Paths are workspace-relative. Read at most 200 lines per call.",
            ReadFileInput,
        ),
        (
            "edit_file",
            "Create one file without overwriting (create=true), or replace exactly one old_text using expected_hash from read_file. Paths are workspace-relative. Changes persist after failure or cancellation.",
            EditFileInput,
        ),
    ]
    declarations += GIT_DECLARATIONS
    tools = []
    for name, description, schema in declarations:

        def bind(name):
            async def execute(**args):
                if scope is None:
                    raise RuntimeError("Tool schema declaration cannot execute")
                result = await scope.file_call(name, args)
                if "error" in result:
                    info = failure_info(result.get("code", "FILE_OPERATION_FAILED"))
                    if info["code"] == "FILE_VERSION_CONFLICT":
                        mark_correctable(info["code"])
                    active_call.get().diagnostic = info
                    raise ToolException(result["error"])
                if getattr(scope, "evidence", None) and name in (
                    "read_file",
                    "git_read_file",
                ):
                    return await scope.evidence.read_result(
                        name, args, result, scope.settings
                    )
                content = result.pop("content")
                if name.startswith("git_"):
                    return content, result
                if result.get("truncated"):
                    content += "\n[file results truncated]"
                if result.get("sha256"):
                    content = "sha256: " + result["sha256"] + "\n" + content
                content, truncated = limit_output(
                    content, scope.settings.output_bytes, scope.settings.output_lines
                )
                result["truncated"] = result.get("truncated", False) or truncated
                return content, result

            return execute

        tools.append(
            StructuredTool.from_function(
                coroutine=bind(name),
                name=name,
                description=description,
                args_schema=schema,
                response_format="content_and_artifact",
                handle_tool_error=True,
            )
        )
    return tools


def builtin_tools():
    return [*file_tools(), *ControlledShellMiddleware().tools]


def assemble(scope, names):
    tools = [t for t in file_tools(scope) if t.name in names]
    middleware = []
    if "shell" in names:
        shell = ControlledShellMiddleware(scope)
        middleware.append(shell)
        tools += shell.tools
    return tools, middleware
