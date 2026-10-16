"""Startup schemas and per-Run tool factories; native messages remain the only protocol."""

from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain_core.tools import StructuredTool, ToolException
from agent_service.failure import failure_info
from agent_service.policy import active_call, mark_correctable
from pydantic import BaseModel, ConfigDict, Field

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
                content = result.pop("content")
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
