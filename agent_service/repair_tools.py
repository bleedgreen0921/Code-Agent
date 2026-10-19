"""Repair-only observations; all identities and permissions come from the scope."""

from typing import Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field


class PageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class DiffInput(PageInput):
    view: Literal["files", "patch"] = "files"


class ExecutionInput(PageInput):
    execution_id: str
    view: Literal["result", "log"] = "result"


class ContextInput(PageInput):
    task_id: str


def repair_tools(scope=None):
    declarations = (
        (
            "candidate_diff",
            DiffInput,
            "Read baseline-to-current candidate file inventory or patch, with version identity. User pre-existing changes are part of baseline.",
        ),
        (
            "read_repair_execution",
            ExecutionInput,
            "Read this repair Run check result or diagnostic log in pages. Checks identify immutable versions.",
        ),
        (
            "read_repair_context",
            ContextInput,
            "Read an explicitly authorized prior formal analysis or Review report in pages. Historical claims must be reconfirmed.",
        ),
    )
    result = []
    for name, schema, description in declarations:

        def bind(name):
            async def execute(**args):
                if scope is None:
                    raise ToolException("Dedicated repair task required")
                return await getattr(scope, name)(**args)

            return execute

        result.append(
            StructuredTool.from_function(
                coroutine=bind(name),
                name=name,
                description=description,
                args_schema=schema,
                response_format="content_and_artifact",
                handle_tool_error=True,
            )
        )
    return result


def repair_shell(scope, declaration):
    from langchain_core.messages import ToolMessage

    from agent_service.policy import active_call

    async def execute(command=None, restart=None, **injected):
        content, artifact = await scope.shell(command, restart)
        return ToolMessage(
            content=content,
            tool_call_id=active_call.get().call_id,
            name="shell",
            status="success" if artifact["exit_code"] == 0 else "error",
            artifact=artifact,
        )

    return StructuredTool.from_function(
        coroutine=execute,
        name="shell",
        description=declaration.description,
        args_schema=declaration.args_schema,
        handle_tool_error=True,
    )
