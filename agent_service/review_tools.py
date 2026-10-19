"""Fixed Review tools. No user-selected command, execution side, or comparison refs."""

from typing import Literal
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from agent_service.policy import active_call
from agent_service.workspace_tools import GitDiffInput


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DiffInput(Input):
    change_id: str | None = None
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class ReadInput(Input):
    side: Literal["base", "head"]
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=100, ge=1, le=200)
    _path = field_validator("path")(GitDiffInput.literal_path.__func__)


class ExecutionInput(Input):
    execution_id: str
    view: Literal["result", "log"] = "result"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class ReproductionInput(Input):
    hypothesis: str = Field(min_length=1, max_length=4096)
    script: str = Field(min_length=1, max_length=16384)

    @field_validator("script")
    @classmethod
    def byte_limit(cls, value):
        if len(value.encode()) > 16384:
            raise ValueError("Script exceeds 16 KiB")
        return value


def review_tools(scope=None):
    definitions = [
        (
            "review_diff",
            "Read fixed change list or one change patch. offset counts list/patch records, never source lines. Follow next_offset; no new refs accepted.",
            DiffInput,
        ),
        (
            "review_read_file",
            "Read fixed base/head text, including deleted files. For pagination set start_line=next_offset+1. Evidence covers only delivered complete lines.",
            ReadInput,
        ),
        (
            "read_review_execution",
            "Read this Run check/reproduction result or logs. Paginate using next_offset. Whole suite outcomes do not prove individual findings.",
            ExecutionInput,
        ),
        (
            "run_review_reproduction",
            "Run a temporary Python script on head only, read-only project and isolated container. Define reproduce() returning {observed: bool, details: str}. Unhandled exceptions are inconclusive. Script is a model-written observation, not proof of introduction.",
            ReproductionInput,
        ),
    ]
    result = []
    for name, description, schema in definitions:

        def bind(name):
            async def execute(**args):
                if scope is None:
                    raise ToolException("Review task required")
                if name == "review_diff":
                    return await scope.evidence.diff(**args)
                if name == "review_read_file":
                    data = await scope.file_call(
                        name, {**args, "range": scope.manifest["review"]}
                    )
                    if "error" in data:
                        raise ToolException(data["error"])
                    return await scope.evidence.source(args, data)
                if name == "read_review_execution":
                    return await scope.read_execution(**args)
                call = active_call.get()
                if call is None:
                    raise ToolException("Trusted tool identity required")
                record = await scope.reproduce(call.call_id, **args)
                # The compact summary is actual execution evidence delivered to the model.
                return await scope.evidence.execution(record)

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
