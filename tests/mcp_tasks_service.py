"""Controlled Tasks server on the real MCP SDK, with no scheduler or external I/O."""

from uuid import uuid4
from datetime import datetime, timezone

from mcp.shared.exceptions import MCPError
from mcp.types import Result, CallToolResult, TextContent
from fastmcp_tasks.models import (
    CreateTaskResult,
    GetTaskResult,
    GetTaskParams,
    CancelTaskParams,
)
from tests.mcp_service import SyntheticMCP
from agent_service.profiles import Registry


class TaskMCP(SyntheticMCP):
    def __init__(self):
        super().__init__()
        self.jobs = {}
        self.status = "working"
        self.cancel_count = 0
        self.poll_count = 0
        self.poll_ms = 20
        self.expire = False
        self.complete_instead = False
        self.cancel_race = False
        self.manager.app.extensions = {"io.modelcontextprotocol/tasks": {}}
        self.manager.app.add_request_handler("tasks/get", GetTaskParams, self.get_task)
        self.manager.app.add_request_handler(
            "tasks/cancel", CancelTaskParams, self.cancel_task
        )

    def registry(self, tmp_path, **kwargs):
        base = super().registry(tmp_path, tools=("write",), **kwargs)
        config = base.mcp.model_copy(
            update={
                "servers": [
                    base.mcp.servers[0].model_copy(
                        update={
                            "tools": [
                                base.mcp.servers[0]
                                .tools[0]
                                .model_copy(update={"execution": "background"})
                            ]
                        }
                    )
                ]
            }
        )
        from agent_service.profiles import RuntimeFile, Profile

        return Registry(
            RuntimeFile(
                version=1,
                mcp=config,
                profiles=[
                    Profile(
                        id="external",
                        tools=(
                            "remote_write",
                            "background_status",
                            "background_result",
                            "cancel_background",
                        ),
                    )
                ],
            )
        )

    async def call_tool(self, context, params):
        self.calls.append((params.name, params.arguments))
        self.effect += 1
        self.started.set()
        if self.complete_instead:
            return CallToolResult(
                content=[TextContent(type="text", text="known immediate result")]
            )
        task_id = uuid4().hex
        self.jobs[task_id] = self.status
        return CreateTaskResult(**self.fields(task_id))

    def fields(self, task_id):
        return dict(
            task_id=task_id,
            status=self.jobs[task_id],
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated_at=datetime.now(timezone.utc).isoformat(),
            ttl_ms=60000,
            poll_interval_ms=self.poll_ms,
        )

    async def get_task(self, context, params):
        self.poll_count += 1
        if self.expire or params.task_id not in self.jobs:
            raise MCPError(code=-32602, message="Task expired")
        state = self.jobs[params.task_id]
        payload = (
            {
                "result": {
                    "content": [{"type": "text", "text": "native result"}],
                    "isError": False,
                }
            }
            if state == "completed"
            else {"error": {"code": -1, "message": "known failure"}}
            if state == "failed"
            else {"input_requests": {"one": {}}}
            if state == "input_required"
            else {}
        )
        return GetTaskResult(**self.fields(params.task_id), **payload)

    async def cancel_task(self, context, params):
        self.cancel_count += 1
        self.jobs[params.task_id] = "completed" if self.cancel_race else "cancelled"
        return Result()
