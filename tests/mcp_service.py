"""Official SDK HTTP service over an in-process, network-denied test transport."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx2
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from agent_service.mcp_config import (
    MCPSettings,
    Selection,
    Server as ServerConfig,
    contract,
)
from agent_service.profiles import Profile, Registry, RuntimeFile

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer", "minimum": 1, "maximum": 10}},
    "required": ["value"],
    "additionalProperties": False,
}


def definition(name):
    return Tool(
        name=name,
        description=f"Synthetic {name}",
        input_schema=SCHEMA,
        annotations={"readOnlyHint": True},
    )


class SyntheticMCP:
    def __init__(self):
        self.tools = [definition("lookup"), definition("write")]
        self.calls = []
        self.requests = []
        self.effect = 0
        self.handler = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_response = None
        self.after_write = None
        self.init_count = 0
        self.session_ids = set()
        self.manager = StreamableHTTPSessionManager(
            Server(
                "synthetic", on_list_tools=self.list_tools, on_call_tool=self.call_tool
            ),
            json_response=True,
        )

    async def list_tools(self, context, params):
        return ListToolsResult(tools=self.tools)

    async def call_tool(self, context, params):
        self.calls.append((params.name, params.arguments))
        if params.name == "write":
            self.effect += 1
        self.started.set()
        if self.handler:
            return await self.handler(context, params)
        return CallToolResult(
            content=[TextContent(type="text", text="done")],
            structured_content={"value": params.arguments["value"]},
        )

    def registry(self, tmp_path, *, tools=("lookup", "write"), **settings):
        path = tmp_path / "catalog.json"
        path.write_text(
            json.dumps(
                {
                    "tools": [
                        contract(t) for t in [definition("lookup"), definition("write")]
                    ]
                }
            )
        )
        return Registry(
            RuntimeFile(
                version=1,
                mcp=MCPSettings(
                    servers=[
                        ServerConfig(
                            id="service",
                            url="http://127.0.0.1/mcp",
                            allow_insecure_loopback=True,
                            account_id="test-account",
                            catalog=str(path),
                            bearer_env="TEST_MCP_TOKEN",
                            tools=[
                                Selection(
                                    name=t,
                                    alias="remote_" + t,
                                    access="write" if t == "write" else "read",
                                )
                                for t in tools
                            ],
                        )
                    ],
                    **settings,
                ),
                profiles=[
                    Profile(id="external", tools=tuple("remote_" + t for t in tools))
                ],
            )
        )

    @asynccontextmanager
    async def serve(self, monkeypatch):
        service = self

        class Transport(httpx2.AsyncBaseTransport):
            def __init__(self):
                self.inner = httpx2.ASGITransport(app=service.manager.handle_request)

            async def handle_async_request(self, request):
                payload = json.loads(request.content) if request.content else {}
                service.requests.append(
                    (request.method, payload, dict(request.headers))
                )
                if payload.get("method") == "initialize":
                    service.init_count += 1
                # JSON-response servers need no event-stream GET. Returning 405
                # keeps the in-process ASGI transport from buffering an endless stream.
                if request.method == "GET":
                    return httpx2.Response(405)
                response = await self.inner.handle_async_request(request)
                if response.headers.get("mcp-session-id"):
                    service.session_ids.add(response.headers["mcp-session-id"])
                if (
                    payload.get("method") == "tools/call"
                    and payload.get("params", {}).get("name") == "write"
                ):
                    if service.after_write:
                        await service.after_write()
                    if service.fail_response:
                        return service.fail_response(request)
                return response

            async def aclose(self):
                await self.inner.aclose()

        monkeypatch.setenv("TEST_MCP_TOKEN", "synthetic-secret-must-not-persist")
        monkeypatch.setattr("agent_service.mcp_runtime.http_transport", Transport)
        async with self.manager.run():
            yield self
