"""Run-owned native MCPAdapter connections with guarded dispatch and output."""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from mcp.types import InputRequiredResult, TextContent

from agent_service.external_operations import ExternalFailure, ExternalOperations
from agent_service.mcp_config import MAX_MESSAGE_BYTES, contract, digest
from agent_service.output import bounded_artifact, limit_output
from agent_service.policy import active_call


class LimitedStream(httpx2.AsyncByteStream):
    def __init__(self, stream, is_sse):
        self.stream, self.is_sse = stream, is_sse

    async def __aiter__(self):
        size = 0
        async for chunk in self.stream:
            size += len(chunk)
            # SDK 2.3 enforces max_sse_event_size=1 MiB before JSON parsing.
            # JSON bodies also need a bound before response.aread().
            if not self.is_sse and size > MAX_MESSAGE_BYTES:
                raise ExternalFailure("MCP_MESSAGE_TOO_LARGE")
            yield chunk

    async def aclose(self):
        await self.stream.aclose()


class BoundedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, transport):
        self.transport = transport

    async def handle_async_request(self, request):
        if len(await request.aread()) > MAX_MESSAGE_BYTES:
            raise ExternalFailure("MCP_MESSAGE_TOO_LARGE")
        response = await self.transport.handle_async_request(request)
        if (
            300 <= response.status_code < 400
            or response.headers.get("content-encoding", "identity").lower()
            != "identity"
        ):
            await response.aclose()
            raise ExternalFailure("MCP_PROTOCOL_ERROR")
        is_sse = response.headers.get("content-type", "").startswith(
            "text/event-stream"
        )
        if (
            not is_sse
            and hasattr(response, "_content")
            and len(response.content) > MAX_MESSAGE_BYTES
        ):
            await response.aclose()
            raise ExternalFailure("MCP_MESSAGE_TOO_LARGE")
        response.stream = LimitedStream(response.stream, is_sse)
        return response

    async def aclose(self):
        await self.transport.aclose()


def http_transport():
    return httpx2.AsyncHTTPTransport(retries=0, verify=True, trust_env=False)


def http_client_factory(headers=None, timeout=None, auth=None, **kwargs):
    return httpx2.AsyncClient(
        headers={**(headers or {}), "Accept-Encoding": "identity"},
        timeout=timeout or 30,
        auth=auth,
        follow_redirects=False,
        trust_env=False,
        transport=BoundedTransport(http_transport()),
    )


async def reject_elicitation(*args):
    from fastmcp.client.elicitation import ElicitResult

    return ElicitResult(action="decline")


async def ignore_remote_event(*args, **kwargs):
    return None


class RestrictedClient(Client):
    # Must be disabled before Client.__init__ folds its session capabilities.
    _auto_internal_extensions = False

    async def _drive_input_required(self, first, retry):
        if isinstance(first, InputRequiredResult):
            # The native driver's continuation would send tools/call again.
            raise ExternalFailure("MCP_INTERACTION_UNSUPPORTED")
        return first

    def redact(self, value):
        if isinstance(value, str):
            for secret in self.redactions:
                value = value.replace(secret, "[credential omitted]")
            return value
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    async def call_tool(self, name, arguments, **kwargs):
        result = await self.call_tool_mcp(name, arguments)
        parts = []
        for block in result.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
            else:
                parts.append(f"[{block.type} content omitted]")
        if result.structured_content is not None:
            parts.append(json.dumps(result.structured_content, ensure_ascii=False))
        text, truncated = limit_output(
            self.redact("\n".join(parts)), *self.output_limits
        )
        return SimpleNamespace(
            content=[TextContent(type="text", text=text)],
            is_error=result.is_error,
            structured_content=bounded_artifact(
                self.redact(result.structured_content), self.output_limits[0]
            )
            if result.structured_content is not None
            else ({"truncated": True} if truncated else None),
        )


def make_client(server, settings, output_limits):
    headers = {key: os.environ[value] for key, value in server.headers_env.items()}
    if server.bearer_env:
        headers["Authorization"] = "Bearer " + os.environ[server.bearer_env]
    if any("\n" in value or "\r" in value for value in headers.values()):
        raise ExternalFailure("MCP_CREDENTIALS_INVALID")
    transport = StreamableHttpTransport(
        server.url, headers=headers, httpx_client_factory=http_client_factory
    )
    client = RestrictedClient(
        transport,
        timeout=settings.call_timeout,
        init_timeout=settings.init_timeout,
        cache=False,
        extensions=[],
        mode="legacy",
        elicitation_handler=reject_elicitation,
        log_handler=ignore_remote_event,
        progress_handler=ignore_remote_event,
    )
    client.output_limits = output_limits
    client.redactions = tuple(
        sorted(
            {v for v in headers.values() if v}
            | ({os.environ[server.bearer_env]} if server.bearer_env else set()),
            key=len,
            reverse=True,
        )
    )
    return client


class MCPScope:
    def __init__(self, db, row, registry):
        self.db, self.row, self.registry = db, row, registry
        self.stack = AsyncExitStack()
        self.closing = False
        self.pending = set()
        self.seen = set()
        self.operations = ExternalOperations(db, row)
        self.settings = registry.mcp

    def remaining(self, maximum):
        seconds = (self.row["deadline_at"] - datetime.now(timezone.utc)).total_seconds()
        if seconds <= 0:
            raise TimeoutError("Run deadline expired")
        return min(maximum, seconds)

    async def prepare(self, profile):
        from agent_service.profiles import ConfigurationChanged

        selected = {
            name: self.registry.mcp_tools[name]
            for name in profile.tools
            if name in self.registry.mcp_tools
        }
        actual = {}
        init_deadline = asyncio.get_running_loop().time() + self.remaining(
            self.settings.init_timeout
        )
        for server_id in sorted({t.server.id for t in selected.values()}):
            entries = {n: t for n, t in selected.items() if t.server.id == server_id}
            server = next(iter(entries.values())).server
            try:
                async with asyncio.timeout_at(init_deadline):
                    client = make_client(
                        server,
                        self.settings,
                        (
                            self.registry.execution.output_bytes,
                            self.registry.execution.output_lines,
                        ),
                    )
                    adapter = await self.stack.enter_async_context(MCPAdapter(client))
                    found = await client.list_tools(cache_mode="bypass")
                    definitions = {t.name: contract(t) for t in found}
                    if len(definitions) != len(found) or any(
                        definitions.get(t.selection.name) != t.definition
                        for t in entries.values()
                    ):
                        raise ConfigurationChanged("Selected MCP tool contract changed")
                    native = {
                        t.name: t for t in await adapter.list_tools(cache_mode="bypass")
                    }
                    # list_tools re-discovers: verify that exact listing too, so a
                    # server cannot drift between verification and tool binding.
                    found_again = await client.list_tools(cache_mode="bypass")
                    if any(
                        {t.name: contract(t) for t in found_again}.get(t.selection.name)
                        != t.definition
                        for t in entries.values()
                    ):
                        raise ConfigurationChanged("Selected MCP tool contract changed")
                    for alias, entry in entries.items():
                        tool = native[entry.selection.name]
                        if (
                            tool.args_schema != entry.declaration().args_schema
                            or tool.description
                            != entry.definition.get("description", "")
                        ):
                            raise ConfigurationChanged(
                                "Adapted MCP tool contract changed"
                            )
                        actual[alias] = self.wrap(alias, entry, tool)
            except (ConfigurationChanged, asyncio.CancelledError):
                raise
            except Exception:
                raise ExternalFailure("MCP_CONNECTION_FAILED") from None
        return actual

    def wrap(self, alias, entry, native):
        async def invoke(**arguments):
            call = active_call.get()
            if self.closing or call is None:
                raise ExternalFailure("MCP_CONNECTION_CLOSED")
            write = entry.selection.access == "write"
            if write and (
                len(call.batch) != 1
                or call.batch[0].get("name") != alias
                or call.batch[0].get("id") != call.call_id
            ):
                return ToolMessage(
                    content='{"code":"MCP_WRITE_BATCH_REJECTED"}',
                    name=alias,
                    tool_call_id=call.call_id,
                    status="error",
                )
            if write and call.call_id in self.seen:
                raise ExternalFailure("MCP_REPLAY_DENIED")
            if write:
                self.seen.add(call.call_id)
                await self.operations.dispatch(call.call_id, entry, digest(arguments))
                self.pending.add(call.call_id)
            try:
                async with asyncio.timeout(self.remaining(self.settings.call_timeout)):
                    result = await native.ainvoke(
                        {
                            "type": "tool_call",
                            "id": call.call_id,
                            "name": native.name,
                            "args": arguments,
                        }
                    )
                if write:
                    state = "error" if result.status == "error" else "succeeded"
                    await self.operations.outcome(
                        call.call_id,
                        state,
                        {"content": result.content, "artifact": result.artifact},
                        "MCP_WRITE_ERROR" if state == "error" else None,
                    )
                    self.pending.discard(call.call_id)
                    if state == "error":
                        raise ExternalFailure("MCP_WRITE_ERROR")
                return result.model_copy(update={"name": alias})
            except asyncio.CancelledError:
                raise
            except ExternalFailure as exc:
                if write and call.call_id in self.pending:
                    await self.unknown(call.call_id)
                    raise ExternalFailure("MCP_RESULT_UNKNOWN") from None
                raise exc from None
            except Exception:
                if write:
                    await self.unknown(call.call_id)
                raise ExternalFailure(
                    "MCP_RESULT_UNKNOWN" if write else "MCP_CALL_FAILED"
                ) from None

        return StructuredTool(
            name=alias,
            description=native.description,
            args_schema=native.args_schema,
            coroutine=invoke,
            response_format="content",
            metadata={
                "external_access": entry.selection.access,
                "server_id": entry.server.id,
            },
        )

    async def unknown(self, call_id):
        await self.operations.outcome(call_id, "unknown", code="MCP_RESULT_UNKNOWN")
        self.pending.discard(call_id)

    async def close(self):
        self.closing = True
        deadline = asyncio.get_running_loop().time() + self.settings.cleanup_timeout
        try:
            try:
                async with asyncio.timeout_at(deadline):
                    records = await self.db.external_operations(self.row["id"])
                    self.pending.update(
                        r["call_id"] for r in records if r["state"] == "dispatched"
                    )
                    for call_id in tuple(self.pending):
                        await self.unknown(call_id)
            finally:
                # A database failure must not skip connection cleanup.
                async with asyncio.timeout_at(deadline):
                    await self.stack.aclose()
            return True
        except Exception:
            return False


class MCPBatchMiddleware(AgentMiddleware):
    def __init__(self, policy):
        self.writes = {
            n
            for n, t in policy.tools.items()
            if (t.metadata or {}).get("external_access") == "write"
        }

    async def awrap_tool_call(self, request, handler):
        call = active_call.get()
        if request.tool_call["name"] in self.writes and len(call.batch) != 1:
            return ToolMessage(
                content='{"code":"MCP_WRITE_BATCH_REJECTED"}',
                tool_call_id=call.call_id,
                name=request.tool_call["name"],
                status="error",
            )
        return await handler(request)
