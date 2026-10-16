import asyncio
import json
from uuid import uuid4
from dataclasses import replace

import httpx2
import psycopg
import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from mcp.types import AudioContent, CallToolResult, ImageContent, TextContent

from agent_service.mcp_config import MCPSettings, Selection, Server, contract
from agent_service.profiles import Extension, Profile, Registry, RuntimeFile
from agent_service.runtime import run_config
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.mcp_service import SyntheticMCP, definition
from tests.test_core import execute
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install


async def run_external(r, model, **kwargs):
    await r.store.submit("use selected external tool", profile="external", **kwargs)
    return await execute(r, Worker(r, model=model))


def test_mcp_native_http_handshake_binding_and_session_isolation(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        async with service.serve(monkeypatch):
            for _ in range(2):
                model = SyntheticModel(
                    responses=[call("remote_lookup", {"value": 3}), AIMessage("read")]
                )
                result = await run_external(r, model)
                assert result["status"] == "completed", result
                assert await r.store.execution_resource(result["id"]) is None
                message = next(
                    m for m in model.history[-1] if isinstance(m, ToolMessage)
                )
                assert '"value": 3' in message.content
                snapshot = result["runtime_config"]
                assert snapshot["mcp"][0]["server_id"] == "service"
                assert "synthetic-secret" not in json.dumps(snapshot)
        assert service.init_count == 2 and len(service.session_ids) == 2
        assert len(service.calls) == 2
        assert all(
            headers.get("authorization") == "Bearer synthetic-secret-must-not-persist"
            for method, _, headers in service.requests
        )
        assert await r.store.external_operations() == []
        assert r.registry.bind() == Registry().bind()
        assert all(
            not any(name.startswith("remote_") for name in r.registry.profiles[p].tools)
            for p in ("chat", "workspace_read", "workspace_write")
        )

    database(check)


def test_mcp_write_success_audit_and_no_rpc_replay_after_db_ack_loss(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        count = 0
        # Publication acknowledgement loss must never re-enter the graph.
        finish = r.store.finish

        async def lost_finish(*args, **kwargs):
            nonlocal count
            result = await finish(*args, **kwargs)
            count += 1
            if count == 1:
                raise psycopg.OperationalError("commit acknowledgement lost")
            return result

        monkeypatch.setattr(r.store, "finish", lost_finish)
        async with service.serve(monkeypatch):
            result = await run_external(
                r,
                SyntheticModel(
                    responses=[call("remote_write", {"value": 2}), AIMessage("written")]
                ),
            )
        assert result["status"] == "completed"
        ops = await r.store.external_operations(result["id"])
        assert len(ops) == 1 and ops[0]["state"] == "succeeded"
        assert ops[0]["arguments_hash"] and "value" not in ops[0]
        assert service.effect == 1 and len(service.calls) == 1
        timeline = await r.store.timeline(result["id"])
        assert "external_operation_succeeded" in [
            e["status"] for e in timeline["events"]
        ]
        assert timeline["external_operations"][0]["state"] == "succeeded"

    database(check)


@pytest.mark.parametrize("access", ["lookup", "write"])
def test_mcp_explicit_business_error_read_feedback_write_fatal(
    database, tmp_path, monkeypatch, access
):
    async def check(r):
        service = SyntheticMCP()

        async def fail(context, params):
            return CallToolResult(
                is_error=True, content=[TextContent(type="text", text="business error")]
            )

        service.handler = fail
        install(r, service.registry(tmp_path))
        model = SyntheticModel(
            responses=[call("remote_" + access, {"value": 1}), AIMessage("handled")]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        if access == "lookup":
            assert result["status"] == "completed"
            assert (
                next(m for m in model.history[-1] if isinstance(m, ToolMessage)).status
                == "error"
            )
        else:
            assert result["error_code"] == "MCP_WRITE_ERROR"
            assert (await r.store.external_operations(result["id"]))[0][
                "state"
            ] == "error"
            assert len(model.history) == 1

    database(check)


@pytest.mark.parametrize("failure", ["disconnect", "malformed", "oversize", "timeout"])
def test_mcp_write_after_remote_effect_unknown_never_resent(
    database, tmp_path, monkeypatch, failure
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path, call_timeout=0.1))
        if failure == "timeout":

            async def block():
                await service.release.wait()

            service.after_write = block
        elif failure == "disconnect":

            def disconnected(request):
                raise httpx2.ReadError("response lost")

            service.fail_response = disconnected
        elif failure == "malformed":
            service.fail_response = lambda request: httpx2.Response(
                200, headers={"content-type": "application/json"}, content=b"{bad"
            )
        else:
            service.fail_response = lambda request: httpx2.Response(
                200,
                headers={"content-type": "application/json"},
                content=b"x" * (1024 * 1024 + 1),
            )
        async with service.serve(monkeypatch):
            result = await run_external(
                r, SyntheticModel(responses=[call("remote_write", {"value": 1})])
            )
        assert result["error_code"] == "MCP_RESULT_UNKNOWN", result
        assert service.effect == 1
        ops = await r.store.external_operations(result["id"])
        assert len(ops) == 1 and ops[0]["state"] == "unknown"
        await r.store.maintain()
        assert service.effect == 1

    database(check)


def test_mcp_mixed_batch_rejects_write_but_runs_read(database, tmp_path, monkeypatch):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        batch = AIMessage(
            "",
            tool_calls=[
                *call("remote_write", {"value": 1}, id="write").tool_calls,
                *call("remote_lookup", {"value": 2}, id="read").tool_calls,
            ],
        )
        model = SyntheticModel(responses=[batch, AIMessage("done")])
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["status"] == "completed"
        assert service.calls == [("lookup", {"value": 2})] and service.effect == 0
        errors = [
            m
            for m in model.history[-1]
            if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert len(errors) == 1 and errors[0].tool_call_id == "write"
        assert "MCP_WRITE_BATCH_REJECTED" in errors[0].content
        assert not await r.store.external_operations(result["id"])

    database(check)


def test_mcp_hook_final_json_schema_and_write_batch_guard(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        registry = service.registry(tmp_path)

        class Rewrite(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                args = request.tool_call["args"]
                if args.get("value") == 1:
                    request = request.override(
                        tool_call={**request.tool_call, "args": {"value": 100}}
                    )
                else:
                    request.state["messages"][-1].tool_calls.clear()
                    request = request.override(
                        tool_call={**request.tool_call, "name": "remote_write"},
                        tool=r.registry.tools["remote_write"],
                    )
                return await handler(request)

        registry = Registry(
            RuntimeFile(
                version=1, mcp=registry.mcp, profiles=[registry.profiles["external"]]
            ),
            extensions={"external": Extension(middleware_factory=lambda: [Rewrite()])},
        )
        install(r, registry)
        model = SyntheticModel(
            responses=[call("remote_lookup", {"value": 1}), AIMessage("done")]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["status"] == "completed" and service.calls == []
        assert (
            "INVALID_TOOL_ARGUMENTS"
            in next(m for m in model.history[-1] if isinstance(m, ToolMessage)).content
        )

    database(check)


@pytest.mark.parametrize(
    "drift", ["missing", "description", "schema", "annotations", "output"]
)
def test_mcp_selected_catalog_drift_fails_before_model(
    database, tmp_path, monkeypatch, drift
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path, tools=("lookup",)))
        if drift == "missing":
            service.tools = []
        else:
            update = (
                {"description": "changed"}
                if drift == "description"
                else {"input_schema": {"type": "object"}}
                if drift == "schema"
                else {"annotations": None}
                if drift == "annotations"
                else {"output_schema": {"type": "object"}}
            )
            service.tools[0] = service.tools[0].model_copy(update=update)
        model = SyntheticModel()
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["error_code"] == "CONFIGURATION_CHANGED"
        assert model.history == [] and service.calls == []

    database(check)


@pytest.mark.parametrize("kind", ["structured", "binary", "large_text"])
def test_mcp_output_is_visible_bounded_and_no_binary_checkpoint(
    database, tmp_path, monkeypatch, kind
):
    async def check(r):
        service = SyntheticMCP()

        async def output(context, params):
            if kind == "structured":
                return CallToolResult(
                    content=[], structured_content={"answer": "structured only"}
                )
            if kind == "binary":
                return CallToolResult(
                    content=[
                        ImageContent(
                            type="image", data="QUJD" * 10000, mime_type="image/png"
                        ),
                        AudioContent(
                            type="audio", data="QUJD" * 10000, mime_type="audio/wav"
                        ),
                    ]
                )
            return CallToolResult(
                content=[TextContent(type="text", text="x" * 50000 + "\n" * 250)],
                structured_content={"huge": "y" * 40000},
            )

        service.handler = output
        install(r, service.registry(tmp_path, tools=("lookup",)))
        model = SyntheticModel(
            responses=[call("remote_lookup", {"value": 1}), AIMessage("done")]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["status"] == "completed", result
        message = next(m for m in model.history[-1] if isinstance(m, ToolMessage))
        assert (
            len(message.content.encode()) <= 16384
            and len(message.content.splitlines()) <= 200
        )
        assert len(json.dumps(message.artifact).encode()) <= 16384
        if kind == "structured":
            assert "structured only" in message.content
        if kind == "binary":
            assert (
                "image content omitted" in message.content
                and "audio content omitted" in message.content
            )
            cps = [cp async for cp in r.checkpointer.alist(run_config(result["id"]))]
            assert "QUJDQUJD" not in repr(cps)

    database(check)


@pytest.mark.parametrize("reason", ["cancel", "lease", "stop", "deadline"])
def test_mcp_write_cancel_lease_stop_deadline_keep_unknown(
    database, tmp_path, monkeypatch, reason
):
    async def check(r):
        service = SyntheticMCP()

        async def hold():
            await service.release.wait()

        service.after_write = hold
        install(r, service.registry(tmp_path))
        worker = Worker(
            r, model=SyntheticModel(responses=[call("remote_write", {"value": 1})])
        )
        await r.store.submit("write", profile="external")
        row = await r.store.claim(worker.id)
        # Keep the short test lease valid during connection setup.
        worker.settings = replace(worker.settings, heartbeat_seconds=0.02)
        if reason == "deadline":
            row["remaining"] = 0.2
        async with service.serve(monkeypatch):
            task = asyncio.create_task(worker.execute(row))
            await asyncio.wait_for(service.started.wait(), 3)
            if reason == "cancel":
                await r.store.cancel(row["id"])
            elif reason == "lease":
                async with r.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                        (row["id"],),
                    )
            elif reason == "stop":
                task.cancel()
            result = await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), 5
            )
            if reason == "stop":
                assert isinstance(result[0], asyncio.CancelledError)
        current = await r.store.get(row["id"])
        assert current["status"] == ("cancelled" if reason == "cancel" else "failed")
        assert service.effect == 1
        assert (await r.store.external_operations(row["id"]))[0]["state"] == "unknown"

    database(check)


def test_confirmed_mcp_write_survives_later_run_failure(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))

        async def reply(messages):
            if any(isinstance(m, ToolMessage) for m in messages):
                raise RuntimeError("later model failure")
            return call("remote_write", {"value": 1})

        async with service.serve(monkeypatch):
            result = await run_external(r, SyntheticModel(reply=reply))
        assert result["status"] == "failed"
        await r.store.maintain()
        assert (await r.store.external_operations(result["id"]))[0][
            "state"
        ] == "succeeded"
        assert service.effect == 1

    database(check)


def test_mcp_repeat_call_id_is_not_dispatched_twice(database, tmp_path, monkeypatch):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        async with service.serve(monkeypatch):
            result = await run_external(
                r,
                SyntheticModel(
                    responses=[
                        call("remote_write", {"value": 1}),
                        call("remote_write", {"value": 2}),
                    ]
                ),
            )
        assert result["error_code"] == "MCP_REPLAY_DENIED" and service.effect == 1
        assert len(await r.store.external_operations(result["id"])) == 1

    database(check)


@pytest.mark.parametrize("operation", ["dispatch", "outcome"])
def test_mcp_db_commit_ack_loss_retries_database_only(
    database, tmp_path, monkeypatch, operation
):
    async def check(r):
        from contextlib import asynccontextmanager

        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        worker = Worker(
            r,
            model=SyntheticModel(
                responses=[call("remote_write", {"value": 1}), AIMessage("done")]
            ),
        )
        await r.store.submit("write", profile="external")
        row = await r.store.claim(worker.id)
        if operation == "dispatch":
            locked = r.store.locked
            armed = True

            @asynccontextmanager
            async def ack_lost():
                nonlocal armed
                async with locked() as conn:
                    yield conn
                if armed:
                    armed = False
                    raise psycopg.OperationalError("dispatch commit ack lost")

            monkeypatch.setattr(r.store, "locked", ack_lost)
        else:
            original = psycopg.AsyncTransaction.__aexit__
            armed = True

            async def ack_lost(transaction, *args):
                nonlocal armed
                result = await original(transaction, *args)
                if armed and service.effect == 1 and not args[0]:
                    armed = False
                    raise psycopg.OperationalError("outcome commit ack lost")
                return result

            monkeypatch.setattr(psycopg.AsyncTransaction, "__aexit__", ack_lost)
        async with service.serve(monkeypatch):
            await worker.execute(row)
        result = await r.store.get(row["id"])
        assert result["status"] == "completed", result
        assert service.effect == 1 and len(service.calls) == 1
        assert (await r.store.external_operations(row["id"]))[0]["state"] == "succeeded"

    database(check)


def test_mcp_unreferenced_service_unavailable_does_not_connect(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))

        def forbidden():
            raise AssertionError("No connection for chat")

        monkeypatch.setattr("agent_service.mcp_runtime.http_transport", forbidden)
        await r.store.submit("chat")
        assert (await execute(r))["status"] == "completed"

    database(check)


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/tool.py",
        "http://remote.example/mcp",
        "https://u:p@example.com/mcp",
        "https://example.com/mcp?token=secret",
        "https://example.com/mcp#fragment",
    ],
)
def test_mcp_rejects_non_http_insecure_and_credential_urls(tmp_path, url):
    with pytest.raises(ValueError):
        Server(id="s", url=url, account_id="a", catalog=str(tmp_path / "catalog"))


def test_mcp_catalog_alias_budget_local_refs_and_unrelated_bindings(tmp_path):
    service = SyntheticMCP()
    registry = service.registry(tmp_path)
    assert registry.bind() == Registry().bind()
    declaration = registry.mcp_tools["remote_lookup"]
    copy = declaration.definition
    copy["description"] = "mutated"
    assert declaration.definition["description"] != "mutated"
    config = RuntimeFile(
        version=1, mcp=registry.mcp, profiles=[registry.profiles["external"]]
    )
    bad_server = config.mcp.servers[0].model_copy(
        update={"tools": [Selection(name="lookup", alias="read_file", access="read")]}
    )
    with pytest.raises(ValueError, match="Duplicate"):
        Registry(config.model_copy(update={"mcp": MCPSettings(servers=[bad_server])}))
    with pytest.raises(ValueError, match="Duplicate"):
        Registry(
            config.model_copy(
                update={
                    "mcp": MCPSettings(
                        servers=[registry.mcp.servers[0], registry.mcp.servers[0]]
                    )
                }
            )
        )
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    # Adding an unselected remote tool never changes the selected profile binding.
    first = registry.bind("external")
    catalog["tools"].append(contract(definition("unrelated")))
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    assert Registry(config).bind("external") == first
    catalog["tools"][0]["description"] = "x" * 33000
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="32 KiB"):
        Registry(config)
    catalog["tools"][0]["description"] = "short"
    catalog["tools"][0]["inputSchema"] = {"$ref": "https://example.invalid/schema"}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="local"):
        Registry(config)


def test_mcp_sigkill_orphan_dispatch_becomes_unknown_without_replay(database, tmp_path):
    import subprocess
    import sys

    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        submitted, _ = await r.store.submit("external crash", profile="external")
        script = r"""
import asyncio,sys
from pathlib import Path
from unittest.mock import patch
from agent_service.resources import Resources
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.mcp_service import SyntheticMCP
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install
from scripts.integration_pytest import no_real_http
class Patches:
    def __init__(self):self.items=[]
    def setenv(self,key,value):
        import os
        item=patch.dict(os.environ,{key:value});item.start();self.items.append(item)
    def setattr(self,name,value):
        item=patch(name,value);item.start();self.items.append(item)
async def main():
    root=Path(sys.argv[1]);service=SyntheticMCP()
    async def after():
        (root/'remote-effect').write_text('one write')
        await service.release.wait()
    service.after_write=after
    async with Resources() as r:
        install(r,service.registry(root))
        worker=Worker(r,model=SyntheticModel(responses=[call('remote_write',{'value':1})]))
        row=await r.store.claim(worker.id)
        async with service.serve(Patches()):await worker.execute(row)
with no_real_http():asyncio.run(main())
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(10):
                while not (tmp_path / "remote-effect").exists():
                    assert child.poll() is None
                    await asyncio.sleep(0.02)
            assert (await r.store.external_operations(submitted["id"]))[0][
                "state"
            ] == "dispatched"
            child.kill()
            await asyncio.to_thread(child.wait)
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (submitted["id"],),
                )
            model = SyntheticModel()
            worker = Worker(r, model=model)
            await worker.tick()
            assert model.history == [] and not worker.tasks
            assert (await r.store.get(submitted["id"]))["error_code"] == "LEASE_EXPIRED"
            assert (await r.store.external_operations(submitted["id"]))[0][
                "state"
            ] == "unknown"
            assert (tmp_path / "remote-effect").read_text() == "one write"
        finally:
            if child.poll() is None:
                child.kill()
                await asyncio.to_thread(child.wait)

    database(check)


def test_mcp_hook_cannot_turn_read_into_single_write_or_hide_failure(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()
        registry = service.registry(tmp_path)
        actual_write = None

        class Rewrite(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                nonlocal actual_write
                # Get the actual bound tool via the model policy capture below.
                request.state["messages"][-1].tool_calls[:] = call(
                    "remote_write", {"value": 1}
                ).tool_calls
                return await handler(
                    request.override(
                        tool_call={**request.tool_call, "name": "remote_write"},
                        tool=actual_write,
                    )
                )

            async def awrap_model_call(self, request, handler):
                nonlocal actual_write
                actual_write = next(
                    t for t in request.tools if t.name == "remote_write"
                )
                return await handler(request)

        install(
            r,
            Registry(
                RuntimeFile(
                    version=1,
                    mcp=registry.mcp,
                    profiles=[registry.profiles["external"]],
                ),
                extensions={
                    "external": Extension(middleware_factory=lambda: [Rewrite()])
                },
            ),
        )
        model = SyntheticModel(
            responses=[call("remote_lookup", {"value": 1}), AIMessage("done")]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["status"] == "completed" and service.effect == 0
        assert (
            "MCP_WRITE_BATCH_REJECTED"
            in next(m for m in model.history[-1] if isinstance(m, ToolMessage)).content
        )

    database(check)


def test_mcp_native_client_has_no_replay_interactions_or_extensions(tmp_path):
    async def check():
        from agent_service.mcp_runtime import make_client, reject_elicitation
        from agent_service.external_operations import ExternalFailure
        from mcp.types import InputRequiredResult

        registry = SyntheticMCP().registry(tmp_path)
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"TEST_MCP_TOKEN": "secret"}):
            client = make_client(registry.mcp.servers[0], registry.mcp, (16384, 200))
        assert client._session_kwargs["sampling_callback"] is None
        assert client._session_kwargs["list_roots_callback"] is None
        assert not client._auto_internal_extensions and client._response_cache is None
        assert not client._session_kwargs.get("extensions")
        assert not client._session_kwargs.get("result_claims")
        assert (await reject_elicitation()).action == "decline"
        retried = []

        async def forbidden(*args):
            retried.append(args)

        response = InputRequiredResult.model_construct(
            result_type="input_required", requests={}
        )
        with pytest.raises(ExternalFailure, match="MCP_INTERACTION_UNSUPPORTED"):
            await client._drive_input_required(response, forbidden)
        assert retried == []
        await client.close()

    asyncio.run(check())


def test_mcp_bounded_http_stream_rejects_before_buffering_and_redirects():
    async def check():
        from agent_service.mcp_runtime import BoundedTransport
        from agent_service.external_operations import ExternalFailure

        read = []

        class Stream(httpx2.AsyncByteStream):
            async def __aiter__(self):
                for n in range(100):
                    read.append(n)
                    yield b"x" * 65536

            async def aclose(self):
                pass

        class Transport(httpx2.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx2.Response(
                    200, headers={"content-type": "application/json"}, stream=Stream()
                )

        async with httpx2.AsyncClient(
            transport=BoundedTransport(Transport())
        ) as client:
            with pytest.raises(ExternalFailure, match="MCP_MESSAGE_TOO_LARGE"):
                await client.post("https://synthetic.invalid/mcp", json={})
        assert len(read) == 17
        requests = []

        async def redirect(request):
            requests.append(request)
            return httpx2.Response(
                307, headers={"location": "https://elsewhere.invalid/mcp"}
            )

        async with httpx2.AsyncClient(
            transport=BoundedTransport(httpx2.MockTransport(redirect)),
            follow_redirects=True,
        ) as client:
            with pytest.raises(ExternalFailure, match="MCP_PROTOCOL_ERROR"):
                await client.post(
                    "https://synthetic.invalid/mcp", headers={"Authorization": "secret"}
                )
        assert len(requests) == 1

    asyncio.run(check())


def test_mcp_cleanup_is_before_completed_publication(database, tmp_path, monkeypatch):
    async def check(r):
        service = SyntheticMCP()
        install(r, service.registry(tmp_path))
        from agent_service.mcp_runtime import MCPScope

        original = MCPScope.close
        closed = []

        async def close(scope, deadline=None):
            result = await original(scope, deadline)
            closed.append(result)
            return result

        finish = r.store.finish

        async def publish(*args, **kwargs):
            assert closed == [True]
            return await finish(*args, **kwargs)

        monkeypatch.setattr(MCPScope, "close", close)
        monkeypatch.setattr(r.store, "finish", publish)
        async with service.serve(monkeypatch):
            result = await run_external(
                r,
                SyntheticModel(
                    responses=[call("remote_lookup", {"value": 1}), AIMessage("done")]
                ),
            )
        assert result["status"] == "completed"

    database(check)


def test_mcp_database_cleanup_failure_still_closes_connections():
    async def check():
        from types import SimpleNamespace
        from agent_service.mcp_runtime import MCPScope

        closed = []

        async def broken(*args):
            raise psycopg.OperationalError("unavailable")

        db = SimpleNamespace(external_operations=broken)
        scope = MCPScope(db, {"id": uuid4()}, SimpleNamespace(mcp=MCPSettings()))
        scope.stack.callback(lambda: closed.append(True))
        assert await scope.close() is False
        assert scope.closing and closed == [True]

    asyncio.run(check())


def test_mcp_known_credentials_are_omitted_from_result_and_receipt(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = SyntheticMCP()

        async def echo(context, params):
            return CallToolResult(
                content=[
                    TextContent(type="text", text="synthetic-secret-must-not-persist")
                ],
                structured_content={"echo": "synthetic-secret-must-not-persist"},
            )

        service.handler = echo
        install(r, service.registry(tmp_path))
        model = SyntheticModel(
            responses=[call("remote_write", {"value": 1}), AIMessage("done")]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model)
        assert result["status"] == "completed"
        assert "synthetic-secret-must-not-persist" not in repr(
            await r.store.external_operations(result["id"])
        )
        messages = [m for m in model.history[-1] if isinstance(m, ToolMessage)]
        assert "synthetic-secret-must-not-persist" not in repr(messages)
        assert "credential omitted" in messages[0].content

    database(check)


def test_mcp_with_real_workspace_closes_both_resources_before_success(
    database, tmp_path, monkeypatch
):
    async def check(r):
        from tests.test_execution import install as install_workspace
        from agent_service.profiles import Workspace
        from agent_service.execution import DockerClient

        service = SyntheticMCP()
        remote = service.registry(tmp_path)
        install_workspace(r, tmp_path)
        install(
            r,
            Registry(
                RuntimeFile(
                    version=1,
                    execution=r.registry.execution,
                    workspaces=[Workspace(id="source", root=str(tmp_path))],
                    mcp=remote.mcp,
                    profiles=[
                        Profile(id="external", tools=("read_file", "remote_lookup"))
                    ],
                )
            ),
        )
        (tmp_path / "data.txt").write_text("local file")
        model = SyntheticModel(
            responses=[
                call("read_file", {"path": "data.txt"}, id="file"),
                call("remote_lookup", {"value": 1}, id="remote"),
                AIMessage("done"),
            ]
        )
        async with service.serve(monkeypatch):
            result = await run_external(r, model, workspace_id="source")
        assert result["status"] == "completed", result
        resource = await r.store.execution_resource(result["id"])
        assert resource["state"] == "released"
        assert await DockerClient().inspect(resource["container_id"]) is None
        assert len(service.calls) == 1

    database(check)
