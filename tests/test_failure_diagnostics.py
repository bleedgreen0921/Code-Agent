"""Batch one: trusted failures, immutable outcomes, safe observation and diagnostics."""

import asyncio
import io
import json
import logging
import sys
from uuid import uuid4

import psycopg
import pytest
from langchain.agents.middleware import (
    AgentMiddleware,
    ToolErrorMiddleware as NativeErrors,
)
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError
from psycopg.types.json import Jsonb

from agent_service.background import BackgroundReceipt, monitor_operation
from agent_service.execution_errors import CommandTimeout, EnvironmentLost
from agent_service.external_operations import ExternalFailure, ExternalOperations
from agent_service.failure import FailureInfo, SPECS, failure_info, is_control
from agent_service.logging import JsonFormatter, FIELDS
from agent_service.policy import HookContractError, ToolErrorMiddleware
from agent_service.worker import Worker, failure_code
from tests.models import SyntheticModel
from tests.test_profiles_policy import call, invoke
from agent_service.runtime import build_agent


RECEIPT = dict(
    exit_code=0,
    timed_out=False,
    cancelled=False,
    truncated=False,
    total_bytes=0,
    output="",
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", True),
        ("exit_code", 0.0),
        ("exit_code", "0"),
        ("timed_out", 0),
        ("timed_out", "false"),
        ("cancelled", 1),
        ("truncated", "false"),
        ("total_bytes", True),
        ("total_bytes", -1),
        ("total_bytes", 0.0),
        ("output", None),
        ("output", []),
        ("extra", 1),
    ],
)
def test_receipt_rejects_coercion_and_extra_fields(field, value):
    with pytest.raises(ValidationError):
        BackgroundReceipt.model_validate({**RECEIPT, field: value})


@pytest.mark.parametrize("field", list(RECEIPT))
def test_receipt_requires_every_fixed_image_field(field):
    with pytest.raises(ValidationError):
        BackgroundReceipt.model_validate(
            {k: v for k, v in RECEIPT.items() if k != field}
        )


def test_receipt_rejects_contradictions_and_broken_json():
    for raw in (b"{", json.dumps({**RECEIPT, "timed_out": True, "cancelled": True})):
        with pytest.raises(ValidationError):
            BackgroundReceipt.model_validate_json(raw)


def test_failure_contract_is_strict_immutable_bounded_and_uses_only_templates():
    for code in SPECS:
        info = FailureInfo.model_validate(failure_info(code))
        assert (
            len(info.message.encode()) <= 512
            and len(info.model_dump_json().encode()) <= 2048
        )
    info = FailureInfo.model_validate(failure_info("RUN_TIMEOUT"))
    with pytest.raises(ValidationError):
        info.code = "OTHER"
    for update in (
        {"version": True},
        {"version": 1.0},
        {"extra": 1},
        {"message": "中" * 171},
        {"code": "x" * 97},
    ):
        with pytest.raises(ValidationError):
            FailureInfo.model_validate(info.model_dump() | update)
    assert failure_info("remote secret TIMEOUT") == failure_info("EXECUTION_FAILED")
    assert failure_info("MCP_PROTOCOL_ERROR")["effect"] == "unknown"
    assert failure_info("FILE_VERSION_CONFLICT")["recovery"] == "reread"


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.CancelledError(),
        EnvironmentLost(),
        CommandTimeout(),
        TimeoutError(),
        HookContractError(),
        psycopg.OperationalError("secret"),
        ExceptionGroup("secret group", [ValueError(), EnvironmentLost()]),
    ],
)
def test_control_failures_survive_outer_hook_and_native_error_handler(exc):
    async def check():
        effects = []

        @tool
        async def action() -> str:
            """Exercise a real native tool boundary."""
            effects.append(1)
            raise exc

        class Swallow(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                try:
                    return await handler(request)
                except BaseException:
                    return ToolMessage("hidden", tool_call_id=request.tool_call["id"])

        for middleware in ((Swallow(),), (NativeErrors(lambda e, r: "hidden"),)):
            model = SyntheticModel(responses=[call("action"), AIMessage("forbidden")])
            with pytest.raises(BaseException) as caught:
                await invoke(build_agent(model, tools=[action], middleware=middleware))
            assert is_control(caught.value)
            assert len(model.history) == 1
        assert effects == [1, 1]

    asyncio.run(check())


@pytest.mark.parametrize(
    "kind",
    [Exception, RuntimeError, OSError, psycopg.Error, TimeoutError, HookContractError],
)
def test_recoverable_configuration_rejects_control_parents(kind):
    with pytest.raises(ValueError):
        ToolErrorMiddleware((kind,))


def test_group_classification_preserves_hook_and_persistence_codes():
    assert (
        failure_code(ExceptionGroup("private", [ValueError(), HookContractError()]))
        == "HOOK_CONTRACT_ERROR"
    )
    assert failure_code(psycopg.OperationalError("private")) == "PERSISTENCE_FAILED"


def test_actual_log_output_is_bounded_correlated_and_credential_safe():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    logger = logging.Logger("test")
    logger.addHandler(handler)
    try:
        raise ValueError("Authorization: Bearer secret-password")
    except ValueError:
        context = {k: "中" * 200 for k in FIELDS if k != "duration_seconds"}
        logger.error(
            "run_execution_failed",
            exc_info=True,
            extra={
                **context,
                "duration_seconds": 1.5,
                "arguments": "secret-args",
                "reported_usage": {str(i): "中" * 100 for i in range(100)},
                "delegation": [{"detail": "中" * 100} for _ in range(100)],
            },
        )
    raw = output.getvalue()
    assert (
        "secret" not in raw
        and "Authorization" not in raw
        and "raise ValueError" not in raw
    )
    assert len(raw.encode()) <= 16385
    payload = json.loads(raw)
    assert set(FIELDS) <= payload.keys() and payload["truncated"]
    assert all(
        len(payload[k].encode()) <= 256 for k in FIELDS if isinstance(payload[k], str)
    )
    if "exception" in payload:
        assert payload["exception"]["type"] == "ValueError"
        assert set(payload["exception"]["frames"][0]) == {"file", "function", "line"}

    # No message formatting, repr(), or exception stringification on formatter failure.
    class Bad:
        def __str__(self):
            raise RuntimeError("secret")

    record = logging.LogRecord(
        "test", logging.ERROR, __file__, 1, Bad(), (), sys.exc_info()
    )
    assert json.loads(JsonFormatter().format(record))["message"] == "log_event"


async def reserved(r):
    task, _ = await r.store.create_task("state facts")
    run = await r.store.claim(uuid4())
    op = await r.store.reserve_operation(run, "original", "mcp", "hash", {})
    return task, run, op


@pytest.mark.parametrize(
    "initial",
    [
        "starting",
        "running",
        "cancelling",
        "succeeded",
        "failed",
        "cancelled",
        "unknown",
    ],
)
def test_database_state_matrix_uses_current_record_and_protects_terminal_result(
    database, initial
):
    async def check(r):
        t, run, op = await reserved(r)
        states = (
            "starting",
            "running",
            "cancelling",
            "succeeded",
            "failed",
            "cancelled",
            "unknown",
        )
        for target in states:
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.tasks SET status='running',error_code=NULL,finished_at=NULL WHERE id=%s",
                    (t["id"],),
                )
                await conn.execute(
                    "UPDATE agent.runs SET status='running' WHERE id=%s", (run["id"],)
                )
                await conn.execute(
                    """UPDATE agent.background_operations SET state=%s,result=NULL,error_code=NULL,
                    failure_info=NULL,cancel_requested_at=NULL,finished_at=NULL WHERE id=%s""",
                    (initial, op["id"]),
                )
            expected = target
            if initial == "cancelling" and target in ("starting", "running"):
                expected = "cancelling"
            permitted = (
                initial == "starting"
                or initial == "running"
                and target != "starting"
                or initial == "cancelling"
                or initial == target
            )
            if permitted:
                after = await r.store.operation_update(op, state=target)
                assert after["state"] == expected
            else:
                with pytest.raises(ExternalFailure):
                    await r.store.operation_update(op, state=target)
                assert (await r.store.operation(t["id"], op["id"], raw=True))[
                    "state"
                ] == initial
        if initial in ("succeeded", "failed", "cancelled", "unknown"):
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET state=%s WHERE id=%s",
                    (initial, op["id"]),
                )
            for field, value in (
                ("result", {"forged": True}),
                ("error_code", "OTHER"),
                ("failure_info", failure_info()),
            ):
                with pytest.raises(ExternalFailure):
                    await r.store.operation_update(op, **{field: value})

    database(check)


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled"])
def test_late_authoritative_facts_never_resume_task_or_publish_checkpoint(
    database, terminal
):
    async def check(r):
        t, run, op = await reserved(r)
        await r.store.operation_update(
            op, state="unknown", error_code="MCP_RESULT_UNKNOWN"
        )
        # Failure must already be visible, before maintain/sweep.
        assert (await r.store.task(t["id"]))["status"] == "failed"
        with pytest.raises(ExternalFailure):
            await r.store.operation_update(op, state=terminal, result={"late": True})
        await r.store.operation_reconcile(
            op,
            state=terminal,
            result={"late": True},
            error_code=None,
            failure_info=None,
        )
        finish = op["finished_at"]
        events = await r.store.task_timeline(t["id"])
        await r.store.operation_reconcile(
            op,
            state=terminal,
            result={"late": True},
            error_code=None,
            failure_info=None,
        )
        assert (await r.store.task_timeline(t["id"])) == events
        assert op["finished_at"] == finish
        with pytest.raises(ExternalFailure):
            await r.store.operation_update(op, state=terminal, result={"late": False})
        with pytest.raises(ExternalFailure):
            await r.store.operation_update(op, state=terminal)
        await r.store.maintain()
        current = await r.store.task(t["id"])
        assert current["status"] == "failed" and current["last_success_run_id"] is None
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 1
        assert op["consumed_run_id"] is None
        assert (await r.store.get(run["id"]))["final_checkpoint"] is None

    database(check)


def test_identity_cleanup_and_confirmation_are_independent(database):
    async def check(r):
        t, run, op = await reserved(r)
        await r.store.operation_update(
            op, remote_handle="original-handle", receipt={"taskId": "original-handle"}
        )
        for change in (
            {"remote_handle": "replacement"},
            {"receipt": {"taskId": "replacement"}},
        ):
            with pytest.raises(ExternalFailure):
                await r.store.operation_update(op, **change)
        await r.store.operation_update(
            op,
            state="failed",
            result={"effect": "written"},
            error_code="MCP_WRITE_ERROR",
        )
        original = {
            k: op[k]
            for k in ("state", "result", "error_code", "failure_info", "finished_at")
        }
        await r.store.operation_cleanup(
            op, cleanup_state="failed", cleanup_error="RESOURCE_CLEANUP_FAILED"
        )
        await r.store.operation_cleanup(
            op, cleanup_state="released", cleanup_error=None
        )
        assert {k: op[k] for k in original} == original
        with pytest.raises(ExternalFailure):
            await r.store.operation_cleanup(op, cleanup_state="pending")
        with pytest.raises(ValueError):
            await r.store.operation_cleanup(op, error_code="OTHER")

    database(check)


def test_external_results_require_live_original_dispatch_and_exact_confirmation(
    database,
):
    from types import SimpleNamespace

    async def check(r):
        t, run, op = await reserved(r)
        journal = ExternalOperations(r.store, run)
        entry = SimpleNamespace(
            server=SimpleNamespace(id="s", account_id="a"),
            selection=SimpleNamespace(name="write"),
        )
        await journal.dispatch("external", entry, "hash")
        await journal.outcome(
            "external", "error", {"effect": "written"}, "MCP_WRITE_ERROR"
        )
        before = await r.store.external_operations(run["id"])
        await journal.outcome(
            "external", "error", {"effect": "written"}, "MCP_WRITE_ERROR"
        )
        assert before == await r.store.external_operations(run["id"])
        with pytest.raises(ExternalFailure):
            await journal.outcome(
                "external", "error", {"effect": "different"}, "MCP_WRITE_ERROR"
            )
        await journal.dispatch("late", entry, "hash")
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (run["id"],),
            )
        with pytest.raises(ExternalFailure):
            await journal.outcome("late", "succeeded", {"late": True})
        await r.store.maintain()
        assert (await r.store.task(t["id"]))["status"] == "failed"
        await journal.reconcile_original("late", "succeeded", {"late": True})
        assert (await r.store.task(t["id"]))["status"] == "failed"
        records = await r.store.external_operations(run["id"])
        assert records[0]["failure_info"]["effect"] == "possible_partial"

    database(check)


def test_unresolved_counts_before_limit_scopes_all_runs_and_uses_safe_snapshot(
    database, monkeypatch
):
    async def check(r):
        assert await r.store.unresolved() == dict(
            unknown_operations=0,
            unreleased_resources=0,
            oldest_age_seconds=None,
            truncated=False,
            reasons={},
            records=[],
        )
        t, run, op = await reserved(r)
        await r.store.operation_update(
            op, state="unknown", error_code="MCP_RESULT_UNKNOWN"
        )
        other, _ = await r.store.submit("second run")
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET status='failed',task_id=%s,execution_no=2 WHERE id=%s",
                (t["id"], other["id"]),
            )
            for i in range(55):
                await conn.execute(
                    """INSERT INTO agent.external_operations
                    (run_id,call_id,lease_token,dispatch_id,server_id,account_id,tool_name,arguments_hash,state,error_code)
                    VALUES(%s,%s,%s,%s,'s','a','write','h','unknown','Authorization Bearer secret')""",
                    (other["id"], str(i), uuid4(), uuid4()),
                )

        async def forbidden(*args, **kwargs):
            raise AssertionError("Read-only diagnostics attempted maintenance")

        monkeypatch.setattr(r.store, "maintain", forbidden)
        monkeypatch.setattr(r.store, "sweep", forbidden)
        task = await r.store.task(t["id"])
        summary = task["unresolved"]
        assert (
            summary["unknown_operations"] == 56 and summary["unreleased_resources"] == 1
        )
        assert len(summary["records"]) == 50 and summary["truncated"]
        assert summary["reasons"] == {"MCP_RESULT_UNKNOWN": 1, "LEGACY_FAILURE": 55}
        assert "secret" not in json.dumps(summary, default=str)
        assert (await r.store.run_detail(run["id"]))["unresolved"][
            "unknown_operations"
        ] == 1
        assert (await r.store.run_detail(other["id"]))["unresolved"][
            "unreleased_resources"
        ] == 0
        async with r.store.snapshot() as conn:
            before = await r.store.unresolved(task_id=t["id"], conn=conn)
            async with r.pool.connection() as writer:
                await writer.execute(
                    "UPDATE agent.external_operations SET state='succeeded' WHERE run_id=%s",
                    (other["id"],),
                )
            assert before == await r.store.unresolved(task_id=t["id"], conn=conn)
        assert (await r.store.unresolved(task_id=t["id"]))["unknown_operations"] == 1

    database(check)


@pytest.mark.parametrize(
    "invalid",
    [
        b"{",
        {**RECEIPT, "exit_code": True},
        {**RECEIPT, "output": 1},
        {**RECEIPT, "extra": 1},
        {k: v for k, v in RECEIPT.items() if k != "total_bytes"},
    ],
)
def test_invalid_real_background_receipt_cannot_succeed_or_infer_cancellation(
    database, tmp_path, monkeypatch, invalid
):
    from agent_service.execution import DockerClient
    from tests.test_background import setup, task
    from tests.test_core import execute

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        model = SyntheticModel(
            responses=[
                call("start_background_shell", {"command": "echo once >> effects"}),
                AIMessage("waiting"),
            ]
        )
        run = await execute(r, Worker(r, model=model))
        assert run["status"] == "completed"
        op = (await r.store.operations(t["id"], raw=True))[0]
        client = DockerClient()
        for _ in range(100):
            if not (await client.inspect(op["container_id"]))["State"]["Running"]:
                break
            await asyncio.sleep(0.02)
        await r.store.cancel_operation(t["id"], op["id"])
        original = DockerClient.call

        async def corrupt(self, *args, **kwargs):
            if args[0] == "logs":
                return (
                    0,
                    invalid
                    if isinstance(invalid, bytes)
                    else json.dumps(invalid).encode(),
                    b"",
                )
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "call", corrupt)
        for claimed in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, claimed)
        after = await r.store.operation(t["id"], op["id"], raw=True)
        assert after["state"] == "unknown" and after["cleanup_state"] == "released"
        assert after["error_code"] == "BACKGROUND_RECEIPT_INVALID"
        assert after["failure_info"]["effect"] == "unknown"
        assert (await r.store.task(t["id"]))["status"] == "failed"
        assert (tmp_path / "effects").read_text() == "once\n" and len(
            model.history
        ) == 2

    database(check)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_background_cleanup_failure_preserves_actual_execution_result(
    database, tmp_path, monkeypatch, exit_code
):
    from agent_service.execution import DockerClient
    from tests.test_background import setup, task
    from tests.test_core import execute

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell",
                            {"command": f"echo once >> effects; exit {exit_code}"},
                        ),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        op = (await r.store.operations(t["id"], raw=True))[0]
        client = DockerClient()
        for _ in range(100):
            if not (await client.inspect(op["container_id"]))["State"]["Running"]:
                break
            await asyncio.sleep(0.02)
        original = DockerClient.call

        async def fail_remove(self, *args, **kwargs):
            if args[0] == "rm":
                raise RuntimeError("Authorization secret command output")
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "call", fail_remove)
        for claimed in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, claimed)
        after = await r.store.operation(t["id"], op["id"], raw=True)
        assert after["state"] == ("failed" if exit_code else "succeeded")
        assert (
            after["cleanup_state"] == "failed"
            and "secret" not in after["cleanup_error"]
        )
        if exit_code:
            assert after["failure_info"]["effect"] == "possible_partial"
            assert after["error_code"] == "BACKGROUND_COMMAND_FAILED"
        else:
            assert after["failure_info"] is None and after["error_code"] is None
        protected = {
            k: after[k]
            for k in ("state", "result", "error_code", "failure_info", "finished_at")
        }
        await r.store.maintain()
        assert (await r.store.task(t["id"]))["status"] == "failed"
        monkeypatch.setattr(DockerClient, "call", original)
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.background_operations SET next_poll_at=clock_timestamp() WHERE id=%s",
                (op["id"],),
            )
        for claimed in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, claimed)
        final = await r.store.operation(t["id"], op["id"], raw=True)
        assert (
            final["cleanup_state"] == "released"
            and {k: final[k] for k in protected} == protected
        )
        assert (await r.store.task(t["id"]))["status"] == "failed"
        assert (tmp_path / "effects").read_text() == "once\n"

    database(check)


def test_failure_columns_enforce_database_object_bound_and_application_contract(
    database,
):
    from psycopg import sql

    async def check(r):
        t, run, op = await reserved(r)
        async with r.pool.connection() as conn:
            for table, key, identity in (
                ("runs", "id", run["id"]),
                ("background_operations", "id", op["id"]),
            ):
                statement = sql.SQL(
                    "UPDATE agent.{} SET failure_info=%s WHERE {}=%s"
                ).format(sql.Identifier(table), sql.Identifier(key))
                for bad in ([], "message", None, {"large": "中" * 1000}):
                    with pytest.raises(psycopg.errors.CheckViolation):
                        async with conn.transaction():
                            await conn.execute(statement, (Jsonb(bad), identity))
                await conn.execute(statement, (None, identity))
        with pytest.raises(ValidationError):
            await r.store.operation_update(op, failure_info={})

    database(check)


def test_tool_diagnosis_survives_unicode_output_trimming_and_native_argument_error():
    from agent_service.policy import ToolLifecycleMiddleware

    limits = ToolLifecycleMiddleware((512, 4))
    for code in SPECS:
        message = ToolMessage(
            "中" * 3000,
            tool_call_id="call",
            status="error",
            artifact={"body": "中" * 4000, "failure_info": failure_info(code)},
        )
        final = limits.limit_result(message)
        assert final.artifact["failure_info"] == failure_info(code)
        assert (
            len(final.content.encode())
            + len(json.dumps(final.artifact, ensure_ascii=False).encode())
            <= 512
        )

    async def check():
        effects = []

        @tool
        async def action(value: int) -> str:
            """Receive native validated input."""
            effects.append(value)
            return "done"

        model = SyntheticModel(
            responses=[call("action", {"value": []}), AIMessage("fixed")]
        )
        from langgraph.checkpoint.memory import InMemorySaver

        graph = build_agent(model, tools=[action], checkpointer=InMemorySaver())
        state = await invoke(graph)
        message = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        assert (
            message.status == "error"
            and message.artifact["failure_info"]["effect"] == "not_dispatched"
        )
        assert "value" in message.content and not effects
        saved = await graph.aget_state({"configurable": {"thread_id": "policy"}})
        assert saved.values["thread_tool_call_count"]["__all__"] == 1

    asyncio.run(check())


@pytest.mark.parametrize("mismatch", ["task_id", "is_error", "schema"])
def test_mcp_status_identity_and_sdk_result_are_verified_before_success(
    database, tmp_path, monkeypatch, mismatch
):
    from tests.mcp_tasks_service import TaskMCP
    from tests.test_mcp_tasks import start

    class Broken(TaskMCP):
        async def get_task(self, context, params):
            current = await super().get_task(context, params)
            if mismatch == "task_id":
                return current.model_copy(update={"task_id": "another-task"})
            if mismatch == "is_error":
                return current.model_copy(
                    update={"result": {"content": [], "isError": True}}
                )
            return current.model_copy(
                update={
                    "result": {"content": [], "structuredContent": {"invalid": True}}
                }
            )

    async def check(r):
        if mismatch == "schema":
            from tests import mcp_service

            original_definition = mcp_service.definition

            def output_definition(name):
                return original_definition(name).model_copy(
                    update={
                        "output_schema": {"type": "object", "required": ["expected"]}
                    }
                )

            monkeypatch.setattr(mcp_service, "definition", output_definition)
        service = Broken()
        service.status = "completed"
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            for claimed in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, claimed)
            op = (await r.store.operations(t["id"], raw=True))[0]
            assert op["state"] == ("failed" if mismatch == "is_error" else "unknown")
            assert op["failure_info"] is not None
            if mismatch == "schema":
                assert op["error_code"] == "MCP_TASK_PROTOCOL_ERROR"
            if mismatch == "task_id":
                assert op["error_code"] == "MCP_TASK_IDENTITY_CHANGED"
                assert service.cancel_count == 0
            assert service.effect == 1 and len(service.calls) == 1

    database(check)


def test_run_resources_keep_original_identity_and_count_separately_from_unknown(
    database, tmp_path
):
    async def check(r):
        t, run, op = await reserved(r)
        async with r.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO agent.execution_resources
                (run_id,lease_token,worker_id,workspace_root,container_name)
                VALUES(%s,%s,%s,%s,%s)""",
                (
                    run["id"],
                    run["lease_token"],
                    run["worker_id"],
                    str(tmp_path),
                    "test-original-" + uuid4().hex,
                ),
            )
        assert await r.store.resource_update(
            run["id"], run["lease_token"], daemon_id="original", container_id="a" * 64
        )
        assert (
            await r.store.resource_update(
                run["id"], run["lease_token"], daemon_id="replacement"
            )
            is None
        )
        assert (
            await r.store.resource_update(
                run["id"], run["lease_token"], container_id="b" * 64
            )
            is None
        )
        await r.store.operation_update(
            op, state="unknown", error_code="BACKGROUND_CONFIGURATION_CHANGED"
        )
        assert op["failure_info"]["effect"] == "unknown"
        summary = (await r.store.run_detail(run["id"]))["unresolved"]
        assert (
            summary["unknown_operations"] == 1 and summary["unreleased_resources"] == 2
        )
        assert len(summary["records"]) == 2
        resource = await r.store.execution_resource(run["id"])
        assert (
            resource["daemon_id"] == "original" and resource["container_id"] == "a" * 64
        )
        await r.store.resource_update(run["id"], run["lease_token"], state="released")
        assert (
            await r.store.resource_update(run["id"], run["lease_token"], state="active")
            is None
        )

    database(check)


def test_docker_identity_is_verified_before_stop_or_removal(
    database, tmp_path, monkeypatch
):
    from agent_service.execution import DockerClient
    from tests.test_background import setup, task
    from tests.test_core import execute

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell",
                            {"command": "echo once >> effects; sleep 3"},
                        ),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        op = (await r.store.operations(t["id"], raw=True))[0]
        await r.store.cancel_operation(t["id"], op["id"])
        original_inspect, original_call = DockerClient.inspect, DockerClient.call
        actions = []

        async def wrong_id(self, identity):
            item = await original_inspect(self, identity)
            return (
                {**item, "Id": "f" * 64}
                if identity == op["container_id"] and item
                else item
            )

        async def observe(self, *args, **kwargs):
            if args[0] in ("stop", "rm"):
                actions.append(args)
            return await original_call(self, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "inspect", wrong_id)
        monkeypatch.setattr(DockerClient, "call", observe)
        try:
            for claimed in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, claimed)
            after = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                after["state"] == "unknown"
                and after["error_code"] == "BACKGROUND_RESOURCE_IDENTITY_CHANGED"
            )
            assert not actions and (tmp_path / "effects").read_text() == "once\n"
        finally:
            monkeypatch.setattr(DockerClient, "inspect", original_inspect)
            monkeypatch.setattr(DockerClient, "call", original_call)

    database(check)


def test_historic_unknown_can_clean_original_resource_without_backfilling_diagnosis(
    database, tmp_path, monkeypatch
):
    from tests.mcp_tasks_service import TaskMCP
    from tests.test_mcp_tasks import start

    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            op = (await r.store.operations(t["id"], raw=True))[0]
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET state='unknown',error_code='MCP_RESULT_UNKNOWN',failure_info=NULL WHERE id=%s",
                    (op["id"],),
                )
            for claimed in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, claimed)
            after = await r.store.operation(t["id"], op["id"], raw=True)
            assert after["state"] == "unknown" and after["failure_info"] is None
            assert (
                after["error_code"] == "MCP_RESULT_UNKNOWN"
                and service.cancel_count == 1
            )
            assert (await r.store.task(t["id"]))["status"] == "failed"
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET next_poll_at=clock_timestamp() WHERE id=%s",
                    (op["id"],),
                )
            for claimed in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, claimed)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                final["state"] == "cancelled" and final["cleanup_state"] == "released"
            )
            assert (await r.store.task(t["id"]))["status"] == "failed"
            assert service.effect == 1 and service.cancel_count == 1

    database(check)
