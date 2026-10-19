"""Second-batch contracts, durable PostgreSQL recovery and real external side effects."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx2
import psycopg
import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agent_service.background import monitor_operation
from agent_service.config import Settings
from agent_service.execution import DockerClient
from agent_service.execution_errors import EnvironmentUnavailable
from agent_service.failure import failure_info
from agent_service.mcp_config import Selection
from agent_service.profiles import Registry, RuntimeFile
from agent_service.reliability import (
    ReliabilitySettings,
    backoff,
    poll_interval,
    remaining_execution,
    transient_database,
    transient_observation,
)
from agent_service.runtime import RunContext
from agent_service.worker import Worker
from tests.mcp_tasks_service import TaskMCP
from tests.models import SyntheticModel
from tests.test_background import setup, task
from tests.test_core import execute
from tests.test_mcp_tasks import start
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install


@pytest.mark.parametrize(
    "values",
    [
        {"observation_max_failures": True},
        {"observation_max_failures": 1.0},
        {"observation_max_failures": 0},
        {"cancel_max_attempts": 4},
        {"cancel_max_attempts": False},
        {"maintenance_max_failures": "5"},
        {"observation_window": float("inf")},
        {"observation_window": float("nan")},
        {"publication_reserve": 0},
        {"observation_backoff_cap": 0.5},
        {"cleanup_backoff_cap": 7.0},
        {"maintenance_backoff_cap": 0.5},
        {"observation_window": -1.0},
        {"unknown_option": 1},
    ],
)
def test_strict_reliability_policy(values):
    with pytest.raises(ValidationError):
        ReliabilitySettings(**values)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 50])
@pytest.mark.parametrize("base,cap", [(1, 8), (8, 60)])
def test_backoff_exact_jitter_and_cap(n, base, cap):
    for jitter in (0.8, 1.0, 1.2):
        assert backoff(n, base, cap, uniform=lambda a, b: jitter) == min(
            cap, base * 2 ** min(n - 1, 30) * jitter
        )


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan"), 2**63, "20"])
def test_invalid_remote_interval(value):
    with pytest.raises(Exception, match="MCP_TASK_PROTOCOL_ERROR"):
        poll_interval(value)


def test_remote_interval_rounds_up_and_default_selection_keeps_sync_manifest():
    assert poll_interval(0.1) == 1
    assert poll_interval(20.0) == 20
    assert poll_interval(None) is None
    with pytest.raises(ValidationError):
        Selection(name="x", alias="x", access="write", cancel_retry_safe=True)
    assert Selection(name="x", alias="x", access="write").cancel_retry_safe is False


@pytest.mark.parametrize(
    "exc,expected",
    [
        (ConnectionResetError(), True),
        (httpx2.ReadTimeout("secret"), True),
        (EnvironmentUnavailable("Cannot connect to the Docker daemon"), True),
        (EnvironmentUnavailable("permission denied"), False),
        (ValueError("invalid JSON"), False),
        (KeyError("bug"), False),
        (psycopg.OperationalError("connection refused"), False),
        (ExceptionGroup("all", [ConnectionError(), TimeoutError()]), True),
        (ExceptionGroup("mixed", [ConnectionError(), ValueError()]), False),
    ],
)
def test_explicit_observation_classifier(exc, expected):
    assert transient_observation(exc) is expected


@pytest.mark.parametrize(
    "exc,expected",
    [
        (psycopg.errors.ConnectionFailure(), True),
        (psycopg.OperationalError("connection refused"), True),
        (psycopg.errors.InvalidPassword(), False),
        (psycopg.errors.InsufficientPrivilege(), False),
        (psycopg.errors.UndefinedColumn(), False),
        (psycopg.OperationalError("unclassified contract error"), False),
        (TimeoutError(), False),
    ],
)
def test_explicit_maintenance_classifier(exc, expected):
    assert transient_database(exc) is expected


async def reserved(r, *, safe=False):
    t, _ = await r.store.create_task("database contract")
    row = await r.store.claim(uuid4())
    op = await r.store.reserve_operation(
        row,
        "original",
        "mcp",
        "args",
        {"manifest": {"tool": {"cancel_retry_safe": safe}}},
    )
    return t, row, op


async def due(r, op, *, remote=False):
    async with r.pool.connection() as conn:
        await conn.execute(
            """UPDATE agent.background_operations SET next_poll_at=clock_timestamp(),
            remote_poll_not_before=CASE WHEN %s THEN clock_timestamp() ELSE remote_poll_not_before END
            WHERE id=%s""",
            (remote, op["id"]),
        )
    return await r.store.operation(op["task_id"], op["id"], raw=True)


async def claimed(r, op, *, remote=False):
    await due(r, op, remote=remote)
    items = await r.store.claim_operations()
    return next(item for item in items if item["id"] == op["id"])


@pytest.mark.parametrize("end", ["success", "count", "window"])
def test_durable_observation_budget_and_late_facts(database, end):
    async def check(r):
        t, row, op = await reserved(r)
        await r.store.release_monitor(op, 0)
        first = None
        for n in range(1, 6 if end == "count" else 3):
            op = await claimed(r, op)
            assert await r.store.begin_observation(op)
            await r.store.finish_observation(
                op, error=failure_info("BACKGROUND_OBSERVATION_TRANSIENT")
            )
            assert op["observation_failures"] == n
            if first is None:
                first = op["observation_first_failed_at"]
            assert op["observation_first_failed_at"] == first
            # A duplicate confirmation cannot debit or move its wakeup.
            wakeup = op["next_poll_at"]
            await r.store.finish_observation(
                op, error=failure_info("BACKGROUND_OBSERVATION_TRANSIENT")
            )
            assert op["observation_failures"] == n and op["next_poll_at"] == wakeup
            if n < 5:
                assert op["state"] == "starting" and op["cleanup_state"] == "pending"
                assert op["error_code"] is None and op["failure_info"] is None
                assert (await r.store.task(t["id"]))["status"] == "running"
            await r.store.release_monitor(op)
        if end == "window":
            async with r.pool.connection() as conn:
                await conn.execute(
                    """UPDATE agent.background_operations SET
                    observation_first_failed_at=clock_timestamp()-interval '30 seconds',
                    remote_poll_not_before=clock_timestamp()+interval '1 hour' WHERE id=%s""",
                    (op["id"],),
                )
            op = await claimed(r, op)
            assert not await r.store.begin_observation(op)
        elif end == "success":
            op = await claimed(r, op)
            assert await r.store.begin_observation(op)
            await r.store.finish_observation(op)
            assert op["observation_failures"] == 0
            assert op["observation_first_failed_at"] is None
            assert op["last_observation_error"] is None and op["last_observed_at"]
            await r.store.operation_update(
                op, state="succeeded", result={"known": True}, cleanup_state="released"
            )
            await r.store.finish(row["id"], row["lease_token"], error_code="TEST_END")
            return
        else:
            op = await claimed(r, op)
        assert (
            op["state"] == "unknown"
            and op["error_code"] == "BACKGROUND_OBSERVATION_EXHAUSTED"
        )
        assert (await r.store.task(t["id"]))["status"] == "failed"
        assert op["result"] is None and op["cleanup_state"] != "released"
        await r.store.operation_reconcile(
            op, state="succeeded", result={"late": True}, cleanup_state="released"
        )
        assert (await r.store.task(t["id"]))["status"] == "failed"
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 1
        assert op["consumed_run_id"] is None

    database(check)


def test_orphan_round_debited_once_and_remote_interval_survives_claim(database):
    async def check(r):
        _, _, op = await reserved(r)
        assert await r.store.begin_observation(op)
        await r.store.remote_poll(op, 60_000)
        start_at = op["observation_started_at"]
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.background_operations SET monitor_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (op["id"],),
            )
        fresh = (await r.store.claim_operations())[0]
        assert fresh["observation_failures"] == 1
        assert fresh["observation_first_failed_at"] == start_at
        assert not fresh["observation_inflight"]
        assert fresh["remote_poll_interval_ms"] == 60000
        assert not await r.store.begin_observation(fresh)
        await r.store.release_monitor(fresh)
        again = await claimed(r, fresh)
        assert again["observation_failures"] == 1
        assert not await r.store.begin_observation(again)
        assert (
            again["last_observation_error"]["code"]
            == "BACKGROUND_OBSERVATION_INTERRUPTED"
        )

    database(check)


@pytest.mark.parametrize("phase", ["begin", "finish", "cancel", "ack"])
def test_database_commit_ack_loss_confirms_full_recovery_write(
    database, monkeypatch, phase
):
    async def check(r):
        _, _, op = await reserved(r)
        original = r.store.locked
        lost = False
        armed = False

        @asynccontextmanager
        async def lose_once():
            nonlocal lost
            async with original() as conn:
                yield conn
            if armed and not lost:
                lost = True
                raise psycopg.OperationalError("commit acknowledgement lost")

        monkeypatch.setattr(r.store, "locked", lose_once)
        armed = phase == "begin"
        assert await r.store.begin_observation(op)
        started = op["observation_started_at"]
        armed = phase == "finish"
        await r.store.finish_observation(
            op, error=failure_info("BACKGROUND_OBSERVATION_TRANSIENT")
        )
        assert (
            op["observation_failures"] == 1 and op["observation_started_at"] == started
        )
        attempt = uuid4()
        armed = phase == "cancel"
        assert await r.store.reserve_cancel_attempt(op, attempt)
        assert op["cancel_attempts"] == 1
        armed = phase == "ack"
        await r.store.confirm_cancel_attempt(op, attempt)
        assert op["cancel_acknowledged_at"] and op["cancel_attempts"] == 1
        assert lost

    database(check)


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize(
    "crash", ["intent", "reserved", "transport", "ack", "protocol"]
)
def test_cancel_intent_attempt_ack_and_retry_authority(database, safe, crash):
    async def check(r):
        _, _, op = await reserved(r, safe=safe)
        now = datetime.now(timezone.utc)
        await r.store.operation_update(op, cancel_requested_at=now)
        attempt = uuid4()
        if crash != "intent":
            assert await r.store.reserve_cancel_attempt(op, attempt)
        if crash == "transport":
            await r.store.confirm_cancel_attempt(
                op, attempt, error=failure_info("MCP_CANCEL_TRANSPORT_FAILED")
            )
        elif crash == "ack":
            await r.store.confirm_cancel_attempt(op, attempt)
        elif crash == "protocol":
            await r.store.confirm_cancel_attempt(
                op, attempt, error=failure_info("MCP_TASK_PROTOCOL_ERROR")
            )
        await r.store.release_monitor(op, 0)
        fresh = await claimed(r, op)
        allowed = crash == "intent" or (safe and crash == "transport")
        assert await r.store.reserve_cancel_attempt(fresh, uuid4()) is bool(allowed)
        assert fresh["cancel_attempts"] == (0 if crash == "intent" else 1) + int(
            allowed
        )
        assert bool(fresh["cancel_acknowledged_at"]) is (crash == "ack")
        if safe and crash == "transport":
            for _ in range(5):
                await r.store.confirm_cancel_attempt(
                    fresh,
                    fresh["cancel_attempt_id"],
                    error=failure_info("MCP_CANCEL_TRANSPORT_FAILED"),
                )
                await r.store.reserve_cancel_attempt(fresh, uuid4())
            assert fresh["cancel_attempts"] == 3

    database(check)


@pytest.mark.parametrize("boundary", ["daemon", "inspect", "logs"])
def test_real_docker_transient_observation_never_restarts_command(
    database, tmp_path, monkeypatch, boundary
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        model = SyntheticModel(
            responses=[
                call(
                    "start_background_shell",
                    {"command": "echo once >> count; echo receipt"},
                ),
                AIMessage("waiting"),
            ]
        )
        run = await execute(r, Worker(r, model=model))
        assert run["status"] == "completed", run
        op = (await r.store.operations(t["id"], raw=True))[0]
        # Wait for the actual command; only observations are fault-injected.
        client = DockerClient()
        while (await client.inspect(op["container_id"]))["State"]["Running"]:
            await asyncio.sleep(0.02)
        original = DockerClient.call
        failures = 0

        async def flaky(self, *args, **kwargs):
            nonlocal failures
            match = (
                args[0]
                == {"daemon": "info", "inspect": "container", "logs": "logs"}[boundary]
            )
            if match and failures < 2:
                failures += 1
                raise EnvironmentUnavailable("Cannot connect to the Docker daemon")
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "call", flaky)
        for n in range(2):
            current = await claimed(r, op)
            await monitor_operation(r.store, r.registry, current)
            op = await r.store.operation(t["id"], op["id"], raw=True)
            assert op["observation_failures"] == n + 1 and op["state"] == "running"
            assert (await r.store.task(t["id"]))["status"] == "waiting_background"
        current = await claimed(r, op)
        await monitor_operation(r.store, r.registry, current)
        op = await r.store.operation(t["id"], op["id"], raw=True)
        assert op["state"] == "succeeded" and op["cleanup_state"] == "released"
        assert op["observation_failures"] == 0 and op["last_observed_at"]
        assert (tmp_path / "count").read_text() == "once\n"
        assert len(model.history) == 2

    database(check)


@pytest.mark.parametrize("method", ["server/discover", "tasks/get"])
def test_native_mcp_transport_recovers_without_business_replay(
    database, tmp_path, monkeypatch, method
):
    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(t["id"], raw=True))[0]
            original = __import__(
                "agent_service.mcp_runtime", fromlist=["http_transport"]
            ).http_transport
            failures = 0

            class Fault(httpx2.AsyncBaseTransport):
                def __init__(self):
                    self.inner = original()

                async def handle_async_request(self, request):
                    nonlocal failures
                    body = json.loads(request.content) if request.content else {}
                    if body.get("method") == method and failures < 2:
                        failures += 1
                        raise httpx2.ConnectError("temporary", request=request)
                    return await self.inner.handle_async_request(request)

                async def aclose(self):
                    await self.inner.aclose()

            monkeypatch.setattr("agent_service.mcp_runtime.http_transport", Fault)
            for n in range(2):
                current = await claimed(r, op, remote=True)
                await monitor_operation(r.store, r.registry, current)
                op = await r.store.operation(t["id"], op["id"], raw=True)
                assert op["observation_failures"] == n + 1, op
                assert op["state"] == "running" and op["cleanup_state"] == "pending", op
            service.jobs[op["remote_handle"]] = "completed"
            current = await claimed(r, op, remote=True)
            await monitor_operation(r.store, r.registry, current)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert final["state"] == "succeeded" and final["observation_failures"] == 0
            assert service.effect == 1 and len(service.calls) == 1

    database(check)


@pytest.mark.parametrize("safe", [False, True])
def test_native_cancel_lost_response_observes_before_retry(
    database, tmp_path, monkeypatch, safe
):
    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            registry = service.registry(tmp_path)
            server = registry.mcp.servers[0]
            selection = server.tools[0].model_copy(update={"cancel_retry_safe": safe})
            config = RuntimeFile(
                version=1,
                mcp=registry.mcp.model_copy(
                    update={
                        "servers": [server.model_copy(update={"tools": [selection]})]
                    }
                ),
                profiles=[registry.profiles["external"]],
            )
            install(r, Registry(config))
            t, _ = await r.store.create_task("cancel", profile="external")
            run = await execute(
                r,
                Worker(
                    r,
                    model=SyntheticModel(
                        responses=[
                            call("remote_write", {"value": 1}),
                            AIMessage("waiting"),
                        ]
                    ),
                ),
            )
            assert run["status"] == "completed", run
            op = (await r.store.operations(t["id"], raw=True))[0]
            await r.store.cancel_operation(t["id"], op["id"])
            original = __import__(
                "agent_service.mcp_runtime", fromlist=["http_transport"]
            ).http_transport
            sent = []

            class LostCancel(httpx2.AsyncBaseTransport):
                def __init__(self):
                    self.inner = original()

                async def handle_async_request(self, request):
                    body = json.loads(request.content) if request.content else {}
                    method = body.get("method")
                    sent.append(method)
                    if method == "tasks/cancel":
                        # Receipt is lost before the service acts. No cancellation is inferred.
                        raise httpx2.ReadError("lost", request=request)
                    return await self.inner.handle_async_request(request)

                async def aclose(self):
                    await self.inner.aclose()

            monkeypatch.setattr("agent_service.mcp_runtime.http_transport", LostCancel)
            for _ in range(5):
                current = await claimed(r, op, remote=True)
                await monitor_operation(r.store, r.registry, current)
                op = await r.store.operation(t["id"], op["id"], raw=True)
            assert op["cancel_attempts"] == (3 if safe else 1), {
                k: op[k]
                for k in (
                    "state",
                    "error_code",
                    "cancel_attempts",
                    "cancel_last_error",
                    "last_observation_error",
                )
            }
            assert op["cancel_acknowledged_at"] is None and op["cancel_sent_at"]
            assert sent.count("tasks/cancel") == (3 if safe else 1)
            for i, method in enumerate(sent):
                if method == "tasks/cancel":
                    assert "tasks/get" in sent[:i]
            assert service.effect == 1
            service.jobs[op["remote_handle"]] = "completed"
            current = await claimed(r, op, remote=True)
            await monitor_operation(r.store, r.registry, current)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                final["state"] == "succeeded"
                and final["cancel_acknowledged_at"] is None
            )

    database(check)


@pytest.mark.parametrize("resources", ["none", "docker", "mcp", "both"])
def test_effective_deadline_resource_reserves_and_shared_child(
    database, tmp_path, resources
):
    from agent_service.execution_config import ExecutionSettings
    from agent_service.profiles import Profile, Workspace
    import os

    async def check(r):
        tools = (
            ()
            if resources == "none"
            else ("read_file",)
            if resources == "docker"
            else ("remote_write",)
        )
        mcp = TaskMCP().registry(tmp_path).mcp if resources in ("mcp", "both") else None
        if resources == "both":
            tools = ("read_file", "remote_write")
        install(
            r,
            Registry(
                RuntimeFile(
                    version=1,
                    execution=ExecutionSettings(
                        image=os.environ["AGENT_TEST_EXECUTION_IMAGE"]
                    ),
                    **({"mcp": mcp} if mcp else {}),
                    workspaces=[Workspace(id="source", root=str(tmp_path))],
                    profiles=[Profile(id="budget", tools=tools)],
                )
            ),
        )
        await r.store.create_task(
            "budget",
            profile="budget",
            workspace_id="source" if "read_file" in tools else None,
        )
        worker = Worker(r, model=SyntheticModel())
        row = await r.store.claim(worker.id)
        captured = {}

        async def generate(item, scope):
            captured.update(item)
            raise TimeoutError()

        worker.generate = generate
        await worker.execute(row)
        reserve = (
            2
            + (10 if resources in ("docker", "both") else 0)
            + (10 if resources in ("mcp", "both") else 0)
        )
        assert (
            captured["deadline_at"] - captured["effective_deadline_at"]
        ).total_seconds() == reserve
        context = RunContext(
            run_id=str(row["id"]),
            deadline_at=row["deadline_at"],
            effective_deadline_at=row["effective_deadline_at"],
        )
        child = replace(context, agent_depth=1)
        assert child.effective_deadline_at == context.effective_deadline_at
        assert remaining_execution(child) <= 300 - reserve

    database(check)


def test_no_budget_stops_before_initialization_or_model(database, tmp_path):
    async def check(r):
        setup(r, tmp_path)
        await r.store.submit(
            "too short", profile="workspace_read", workspace_id="source"
        )
        model = SyntheticModel()
        result = await execute(r, Worker(r, model=model))
        assert result["error_code"] == "RUN_TIMEOUT" and not model.history
        resource = await r.store.execution_resource(result["id"])
        assert resource["state"] == "released" and resource["container_id"] is None

    database(check, Settings(run_timeout=5))


@pytest.mark.parametrize("outcome", ["recover", "exhaust", "permanent", "stop"])
def test_maintenance_owns_transport_budget_and_stops_admission(
    database, monkeypatch, outcome
):
    async def check(r):
        r.registry.reliability = ReliabilitySettings(
            maintenance_backoff_base=0.01,
            maintenance_backoff_cap=0.02,
            maintenance_window=0.2,
        )
        worker = Worker(r, model=SyntheticModel())
        await r.store.submit("do not claim during outage")
        attempts = 0
        original = r.store.maintain

        async def maintain():
            nonlocal attempts
            attempts += 1
            if outcome == "permanent":
                raise psycopg.errors.InsufficientPrivilege()
            if outcome == "recover" and attempts == 3:
                return await original()
            if outcome == "stop":
                worker.stopping.set()
            raise psycopg.errors.ConnectionFailure()

        monkeypatch.setattr(r.store, "maintain", maintain)
        if outcome in ("exhaust", "permanent"):
            with pytest.raises((RuntimeError, psycopg.errors.InsufficientPrivilege)):
                await worker.tick()
            assert attempts == (5 if outcome == "exhaust" else 1)
        else:
            await worker.tick()
            assert attempts == (3 if outcome == "recover" else 1)
        assert len(worker.tasks) == (1 if outcome == "recover" else 0)
        for execution in worker.tasks.values():
            execution.cancel()
        await asyncio.gather(*worker.tasks.values(), return_exceptions=True)

    database(check)


def test_local_confirmed_lease_expires_during_database_outage(database, monkeypatch):
    async def check(r):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def reply(messages):
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

        await r.store.submit("lease guard")
        worker = Worker(r, model=SyntheticModel(reply=reply))
        row = await r.store.claim(worker.id)
        original = r.store.renew

        async def unavailable(*args, **kwargs):
            raise psycopg.errors.ConnectionFailure()

        monkeypatch.setattr(r.store, "renew", unavailable)
        execution = asyncio.create_task(worker.execute(row))
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(execution, 2)
        assert stopped.is_set()
        assert row["_lease_until"] == row["lease_until"]
        monkeypatch.setattr(r.store, "renew", original)
        assert await original(row["id"], row["lease_token"], with_lease=True) is None
        await r.store.maintain()
        current = await r.store.get(row["id"])
        assert (
            current["status"] == "failed" and current["error_code"] == "LEASE_EXPIRED"
        )

    database(check, Settings(heartbeat_seconds=0.03, lease_seconds=0.3))


def test_docker_close_uses_one_cleanup_budget(database, tmp_path, monkeypatch):
    from tests.test_execution import scope_for

    async def check(r):
        scope = await scope_for(r, tmp_path)
        original = r.store.resource_update

        async def fail(*args, **kwargs):
            await asyncio.sleep(10)
            return await original(*args, **kwargs)

        scope.settings = scope.settings.model_copy(update={"cleanup_timeout": 0.1})
        monkeypatch.setattr(r.store, "resource_update", fail)
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert not await scope.close()
        assert loop.time() - started < 0.18
        monkeypatch.setattr(r.store, "resource_update", original)
        scope.settings = scope.settings.model_copy(update={"cleanup_timeout": 10.0})
        assert await scope.close()

    database(check)


def test_core7_historical_cancel_migrates_conservatively(database):
    import os
    from importlib import import_module
    from pathlib import Path
    from unittest.mock import patch
    from alembic import command
    from alembic.config import Config
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    from db import migrate

    dsn = os.environ["MIGRATION_DATABASE_URL"]
    name = "agent_test_cancel_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    historical = make_conninfo(**{**conninfo_to_dict(dsn), "dbname": name})
    try:
        engine = create_engine(
            "postgresql+psycopg://",
            creator=lambda: psycopg.connect(historical),
            poolclass=NullPool,
        )
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql("CREATE SCHEMA agent")
                cfg = Config()
                cfg.set_main_option(
                    "script_location", str(Path("db/alembic").resolve())
                )
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "core_0006")
        finally:
            engine.dispose()
        tid, rid, oid, dispatch = (uuid4() for _ in range(4))
        binding = import_module("db.alembic.versions.core_0002").DEFAULT_BINDING
        from psycopg.types.json import Jsonb

        with psycopg.connect(historical, autocommit=True) as conn:
            conn.execute(
                """INSERT INTO agent.tasks(id,task,profile,submitted_binding,input_hash,status,deadline_at)
                VALUES(%s,'historical','chat',%s,'x','failed',clock_timestamp())""",
                (tid, Jsonb(binding)),
            )
            conn.execute(
                """INSERT INTO agent.runs(id,task,idempotency_scope,status,queue_deadline,submitted_binding,task_id,execution_no)
                VALUES(%s,'historical','standalone','failed',clock_timestamp(),%s,%s,1)""",
                (rid, Jsonb(binding), tid),
            )
            conn.execute(
                """INSERT INTO agent.background_operations
                (id,task_id,run_id,call_id,dispatch_id,backend,arguments_hash,binding,state,deadline_at,cancel_sent_at,result,error_code)
                VALUES(%s,%s,%s,'old',%s,'mcp','x',%s,'unknown',clock_timestamp(),clock_timestamp(),'{"kept":true}','HISTORICAL_CODE')""",
                (oid, tid, rid, dispatch, Jsonb({"original": "binding"})),
            )
            before = conn.execute(
                "SELECT binding,result,error_code,cancel_sent_at,cleanup_state,failure_info FROM agent.background_operations"
            ).fetchone()
        with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": historical}):
            migrate.main()
            migrate.main()
        with psycopg.connect(historical) as conn:
            assert migrate.validate_database(conn) == "core_0011"
            after = conn.execute(
                "SELECT binding,result,error_code,cancel_sent_at,cleanup_state,failure_info FROM agent.background_operations"
            ).fetchone()
            assert after == before
            assert conn.execute("""SELECT cancel_attempts,cancel_attempted_at=cancel_sent_at,
                cancel_acknowledged_at,cancel_attempt_id,observation_failures,observation_inflight
                FROM agent.background_operations""").fetchone() == (
                1,
                True,
                None,
                None,
                0,
                False,
            )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def test_first_remote_interval_and_public_diagnostics_are_readonly(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        service.poll_ms = 60_000
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(t["id"], raw=True))[0]
            assert op["remote_poll_interval_ms"] == 60000
            assert op["remote_poll_not_before"] > op["created_at"] + timedelta(
                seconds=59
            )
            assert not await r.store.claim_operations()
            public = (await r.store.operations(t["id"]))[0]
            assert public["observation"]["observation_failures"] == 0
            assert public["cancellation"]["cancel_attempts"] == 0
            assert "monitor_token" not in json.dumps(public, default=str)
            assert "cancel_attempt_id" not in json.dumps(public, default=str)
            assert "binding" not in public
            assert service.poll_count == 0 and service.cancel_count == 0
            assert (await r.store.operation(t["id"], op["id"], raw=True)) == op

    database(check)


def test_window_wakeup_does_not_query_before_remote_minimum(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(t["id"], raw=True))[0]
            async with r.pool.connection() as conn:
                await conn.execute(
                    """UPDATE agent.background_operations SET observation_failures=1,
                    observation_first_failed_at=clock_timestamp()-interval '31 seconds',
                    remote_poll_not_before=clock_timestamp()+interval '1 hour',next_poll_at=clock_timestamp()
                    WHERE id=%s""",
                    (op["id"],),
                )
            for current in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, current)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                final["state"] == "unknown"
                and final["error_code"] == "BACKGROUND_OBSERVATION_EXHAUSTED"
            )
            assert service.poll_count == 0 and service.effect == 1
            assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_maintenance_database_wait_is_interruptible(database, monkeypatch):
    async def check(r):
        started = asyncio.Event()
        worker = Worker(r, model=SyntheticModel())

        async def blocked():
            started.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(r.store, "maintain", blocked)
        running = asyncio.create_task(worker.maintain())
        await asyncio.wait_for(started.wait(), 1)
        worker.stopping.set()
        assert await asyncio.wait_for(running, 0.3) is False
        assert not worker.tasks

    database(check)


def test_real_postgres_maintenance_connection_loss_recovers(database, monkeypatch):
    import os
    from agent_service.storage import ADMISSION_LOCK

    async def check(r):
        worker = Worker(r, model=SyntheticModel())
        attempts = 0
        original = r.store.maintain

        async def counted():
            nonlocal attempts
            attempts += 1
            return await original()

        monkeypatch.setattr(r.store, "maintain", counted)
        # Hold the actual scheduler lock, terminate the backend while its maintenance
        # transaction is waiting, then let the real pool reconnect and finish.
        owner = await psycopg.AsyncConnection.connect(
            os.environ["MIGRATION_DATABASE_URL"], autocommit=True
        )
        async with owner:
            await owner.execute("SELECT pg_advisory_lock(%s)", (ADMISSION_LOCK,))
            running = asyncio.create_task(worker.maintain())
            try:
                for _ in range(200):
                    waiting = await (
                        await owner.execute("""SELECT pid FROM pg_stat_activity
                        WHERE datname=current_database() AND usename='agent_runtime'
                        AND wait_event_type='Lock' AND query LIKE 'SELECT pg_advisory_xact_lock%%'""")
                    ).fetchone()
                    if waiting:
                        await owner.execute(
                            "SELECT pg_terminate_backend(%s)", (waiting[0],)
                        )
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError(
                        "Maintenance did not reach the PostgreSQL lock"
                    )
            finally:
                await owner.execute("SELECT pg_advisory_unlock(%s)", (ADMISSION_LOCK,))
            assert await asyncio.wait_for(running, 5) is True
            assert attempts == 2 and not worker.tasks

    database(check)


@pytest.mark.parametrize("safe", [False, True])
def test_cancel_acknowledgement_does_not_imply_terminal_or_repeat(
    database, tmp_path, monkeypatch, safe
):
    from mcp.types import Result
    from fastmcp_tasks.models import CancelTaskParams

    async def check(r):
        service = TaskMCP()

        async def accept(context, params):
            service.cancel_count += 1
            return Result()  # Accepted request, task remains working.

        service.manager.app.add_request_handler(
            "tasks/cancel", CancelTaskParams, accept
        )
        async with service.serve(monkeypatch):
            registry = service.registry(tmp_path)
            server = registry.mcp.servers[0]
            selection = server.tools[0].model_copy(update={"cancel_retry_safe": safe})
            install(
                r,
                Registry(
                    RuntimeFile(
                        version=1,
                        mcp=registry.mcp.model_copy(
                            update={
                                "servers": [
                                    server.model_copy(update={"tools": [selection]})
                                ]
                            }
                        ),
                        profiles=[registry.profiles["external"]],
                    )
                ),
            )
            t, _ = await r.store.create_task("ack is not terminal", profile="external")
            run = await execute(
                r,
                Worker(
                    r,
                    model=SyntheticModel(
                        responses=[
                            call("remote_write", {"value": 1}),
                            AIMessage("waiting"),
                        ]
                    ),
                ),
            )
            assert run["status"] == "completed"
            op = (await r.store.operations(t["id"], raw=True))[0]
            await r.store.cancel_operation(t["id"], op["id"])
            for _ in range(3):
                current = await claimed(r, op, remote=True)
                await monitor_operation(r.store, r.registry, current)
                op = await r.store.operation(t["id"], op["id"], raw=True)
            assert op["state"] == "cancelling" and op["cleanup_state"] == "pending"
            assert op["cancel_acknowledged_at"] and op["cancel_attempts"] == 1
            assert service.cancel_count == 1
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET cancel_sent_at=clock_timestamp()-interval '31 seconds' WHERE id=%s",
                    (op["id"],),
                )
            before = service.poll_count
            current = await claimed(r, op)
            await monitor_operation(r.store, r.registry, current)
            op = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                op["state"] == "unknown"
                and op["error_code"] == "MCP_CANCEL_UNCONFIRMED"
            )
            assert service.poll_count == before and service.cancel_count == 1
            assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


@pytest.mark.parametrize("state", ["succeeded", "failed"])
def test_exhaustion_protects_confirmed_execution_and_retains_resource(database, state):
    async def check(r):
        t, _, op = await reserved(r)
        result = {"confirmed": state}
        code = "MCP_TASK_EXECUTION_FAILED" if state == "failed" else None
        await r.store.operation_update(op, state=state, result=result, error_code=code)
        async with r.pool.connection() as conn:
            await conn.execute(
                """UPDATE agent.background_operations SET observation_failures=5,
                observation_first_failed_at=clock_timestamp()-interval '30 seconds' WHERE id=%s""",
                (op["id"],),
            )
        assert not await r.store.begin_observation(op)
        assert (
            op["state"] == state and op["result"] == result and op["error_code"] == code
        )
        assert op["cleanup_state"] == "failed"
        assert (await r.store.task(t["id"]))["status"] == "failed"
        await r.store.release_monitor(op)
        fresh = await claimed(r, op)
        assert await r.store.begin_observation(fresh)
        await r.store.operation_cleanup(
            fresh, cleanup_state="released", cleanup_error=None
        )
        await r.store.finish_observation(fresh)
        assert fresh["state"] == state and fresh["result"] == result
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_missing_handle_does_not_connect_or_claim_successful_observation(
    database, monkeypatch
):
    async def check(r):
        t, _, op = await reserved(r)

        async def forbidden(*args):
            raise AssertionError("No original handle to observe")

        monkeypatch.setattr("agent_service.background.task_client", forbidden)
        await monitor_operation(r.store, r.registry, op)
        final = await r.store.operation(t["id"], op["id"], raw=True)
        assert final["state"] == "unknown"
        assert final["last_observed_at"] is None and final["observation_failures"] == 0
        assert final["cleanup_state"] == "pending"
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_disconnect_failure_cannot_hide_identity_failure():
    from agent_service.background import observation_connection
    from agent_service.external_operations import ExternalFailure

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise ConnectionResetError()

    async def check():
        with pytest.raises(ExternalFailure, match="MCP_TASK_IDENTITY_CHANGED"):
            async with observation_connection(Client()):
                raise ExternalFailure("MCP_TASK_IDENTITY_CHANGED")

    asyncio.run(check())


@pytest.mark.parametrize("status", [401, 403])
def test_native_mcp_permission_failure_is_not_transient(
    database, tmp_path, monkeypatch, status
):
    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            assert run["status"] == "completed"
            original = __import__(
                "agent_service.mcp_runtime", fromlist=["http_transport"]
            ).http_transport

            class Denied(httpx2.AsyncBaseTransport):
                def __init__(self):
                    self.inner = original()

                async def handle_async_request(self, request):
                    body = json.loads(request.content) if request.content else {}
                    if body.get("method") == "tasks/get":
                        return httpx2.Response(status, request=request)
                    return await self.inner.handle_async_request(request)

                async def aclose(self):
                    await self.inner.aclose()

            monkeypatch.setattr("agent_service.mcp_runtime.http_transport", Denied)
            op = (await r.store.operations(t["id"], raw=True))[0]
            current = await claimed(r, op, remote=True)
            await monitor_operation(r.store, r.registry, current)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert final["state"] == "unknown" and final["cancel_attempts"] == 0
            assert final["error_code"] != "BACKGROUND_OBSERVATION_EXHAUSTED"
            assert (await r.store.task(t["id"]))["status"] == "failed"
            assert service.effect == 1

    database(check)


def test_cancel_permission_binds_only_selected_tools(tmp_path):
    service = TaskMCP()
    registry = service.registry(tmp_path)
    server = registry.mcp.servers[0]
    original = server.tools[0]
    assert (
        "cancel_retry_safe" not in registry.mcp_tools[original.alias].manifest()["tool"]
    )
    changed = Registry(
        RuntimeFile(
            version=1,
            mcp=registry.mcp.model_copy(
                update={
                    "servers": [
                        server.model_copy(
                            update={
                                "tools": [
                                    original.model_copy(
                                        update={"cancel_retry_safe": True}
                                    )
                                ]
                            }
                        )
                    ]
                }
            ),
            profiles=[registry.profiles["external"]],
        )
    )
    assert registry.bind("chat") == changed.bind("chat")
    assert registry.bind("external") != changed.bind("external")


def test_background_deadline_event_never_polls_remote_early(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        service.poll_ms = 60_000
        async with service.serve(monkeypatch):
            t, run = await start(r, service, tmp_path)
            assert run["status"] == "completed"
            op = (await r.store.operations(t["id"], raw=True))[0]
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET deadline_at=clock_timestamp()-interval '1 second',next_poll_at=clock_timestamp() WHERE id=%s",
                    (op["id"],),
                )
            for current in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, current)
            final = await r.store.operation(t["id"], op["id"], raw=True)
            assert (
                final["state"] == "unknown"
                and final["error_code"] == "BACKGROUND_TIMEOUT"
            )
            assert service.poll_count == 0 and service.cancel_count == 0
            assert final["next_poll_at"] == final["remote_poll_not_before"]
            assert not await r.store.claim_operations()
            assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_orphan_exhaustion_and_task_failure_commit_together(database):
    async def check(r):
        t, _, op = await reserved(r)
        assert await r.store.begin_observation(op)
        async with r.pool.connection() as conn:
            await conn.execute(
                """UPDATE agent.background_operations SET observation_failures=4,
                observation_first_failed_at=clock_timestamp()-interval '1 second',
                monitor_until=clock_timestamp()-interval '1 second' WHERE id=%s""",
                (op["id"],),
            )
        fresh = (await r.store.claim_operations())[0]
        # No begin_observation/maintain is needed to see the atomic failure.
        assert fresh["observation_failures"] == 5 and fresh["state"] == "unknown"
        assert fresh["error_code"] == "BACKGROUND_OBSERVATION_EXHAUSTED"
        assert (await r.store.task(t["id"]))["status"] == "failed"
        assert not await r.store.begin_observation(fresh)
        assert fresh["cleanup_state"] != "released" and fresh["result"] is None

    database(check)


def test_lost_disconnect_after_terminal_cannot_restore_released_occupancy(database):
    async def check(r):
        _, _, op = await reserved(r)
        assert await r.store.begin_observation(op)
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.background_operations SET observation_failures=4 WHERE id=%s",
                (op["id"],),
            )
        await r.store.operation_update(
            op,
            state="succeeded",
            result={"authoritative": True},
            cleanup_state="released",
        )
        await r.store.finish_observation(
            op, error=failure_info("BACKGROUND_OBSERVATION_TRANSIENT")
        )
        assert op["state"] == "succeeded" and op["result"] == {"authoritative": True}
        assert op["cleanup_state"] == "released" and op["error_code"] is None
        assert op["observation_failures"] == 5
        await r.store.release_monitor(op)
        assert not await r.store.claim_operations()

    database(check)
