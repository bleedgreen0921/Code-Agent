"""Official native Tasks client against an in-process real SDK HTTP service."""

import pytest
from langchain_core.messages import AIMessage

from agent_service.background import monitor_operation
from agent_service.worker import Worker
from tests.mcp_tasks_service import TaskMCP
from tests.models import SyntheticModel
from tests.test_background import poll
from tests.test_core import execute
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install


async def start(r, service, tmp_path):
    install(r, service.registry(tmp_path))
    task, _ = await r.store.create_task("native task", profile="external")
    run = await execute(
        r,
        Worker(
            r,
            model=SyntheticModel(
                responses=[call("remote_write", {"value": 1}), AIMessage("submitted")]
            ),
        ),
    )
    return task, run


def test_mcp_tasks_persist_handle_reconnect_and_unique_continuation(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(task["id"], raw=True))[0]
            assert (
                op["remote_handle"] in service.jobs
                and op["receipt"]["taskId"] == op["remote_handle"]
            )
            assert op["binding"]["manifest"]["account_id"] == "test-account"
            assert "synthetic-secret" not in str(op)
            assert (await r.store.task(task["id"]))["status"] == "waiting_background"
            service.jobs[op["remote_handle"]] = "completed"
            await poll(r, task["id"])
            result = await execute(
                r,
                Worker(r, model=SyntheticModel(responses=[AIMessage("native final")])),
            )
            assert result["status"] == "completed", result
            assert (await r.store.task(task["id"]))["answer"] == "native final"
            assert service.effect == 1 and service.poll_count >= 1
            discovers = [
                body
                for _, body, _ in service.requests
                if body.get("method") == "server/discover"
            ]
            assert len(discovers) >= 2

    database(check)


@pytest.mark.parametrize("race", [False, True])
def test_mcp_tasks_cancel_is_followed_by_status_on_new_connection(
    database, tmp_path, monkeypatch, race
):
    async def check(r):
        service = TaskMCP()
        service.cancel_race = race
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(task["id"]))[0]
            await r.store.cancel_operation(task["id"], op["id"])
            ops = await poll(r, task["id"])
            assert ops[0]["state"] == ("succeeded" if race else "cancelled"), ops
            assert service.cancel_count == 1 and service.poll_count >= 2
            assert (await r.store.task(task["id"]))["status"] == "queued"

    database(check)


def test_mcp_tasks_input_required_cancels_without_interactive_continuation(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        service.status = "input_required"
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            ops = await poll(r, task["id"])
            assert ops[0]["state"] == "failed"
            assert ops[0]["error_code"] == "MCP_INTERACTION_UNSUPPORTED"
            assert service.cancel_count == 1
            assert not any(
                body.get("method") == "tasks/update" for _, body, _ in service.requests
            )

    database(check)


@pytest.mark.parametrize(
    "problem", ["unsupported", "expired", "identity", "configuration", "immediate"]
)
def test_mcp_tasks_protocol_identity_and_known_immediate_result(
    database, tmp_path, monkeypatch, problem
):
    async def check(r):
        service = TaskMCP()
        if problem == "unsupported":
            service.manager.app.extensions = {}
        if problem == "immediate":
            service.complete_instead = True
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            if problem == "unsupported":
                assert run["error_code"] == "MCP_TASKS_UNSUPPORTED", run
                assert service.effect == 0
                return
            assert run["status"] == "completed", run
            op = (await r.store.operations(task["id"], raw=True))[0]
            if problem == "immediate":
                assert op["state"] == "failed" and "known immediate result" in str(
                    op["result"]
                )
                assert op["cleanup_state"] == "released" and service.effect == 1
                return
            if problem == "expired":
                service.expire = True
            elif problem == "identity":
                monkeypatch.setenv("TEST_MCP_TOKEN", "changed-account-secret")
            elif problem == "configuration":
                changed = service.registry(tmp_path)
                from agent_service.profiles import Registry, RuntimeFile

                cfg = changed.mcp.model_copy(
                    update={
                        "servers": [
                            changed.mcp.servers[0].model_copy(
                                update={"account_id": "new-account"}
                            )
                        ]
                    }
                )
                install(
                    r,
                    Registry(
                        RuntimeFile(
                            version=1, mcp=cfg, profiles=[changed.profiles["external"]]
                        )
                    ),
                )
            for item in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, item)
            await r.store.maintain()
            assert (await r.store.task(task["id"]))["status"] == "failed"
            after = await r.store.operation(task["id"], op["id"], raw=True)
            assert after["state"] == "unknown" and after["cleanup_state"] != "released"
            assert service.effect == 1
            if problem in ("identity", "configuration"):
                assert service.poll_count == 0

    database(check)


def test_mcp_tasks_poll_interval_respected_and_account_handle_api_isolation(
    database, tmp_path, monkeypatch
):
    async def check(r):
        service = TaskMCP()
        service.poll_ms = 10000
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            for op in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, op)
            assert await r.store.claim_operations() == []
            other, _ = await r.store.create_task("other")
            op = (await r.store.operations(task["id"]))[0]
            from contracts.errors import ApiError

            with pytest.raises(ApiError):
                await r.store.operation(other["id"], op["id"])
            await r.store.cancel_task(task["id"])
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET next_poll_at=clock_timestamp()"
                )
            # Restore short cadence for teardown while retaining the tested persisted delay.
            service.poll_ms = 20
            await poll(r, task["id"])

    database(check)


def test_mcp_tasks_handle_publication_ack_loss_retries_only_db(
    database, tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager
    import psycopg

    async def check(r):
        service = TaskMCP()
        original = r.store.locked
        armed = False
        lost = False

        @asynccontextmanager
        async def lost_commit():
            nonlocal lost
            async with original() as conn:
                yield conn
            if armed and not lost:
                lost = True
                raise psycopg.OperationalError("handle commit acknowledgement lost")

        update = r.store.operation_update

        async def arm(op, **changes):
            nonlocal armed
            armed = "remote_handle" in changes
            try:
                return await update(op, **changes)
            finally:
                armed = False

        monkeypatch.setattr(r.store, "locked", lost_commit)
        monkeypatch.setattr(r.store, "operation_update", arm)
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert run["status"] == "completed", run
            op = (await r.store.operations(task["id"], raw=True))[0]
            assert lost and op["remote_handle"] and service.effect == 1
            events = (await r.store.task_timeline(task["id"]))["events"]
            assert len([e for e in events if e["status"] == "operation_running"]) == 1
            service.jobs[op["remote_handle"]] = "completed"
            assert (await poll(r, task["id"]))[0]["state"] == "succeeded"

    database(check)


def test_mcp_tasks_lost_submission_response_is_unknown_and_never_resubmitted(
    database, tmp_path, monkeypatch
):
    import httpx2

    async def check(r):
        service = TaskMCP()

        def disconnect(request):
            raise httpx2.ReadError(
                "submission response lost after creating remote task"
            )

        service.fail_response = disconnect
        async with service.serve(monkeypatch):
            task, run = await start(r, service, tmp_path)
            assert (
                run["status"] == "failed"
                and run["error_code"] == "BACKGROUND_RESULT_UNKNOWN"
            ), run
            op = (await r.store.operations(task["id"], raw=True))[0]
            assert op["state"] == "unknown" and op["remote_handle"] is None
            assert op["cleanup_state"] != "released"
            assert (await r.store.task(task["id"]))["status"] == "failed"
            service.fail_response = None
            for item in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, item)
            assert service.effect == 1 and len(service.jobs) == 1
            assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1

    database(check)
