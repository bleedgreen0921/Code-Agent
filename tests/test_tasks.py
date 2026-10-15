"""Persistent Tasks: real PostgreSQL, synthetic models, no external network."""

import asyncio
import json
from uuid import uuid4, UUID

import httpx
import psycopg
import pytest

from agent_service.app import app
from agent_service.config import Settings
from agent_service.worker import Worker
from contracts.errors import ApiError
from tests.models import SyntheticModel
from tests.test_core import execute


def test_task_dependencies_and_checkpoint_boundaries(database):
    async def check(r):
        one, _ = await r.store.create_task("first")
        two, _ = await r.store.create_task("second", depends_on=[UUID(one["task_id"])])
        assert two["status"] == "waiting_dependencies"
        assert (await r.store.runs())["items"] == []
        model = SyntheticModel()
        first = await execute(r, Worker(r, model=model))
        assert str(first["task_id"]) == one["task_id"]
        assert (await r.store.task(one["id"]))["status"] == "completed"
        second = await execute(r, Worker(r, model=model))
        assert str(second["task_id"]) == two["task_id"]
        assert second["previous_run_id"] is None
        assert second["id"] != first["id"]
        assert "synthetic answer" in json.dumps(
            [m.content for m in model.history[-1]], default=str
        )
        assert len((await r.store.runs(task_id=one["id"]))["items"]) == 1
        assert (await r.store.task(two["id"]))["answer"] == "synthetic answer"

    database(check)


def test_task_idempotency_race_dependency_order_and_validation(database):
    async def check(r):
        a, _ = await r.store.create_task("a")
        b, _ = await r.store.create_task("b")
        ids = [a["id"], b["id"]]
        rows = await asyncio.gather(
            *(
                r.store.create_task(
                    "goal", "same", depends_on=ids if n % 2 else ids[::-1]
                )
                for n in range(8)
            )
        )
        assert sum(not replay for _, replay in rows) == 1
        assert len({t["id"] for t, _ in rows}) == 1
        with pytest.raises(ApiError) as e:
            await r.store.create_task("changed", "same", depends_on=ids)
        assert e.value.status == 409
        for deps, status in [([uuid4()], 404), ([a["id"], a["id"]], 422)]:
            with pytest.raises(ApiError) as e:
                await r.store.create_task("bad", depends_on=deps)
            assert e.value.status == status
        await r.store.cancel_task(a["id"])
        with pytest.raises(ApiError) as e:
            await r.store.create_task("bad", depends_on=[a["id"]])
        assert e.value.status == 409
        await r.store.maintain()
        assert (await r.store.task(rows[0][0]["id"]))[
            "error_code"
        ] == "DEPENDENCY_FAILED"

    database(check)


def test_task_capacity_wait_does_not_create_runs_and_total_deadline(database):
    async def check(r):
        for n in range(r.settings.max_queued_runs):
            await r.store.submit(str(n))
        task, _ = await r.store.create_task("wait")
        await r.store.maintain()
        assert (await r.store.runs(task_id=task["id"]))["items"] == []
        async with r.pool.connection() as c:
            await c.execute(
                "UPDATE agent.tasks SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=%s",
                (task["id"],),
            )
        await r.store.maintain()
        assert (await r.store.task(task["id"]))["error_code"] == "TASK_TIMEOUT"

    database(check)


def test_task_run_failure_and_run_cancellation_end_task(database):
    async def check(r):
        task, _ = await r.store.create_task("fail")
        run = await r.store.claim(uuid4())
        assert await r.store.finish(
            run["id"], run["lease_token"], error_code="TEST_FAILURE"
        )
        assert (await r.store.task(task["id"]))["error_code"] == "TEST_FAILURE"
        task, _ = await r.store.create_task("cancel")
        await r.store.maintain()
        runs = (await r.store.runs(task_id=task["id"]))["items"]
        await r.store.cancel(runs[0]["id"])
        assert (await r.store.task(task["id"]))["status"] == "cancelled"

    database(check)


def test_task_single_run_under_concurrent_workers(database):
    async def check(r):
        task, _ = await r.store.create_task("once")
        rows = await asyncio.gather(*(r.store.claim(uuid4()) for _ in range(6)))
        assert sum(row is not None for row in rows) == 1
        assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1
        async with r.pool.connection() as conn:
            with pytest.raises(psycopg.errors.UniqueViolation):
                async with conn.transaction():
                    await conn.execute(
                        """INSERT INTO agent.runs(id,task,task_id,execution_no,idempotency_scope,status,queue_deadline,submitted_binding)
                        VALUES(%s,'duplicate',%s,2,'task','queued',clock_timestamp(),'{}'::jsonb)""",
                        (uuid4(), task["id"]),
                    )

    database(check)


def test_task_api_strict_models_filter_trace_and_lifecycle(database):
    async def check(r):
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            assert (
                await client.post("/v1/tasks", json={"task": "x", "extra": True})
            ).status_code == 422
            assert (
                await client.post(
                    "/v1/tasks", json={"task": "x", "depends_on": [str(uuid4())] * 2}
                )
            ).status_code == 422
            created = await client.post(
                "/v1/tasks", json={"task": "answer"}, headers={"Idempotency-Key": "api"}
            )
            assert created.status_code == 202, created.text
            task = created.json()
            assert (
                await client.post(
                    "/v1/tasks",
                    json={"task": "answer"},
                    headers={"Idempotency-Key": "api"},
                )
            ).status_code == 200
            run = await execute(r)
            path = "/v1/tasks/" + task["task_id"]
            assert (await client.get(path)).json()["status"] == "completed"
            assert (
                len(
                    (
                        await client.get("/v1/runs", params={"task_id": task["id"]})
                    ).json()["items"]
                )
                == 1
            )
            trace = (await client.get("/v1/runs/" + str(run["id"]) + "/trace")).json()
            assert trace["task_id"] == task["id"] and trace["execution_no"] == 1
            assert (await client.get(path + "/timeline")).json()["events"]
            assert (await client.get(path + "/operations")).json()["items"] == []
            assert (
                await client.get(path + "/operations/" + str(uuid4()))
            ).status_code == 404

    database(check)


def test_task_unfinished_capacity_and_dependency_output_bound(database):
    async def check(r):
        task, _ = await r.store.create_task("a")
        with pytest.raises(ApiError):
            await r.store.create_task("b")
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.tasks SET status='completed',answer=%s WHERE id=%s",
                ("界" * 60000, task["id"]),
            )
        child, _ = await r.store.create_task("child", depends_on=[task["id"]])
        result = await r.store.task_inputs(child["id"])
        assert result["truncated"]
        assert len(json.dumps(result, ensure_ascii=False).encode()) <= 32768

    database(check, Settings(max_unfinished_tasks=1))


def test_task_deadline_caps_run_and_failed_checkpoint_never_continues(database):
    async def check(r):
        task, _ = await r.store.create_task("limited")
        row = await r.store.claim(uuid4())
        assert row["deadline_at"] <= task["deadline_at"]
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (row["id"],),
            )
        await r.store.maintain()
        assert (await r.store.task(task["id"]))["error_code"] == "LEASE_EXPIRED"
        await r.store.maintain()
        assert len((await r.store.runs(task_id=task["id"]))["items"]) == 1

    database(check, Settings(task_timeout=20))
