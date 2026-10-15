import asyncio
from dataclasses import replace
from uuid import uuid4

import httpx
import psycopg
import pytest
from agent_service.app import app
from agent_service.config import Settings
from agent_service.runtime import run_config
from agent_service.worker import Worker
from contracts.errors import ApiError
from langchain_core.messages import AIMessage

from tests.models import SyntheticModel


async def execute(resources, worker=None):
    worker = worker or Worker(resources, model=SyntheticModel())
    row = await resources.store.claim(worker.id)
    assert row
    await worker.execute(row)
    return await resources.store.get(row["id"])


def test_native_postgres_checkpoints_history_and_commit_ack_loss(database, monkeypatch):
    async def check(r):
        store = r.store
        model = SyntheticModel()
        worker = Worker(r, model=model)
        conv = await store.create_conversation("conversation")
        first, _ = await store.submit("first", conversation_id=conv["id"])
        original = store.finish
        finished = 0

        async def lost_ack(*args, **kwargs):
            nonlocal finished
            result = await original(*args, **kwargs)
            finished += 1
            if finished == 1:
                raise psycopg.OperationalError(
                    "Synthetic acknowledgement loss after commit"
                )
            return result

        monkeypatch.setattr(store, "finish", lost_ack)
        result = await execute(r, worker)
        assert result["status"] == "completed"
        assert len(model.history) == 1 and finished == 1
        final = await worker.agent.aget_state(result["final_checkpoint"])
        assert final.values["messages"][-1].text == result["answer"]
        assert (
            len(
                [
                    s
                    async for s in worker.agent.aget_state_history(
                        run_config(first["id"])
                    )
                ]
            )
            >= 3
        )
        second, _ = await store.submit("second", conversation_id=conv["id"])
        assert second["previous_run_id"] == first["id"]
        await execute(r, worker)
        assert [m.text for m in model.history[-1]] == [
            "first",
            "synthetic answer",
            "second",
        ]
        assert str(first["id"]) != str(second["id"])
        # A new worker/checkpointer reads the persisted checkpoint without replay.
        third, _ = await store.submit("third", conversation_id=conv["id"])
        other_model = SyntheticModel()
        await execute(r, Worker(r, model=other_model))
        assert [m.text for m in other_model.history[0]][-3:] == [
            "second",
            "synthetic answer",
            "third",
        ]
        other = await store.create_conversation("isolated")
        await store.submit("alone", conversation_id=other["id"])
        await execute(r, worker)
        assert [m.text for m in model.history[-1]] == ["alone"]

    database(check)


def test_failed_cancelled_and_lost_lease_turns_do_not_pollute_history(database):
    async def check(r):
        store = r.store
        conv = await store.create_conversation("history")
        good, _ = await store.submit("keep", conversation_id=conv["id"])
        await execute(r)

        def fail(messages):
            raise RuntimeError("Synthetic failure")

        await store.submit("failed input", conversation_id=conv["id"])
        bad = await execute(r, Worker(r, model=SyntheticModel(reply=fail)))
        assert bad["status"] == "failed"
        cancelled, _ = await store.submit("cancelled input", conversation_id=conv["id"])
        await store.cancel(cancelled["id"])
        await store.submit("lost input", conversation_id=conv["id"])
        worker = Worker(r, model=SyntheticModel())
        row = await store.claim(worker.id)
        # Persist intermediate graph state, then expire ownership.
        await worker.generate(row)
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (row["id"],),
            )
        assert not await store.finish(
            row["id"],
            row["lease_token"],
            answer="late",
            checkpoint={"configurable": {"thread_id": str(row["id"])}},
        )
        assert (await store.get(row["id"]))["error_code"] == "LEASE_EXPIRED"
        new, _ = await store.submit("next", conversation_id=conv["id"])
        assert new["previous_run_id"] == good["id"]
        model = SyntheticModel()
        await execute(r, Worker(r, model=model))
        assert [m.text for m in model.history[0]] == [
            "keep",
            "synthetic answer",
            "next",
        ]

    database(check)


def test_idempotency_session_conflict_and_global_capacity(database):
    async def check(r):
        store = r.store
        results = await asyncio.gather(
            *(store.submit("same", "same-key") for _ in range(8))
        )
        assert len({row["id"] for row, replay in results}) == 1
        assert sum(not replay for row, replay in results) == 1
        with pytest.raises(ApiError) as caught:
            await store.submit("different", "same-key")
        assert caught.value.status == 409
        conv = await store.create_conversation("busy")
        outcomes = await asyncio.gather(
            *(store.submit("turn", str(i), conv["id"]) for i in range(3)),
            return_exceptions=True,
        )
        assert (
            sum(
                isinstance(x, ApiError) and x.code == "CONVERSATION_BUSY"
                for x in outcomes
            )
            == 2
        )
        # Eight queued Runs are allowed, even under concurrent admission.
        rest = await asyncio.gather(
            *(store.submit(str(i)) for i in range(20)), return_exceptions=True
        )
        assert sum(not isinstance(x, Exception) for x in rest) == 6
        assert all(
            x.code == "QUEUE_CAPACITY_EXCEEDED" for x in rest if isinstance(x, ApiError)
        )
        claimed = await asyncio.gather(*(store.claim(uuid4()) for _ in range(6)))
        assert sum(x is not None for x in claimed) == 2
        await store.submit("ninth")
        await store.submit("tenth")
        with pytest.raises(ApiError):
            await store.submit("eleventh")

    database(check)


def test_queue_execution_timeout_cancel_race_and_worker_restart(database):
    async def check(r):
        store = r.store
        row, _ = await store.submit("expire queue")
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET queue_deadline=clock_timestamp()-interval '1 second' WHERE id=%s",
                (row["id"],),
            )
        assert await store.claim(uuid4()) is None
        assert (await store.get(row["id"]))["error_code"] == "QUEUE_TIMEOUT"
        row, _ = await store.submit("slow")
        started = asyncio.Event()
        calls = 0

        async def slow(messages):
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.sleep(30)
            return AIMessage("late")

        worker = Worker(r, model=SyntheticModel(reply=slow))
        claim = await store.claim(worker.id)
        work = asyncio.create_task(worker.execute(claim))
        await started.wait()
        await work
        assert (await store.get(row["id"]))["error_code"] == "RUN_TIMEOUT"
        # Running cancellation wins a concurrent success publication.
        row, _ = await store.submit("cancel running")
        claim = await store.claim(worker.id)
        await store.cancel(row["id"])
        await store.finish(
            row["id"], claim["lease_token"], answer="late", checkpoint={"x": 1}
        )
        assert (await store.get(row["id"]))["status"] == "cancelled"
        # Restart never reclaims an in-flight Run, even with native checkpoint data.
        row, _ = await store.submit("crashed")
        claim = await store.claim(worker.id)
        await store.cancel(row["id"])
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (row["id"],),
            )
        restarted = Worker(r, model=SyntheticModel(reply=slow))
        await restarted.tick()
        assert (await store.get(row["id"]))["error_code"] == "LEASE_EXPIRED"
        assert not restarted.tasks and calls == 1
        assert await store.renew(row["id"], claim["lease_token"]) is None

    cfg = replace(Settings(), run_timeout=0.1, heartbeat_seconds=0.02, lease_seconds=1)
    database(check, cfg)


def test_running_cancellation_stops_model_and_rejects_success_late(database):
    async def check(r):
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def slow(messages):
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                stopped.set()
            return AIMessage("late")

        worker = Worker(r, model=SyntheticModel(reply=slow))
        row, _ = await r.store.submit("cancel")
        claim = await r.store.claim(worker.id)
        work = asyncio.create_task(worker.execute(claim))
        await started.wait()
        await r.store.cancel(row["id"])
        await work
        assert stopped.is_set()
        assert (await r.store.get(row["id"]))["status"] == "cancelled"
        assert not await r.store.finish(
            row["id"], claim["lease_token"], answer="late", checkpoint={"x": 1}
        )

    database(check, replace(Settings(), heartbeat_seconds=0.01, lease_seconds=1))


def test_http_contract_removed_endpoints_trace_and_lifecycle(database):
    async def check(r):
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 123)),
            base_url="http://127.0.0.1",
        ) as client:
            for field, value in [("mode", "react"), ("tool_authorizations", [])]:
                assert (
                    await client.post("/v1/runs", json={"task": "hello", field: value})
                ).status_code == 422
            for path in ("/v1/documents", "/v1/memories", "/v1/evidence"):
                assert (await client.get(path)).status_code == 404
            assert (
                await client.get("/v1/runs", params={"status": "partial"})
            ).status_code == 422
            response = await client.post(
                "/v1/runs", json={"task": "hello"}, headers={"Idempotency-Key": "http"}
            )
            assert response.status_code == 202
            run_id = response.json()["run_id"]
            assert (
                await client.post(
                    "/v1/runs",
                    json={"task": "hello"},
                    headers={"Idempotency-Key": "http"},
                )
            ).status_code == 200
            await execute(r)
            body = (await client.get("/v1/runs/" + run_id)).json()
            assert body["result"] == {"answer": "synthetic answer"}
            trace = (await client.get("/v1/runs/" + run_id + "/trace")).json()
            assert trace["final_checkpoint"] and len(trace["checkpoints"]) >= 3
            assert (await client.get("/v1/runs/" + run_id + "/timeline")).json()[
                "events"
            ][-1]["status"] == "completed"
            assert len((await client.get("/v1/runs")).json()["items"]) == 1
            assert (await client.get("/v1/runs/summary")).status_code == 200
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 200
            worker = Worker(r, model=SyntheticModel())
            await r.store.heartbeat(worker.id)
            assert (await client.get("/v1/workers")).json()["online_count"] == 1
            assert (await client.get("/health/accepting")).status_code == 200
            conv = (await client.post("/v1/conversations")).json()["conversation_id"]
            assert len((await client.get("/v1/conversations")).json()["items"]) == 1
            response = await client.post(
                "/v1/conversations/" + conv + "/turns", json={"task": "turn"}
            )
            turn_id = response.json()["turn_id"]
            assert (
                await client.post(
                    "/v1/conversations/" + conv + "/turns",
                    json={"task": "x", "mode": "react"},
                )
            ).status_code == 422
            assert (await client.get("/v1/conversations/" + conv)).json()["turns"][0][
                "run_id"
            ] == turn_id
            assert (
                await client.get("/v1/conversations/" + conv + "/turns/" + turn_id)
            ).status_code == 200
            assert (
                await client.post(
                    "/v1/conversations/" + conv + "/turns/" + turn_id + "/cancel"
                )
            ).json()["status"] == "cancelled"
            assert (
                await client.get("/health", headers={"Origin": "https://example.com"})
            ).status_code == 403
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 1)),
            base_url="http://127.0.0.1",
        ) as client:
            assert (await client.get("/health")).status_code == 403

    database(check)


def test_postgres_conversation_summary_and_fresh_counters(database):
    async def check(r):
        conversation = await r.store.create_conversation("summary")

        def reply(messages):
            if "Context Extraction Assistant" in messages[0].text:
                return AIMessage("important prior context")
            return AIMessage("answer")

        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        await r.store.submit("old context " * 200, conversation_id=conversation["id"])
        await execute(r, worker)
        await r.store.submit("recent question", conversation_id=conversation["id"])
        second = await execute(r, worker)
        snapshot = await worker.agent.aget_state(second["final_checkpoint"])
        texts = [m.text for m in snapshot.values["messages"]]
        assert any("important prior context" in text for text in texts)
        assert "recent question" in texts
        assert "old context " * 200 not in texts
        assert snapshot.values["thread_model_call_count"] == 1
        # Third turn inherits the framework summary as ordinary message history.
        await r.store.submit("follow up", conversation_id=conversation["id"])
        third = await execute(r, worker)
        assert third["status"] == "completed"
        assert any("important prior context" in m.text for m in model.history[-1])

    database(
        check,
        replace(
            Settings(),
            summary_trigger_tokens=150,
            summary_keep_tokens=40,
            summary_trim_tokens=1000,
        ),
    )


def test_worker_scheduler_handles_multiple_runs_and_shutdown(database):
    async def check(r):
        worker = Worker(r, model=SyntheticModel())
        loop = asyncio.create_task(worker.run())
        rows = [(await r.store.submit(str(i)))[0] for i in range(3)]
        try:
            async with asyncio.timeout(5):
                while True:
                    states = [(await r.store.get(row["id"]))["status"] for row in rows]
                    if states == ["completed"] * 3:
                        break
                    await asyncio.sleep(0.01)
            assert (await r.store.workers())["online_count"] == 1
        finally:
            worker.stopping.set()
            await loop
        assert (await r.store.workers())["online_count"] == 0

    database(check)


def test_native_tool_quota_maps_to_failed_run(database):
    from langchain.tools import tool

    async def check(r):
        calls = 0

        @tool
        async def count() -> str:
            """Count a synthetic operation."""
            nonlocal calls
            calls += 1
            return "counted"

        def reply(messages):
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "count",
                        "args": {},
                        "id": str(len(messages)),
                        "type": "tool_call",
                    }
                ],
            )

        model = SyntheticModel(reply=reply)
        from agent_service.profiles import Registry, RuntimeFile, Profile

        registry = Registry(
            RuntimeFile(
                version=1, profiles=[Profile(id="synthetic", tools=("count",))]
            ),
            tools=[count],
        )
        r.registry = r.store.registry = registry
        await r.store.submit("loop", profile="synthetic")
        row = await execute(r, Worker(r, model=model, tools=[count]))
        assert row["status"] == "failed" and row["error_code"] == "QUOTA_EXCEEDED"
        assert calls == 10 and row["answer"] is None and row["final_checkpoint"] is None
        checkpoints = [cp async for cp in r.checkpointer.alist(run_config(row["id"]))]
        assert checkpoints

    database(check)


def test_running_cancelled_checkpoint_is_not_inherited(database):
    async def check(r):
        conversation = await r.store.create_conversation("cancelled history")
        await r.store.submit("successful", conversation_id=conversation["id"])
        await execute(r)
        pending, _ = await r.store.submit(
            "cancel this input", conversation_id=conversation["id"]
        )
        started = asyncio.Event()

        async def slow(messages):
            started.set()
            await asyncio.sleep(30)
            return AIMessage("late")

        worker = Worker(r, model=SyntheticModel(reply=slow))
        row = await r.store.claim(worker.id)
        task = asyncio.create_task(worker.execute(row))
        await started.wait()
        await r.store.cancel(pending["id"])
        await task
        assert (await r.store.get(pending["id"]))["status"] == "cancelled"
        assert [cp async for cp in r.checkpointer.alist(run_config(pending["id"]))]
        await r.store.submit("next", conversation_id=conversation["id"])
        model = SyntheticModel()
        await execute(r, Worker(r, model=model))
        assert [m.text for m in model.history[0]] == [
            "successful",
            "synthetic answer",
            "next",
        ]

    database(check, replace(Settings(), heartbeat_seconds=0.01, lease_seconds=1))
