import asyncio
import json

import httpx
import psycopg
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from typing_extensions import NotRequired

from agent_service.app import app
from agent_service.profiles import Extension, Profile, Registry, RuntimeFile, Workspace
from agent_service.runtime import RunContext
from agent_service.snapshot import safe_endpoint
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_core import execute
from tests.test_profiles_policy import call


def install(r, registry):
    r.registry = r.store.registry = registry


def config(root, prompt="first", tools=()):
    return RuntimeFile(
        version=1,
        workspaces=[Workspace(id="source", root=str(root))],
        profiles=[
            Profile(
                id="work", system_prompt=prompt, tools=tools, requires_workspace=True
            )
        ],
    )


def test_input_idempotency_versions_conversation_root_and_trace(database, tmp_path):
    async def check(r):
        install(r, Registry(config(tmp_path)))
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://127.0.0.1",
        ) as client:
            for payload in [
                {"task": "x", "profile": "missing"},
                {"task": "x", "workspace_id": "missing"},
                {"task": "x", "profile": "work"},
            ]:
                assert (await client.post("/v1/runs", json=payload)).status_code == 422
            for payload in [
                {"profile": "missing"},
                {"workspace_id": "missing"},
                {"profile": "work"},
            ]:
                assert (
                    await client.post("/v1/conversations", json=payload)
                ).status_code == 422
            headers = {"Idempotency-Key": "normalized"}
            original = await client.post(
                "/v1/runs", json={"task": "x"}, headers=headers
            )
            replay = await client.post(
                "/v1/runs",
                json={"task": "x", "profile": "chat", "workspace_id": None},
                headers=headers,
            )
            assert original.status_code == 202 and replay.status_code == 200
            for payload in [
                {"task": "other"},
                {"task": "x", "profile": "work", "workspace_id": "source"},
                {"task": "x", "workspace_id": "source"},
            ]:
                assert (
                    await client.post("/v1/runs", json=payload, headers=headers)
                ).status_code == 409
            await execute(r)
            response = await client.post(
                "/v1/runs",
                json={"task": "work", "profile": "work", "workspace_id": "source"},
                headers={"Idempotency-Key": "work"},
            )
            queued_id = response.json()["run_id"]
            install(r, Registry(config(tmp_path, "changed")))
            model = SyntheticModel()
            failed = await execute(r, Worker(r, model=model))
            assert (
                failed["error_code"] == "CONFIGURATION_CHANGED"
                and not model.history
                and failed["runtime_config"] is None
            )
            replay = await client.post(
                "/v1/runs",
                json={"task": "work", "profile": "work", "workspace_id": "source"},
                headers={"Idempotency-Key": "work"},
            )
            assert replay.status_code == 200 and replay.json()["run_id"] == queued_id
            conv = (
                await client.post(
                    "/v1/conversations",
                    json={"profile": "work", "workspace_id": "source"},
                )
            ).json()
            path = "/v1/conversations/" + conv["conversation_id"] + "/turns"
            assert (
                await client.post(path, json={"task": "x", "profile": "work"})
            ).status_code == 422
            assert (
                await client.post(path, json={"task": "x", "workspace_id": "source"})
            ).status_code == 422
            await client.post(path, json={"task": "first"})
            first = await execute(r, Worker(r, model=model))
            assert model.history[0][0].text == "changed"
            install(r, Registry(config(tmp_path, "new prompt")))
            await client.post(
                path, json={"task": "second"}, headers={"Idempotency-Key": "turn"}
            )
            second = await execute(r, Worker(r, model=model))
            assert second["submitted_binding"] != first["submitted_binding"]
            assert model.history[-1][0].text == "new prompt"
            trace = (
                await client.get("/v1/runs/" + str(second["id"]) + "/trace")
            ).json()
            assert trace["submitted_binding"] == second["submitted_binding"]
            assert trace["runtime_config"]["profile"] == "work"
            root = tmp_path / "replacement"
            root.mkdir()
            install(r, Registry(config(root)))
            assert (await client.post(path, json={"task": "third"})).status_code == 409
            # Replays use the original input, even after the workspace changes.
            assert (
                await client.post(
                    path, json={"task": "second"}, headers={"Idempotency-Key": "turn"}
                )
            ).status_code == 200
            install(r, Registry())
            replay = await client.post(
                "/v1/runs",
                json={"task": "work", "profile": "work", "workspace_id": "source"},
                headers={"Idempotency-Key": "work"},
            )
            assert replay.status_code == 200

    database(check)


def test_snapshot_commit_ack_loss_no_agent_replay_and_lease_guard(
    database, monkeypatch
):
    async def check(r):
        model = SyntheticModel()
        row, _ = await r.store.submit("once")
        record = r.store.record_runtime_config
        writes = 0

        async def lost_ack(*args, **kwargs):
            nonlocal writes
            ok = await record(*args, **kwargs)
            writes += 1
            if writes == 1:
                raise psycopg.OperationalError("ack lost")
            return ok

        monkeypatch.setattr(r.store, "record_runtime_config", lost_ack)
        result = await execute(r, Worker(r, model=model))
        assert (
            result["status"] == "completed" and len(model.history) == 1 and writes == 2
        )
        assert result["runtime_config"]["model"]["name"] == "synthetic"
        assert (
            await record(row["id"], result["lease_token"], result["runtime_config"])
            is False
        )
        assert (
            safe_endpoint(
                "https://user:password@example.com:443/v1/private-key?api_key=secret#key"
            )
            == "https://example.com:443"
        )
        monkeypatch.setattr(r.store, "record_runtime_config", record)
        row, _ = await r.store.submit("lost")
        worker = Worker(r, model=model)
        claim = await r.store.claim(worker.id)
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (row["id"],),
            )
        await worker.execute(claim)
        assert len(model.history) == 1
        assert (await r.store.get(row["id"]))["error_code"] == "LEASE_EXPIRED"

    database(check)


def test_native_structured_output_state_store_and_fresh_run_resources(
    database, tmp_path
):
    class State(AgentState):
        business: NotRequired[int]

    class Answer(BaseModel):
        value: int

    instances = []

    class Hook(AgentMiddleware):
        state_schema = State

        def __init__(self):
            self.invocations = 0
            instances.append(self)

        async def abefore_model(self, state, runtime):
            self.invocations += 1
            return {"business": state.get("business", 0) + 1}

    async def check(r):
        contexts = []

        @tool
        async def inspect(value: int, runtime: ToolRuntime[RunContext]) -> str:
            """Inspect synthetic runtime state and Store."""
            contexts.append(runtime.context)
            assert runtime.store is native_store
            assert runtime.context.deadline_at is not None
            assert runtime.context.workspace_root == str(tmp_path)
            await runtime.store.aput(("test",), "value", {"value": value})
            return str(value)

        native_store = InMemoryStore()
        install(
            r,
            Registry(
                config(tmp_path, tools=("inspect",)),
                tools=[inspect],
                extensions={
                    "work": Extension(
                        middleware_factory=lambda: (Hook(),),
                        state_schema=State,
                        response_format=ToolStrategy(Answer),
                    )
                },
            ),
        )
        conv = await r.store.create_conversation("business", "work", "source")
        model = SyntheticModel(
            responses=[
                call("inspect", {"value": 1}),
                call("Answer", {"value": 1}, "answer1"),
                call("inspect", {"value": 2}),
                call("Answer", {"value": 2}, "answer2"),
            ]
        )
        worker = Worker(r, model=model, store=native_store)
        for value in (1, 2):
            await r.store.submit(str(value), conversation_id=conv["id"])
            result = await execute(r, worker)
            assert result["status"] == "completed" and json.loads(result["answer"]) == {
                "value": value
            }
            final = await worker.agent.aget_state(result["final_checkpoint"])
            assert final.values["business"] == 2
            assert isinstance(final.values["messages"][-1], ToolMessage)
        assert len(model.history) == 4 and len(instances) == 2
        assert instances[0] is not instances[1] and [
            h.invocations for h in instances
        ] == [2, 2]
        assert contexts[0].run_id != contexts[1].run_id
        assert (await native_store.aget(("test",), "value")).value == {"value": 2}

    database(check)


def test_concurrent_runs_isolate_context_and_middleware(database, tmp_path):
    seen = []
    instances = []

    class Hook(AgentMiddleware):
        def __init__(self):
            self.context = None
            instances.append(self)

        async def abefore_model(self, state, runtime):
            if self.context is None:
                self.context = runtime.context
            assert self.context is runtime.context
            await asyncio.sleep(0.01)

    async def check(r):
        @tool
        async def inspect(runtime: ToolRuntime[RunContext]) -> str:
            """Inspect run context."""
            await asyncio.sleep(0.01)
            seen.append(runtime.context)
            return runtime.context.run_id

        install(
            r,
            Registry(
                config(tmp_path, tools=("inspect",)),
                tools=[inspect],
                extensions={"work": Extension(middleware_factory=lambda: (Hook(),))},
            ),
        )

        def reply(messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage("done")
            return call("inspect")

        worker = Worker(r, model=SyntheticModel(reply=reply))
        for task in ("a", "b"):
            await r.store.submit(task, profile="work", workspace_id="source")
        claims = [await r.store.claim(worker.id) for _ in range(2)]
        await asyncio.gather(*(worker.execute(row) for row in claims))
        assert len(seen) == 2 and len({c.run_id for c in seen}) == 2
        assert len(instances) == 2 and {h.context.run_id for h in instances} == {
            c.run_id for c in seen
        }
        assert [(await r.store.get(row["id"]))["status"] for row in claims] == [
            "completed",
            "completed",
        ]

    database(check)


def test_deleted_bindings_never_call_model_and_snapshots_exclude_credentials(
    database, tmp_path, monkeypatch
):
    from agent_service.runtime import model_from_resources
    from agent_service.snapshot import runtime_snapshot

    async def check(r):
        original = Registry(config(tmp_path))
        for replacement in [
            Registry(),
            Registry(RuntimeFile(version=1, profiles=[Profile(id="work")])),
        ]:
            install(r, original)
            await r.store.submit("queued", profile="work", workspace_id="source")
            install(r, replacement)
            model = SyntheticModel()
            row = await execute(r, Worker(r, model=model))
            assert row["error_code"] == "CONFIGURATION_CHANGED" and model.history == []
        monkeypatch.setenv("AGENT_MODEL_KEY", "SENSITIVE_MODEL_KEY")
        monkeypatch.setenv(
            "AGENT_MODEL_URL",
            "https://user:ENDPOINT_PASSWORD@example.com:443/private/API_PATH_SECRET?api_key=QUERY_SECRET",
        )
        monkeypatch.setenv("AGENT_MODEL", "snapshot-model")
        model = model_from_resources(r)
        registry = Registry()
        snapshot = runtime_snapshot(model, r.settings, registry, registry.bind(), ())
        encoded = json.dumps(snapshot)
        for secret in [
            "SENSITIVE_MODEL_KEY",
            "ENDPOINT_PASSWORD",
            "API_PATH_SECRET",
            "QUERY_SECRET",
            "AGENT_DATABASE_URL",
            "MIGRATION_DATABASE_URL",
        ]:
            assert secret not in encoded
        assert snapshot["model"]["endpoint"] == "https://example.com:443"

    database(check)


def test_tool_hook_cannot_swallow_external_cancellation(database, tmp_path):
    from dataclasses import replace
    from agent_service.config import Settings

    started = asyncio.Event()
    stopped = asyncio.Event()

    class Swallow(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            try:
                return await handler(request)
            except BaseException:
                return ToolMessage("swallowed", tool_call_id=request.tool_call["id"])

    async def check(r):
        @tool
        async def wait() -> str:
            """Wait for cancellation."""
            started.set()
            try:
                await asyncio.sleep(30)
            finally:
                stopped.set()
            return "late"

        install(
            r,
            Registry(
                config(tmp_path, tools=("wait",)),
                tools=[wait],
                extensions={"work": Extension(middleware_factory=lambda: (Swallow(),))},
            ),
        )
        row, _ = await r.store.submit("cancel", profile="work", workspace_id="source")
        model = SyntheticModel(responses=[call("wait"), AIMessage("late")])
        worker = Worker(r, model=model)
        claim = await r.store.claim(worker.id)
        execution = asyncio.create_task(worker.execute(claim))
        await asyncio.wait_for(started.wait(), 2)
        await r.store.cancel(row["id"])
        await execution
        assert stopped.is_set() and len(model.history) == 1
        assert (await r.store.get(row["id"]))["status"] == "cancelled"

    database(check, replace(Settings(), heartbeat_seconds=0.01, lease_seconds=1))
