"""Real isolated Docker background execution, journal recovery and publication races."""

import asyncio
from uuid import uuid4

import psycopg
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.background import monitor_operation
from agent_service.config import Settings
from agent_service.execution import DockerClient
from agent_service.execution_config import ExecutionSettings
from agent_service.profiles import Profile, Workspace
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_core import execute
from tests.test_execution import install
from tests.test_profiles_policy import call

TOOLS = (
    "read_file",
    "glob_search",
    "grep_search",
    "edit_file",
    "shell",
    "delegate_read",
    "start_background_shell",
    "background_status",
    "background_result",
    "cancel_background",
    "write_todos",
)


def setup(r, root, **kwargs):
    install(
        r,
        root,
        profiles=[Profile(id="task_work", tools=TOOLS, requires_workspace=True)],
        **kwargs,
    )


async def task(r, goal="background"):
    return (
        await r.store.create_task(goal, profile="task_work", workspace_id="source")
    )[0]


async def poll(r, task_id, *, limit=800):
    for _ in range(limit):
        for op in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, op)
        await r.store.maintain()
        ops = await r.store.operations(task_id, raw=True)
        if ops and all(
            o["cleanup_state"] == "released"
            and o["state"] in ("succeeded", "failed", "cancelled", "unknown")
            for o in ops
        ):
            return ops
        await asyncio.sleep(0.05)
    raise AssertionError(await r.store.operations(task_id))


def test_background_run_releases_slot_restart_resumes_once_with_messages(
    database, tmp_path, monkeypatch
):
    (tmp_path / "a.txt").write_text("old")

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        model = SyntheticModel(
            responses=[
                call(
                    "start_background_shell",
                    {"command": "sleep 1; echo new > a.txt; echo finished"},
                ),
                call("read_file", {"path": "a.txt"}, "read"),
                AIMessage("Started; awaiting result"),
            ]
        )
        first = await execute(r, Worker(r, model=model))
        assert first["status"] == "completed", first
        assert (await r.store.task(t["id"]))["status"] == "waiting_background"
        assert (await r.store.task(t["id"]))["answer"] is None
        assert (await r.store.execution_resource(first["id"]))["state"] == "released"
        ops = await poll(r, t["id"])
        assert ops[0]["state"] == "succeeded", ops
        assert "finished" in ops[0]["result"]["output"]
        assert (tmp_path / "a.txt").read_text().strip() == "new"
        # Multiple schedulers see exactly one continuation after a new Worker starts.
        await asyncio.gather(*(r.store.maintain() for _ in range(6)))
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 2
        second_model = SyntheticModel(responses=[AIMessage("Final verified answer")])
        second_worker = Worker(r, model=second_model)
        original = r.store.finish

        async def lost(*args, **kwargs):
            await original(*args, **kwargs)
            raise psycopg.OperationalError("commit ack lost")

        monkeypatch.setattr(r.store, "finish", lost)
        second = await execute(r, second_worker)
        assert second["status"] == "completed", second
        assert second["previous_run_id"] == first["id"]
        assert second["execution_no"] == 2
        data = "\n".join(m.text for m in second_model.history[0])
        assert "Started; awaiting result" in data and "finished" in data
        assert (await r.store.task(t["id"]))["answer"] == "Final verified answer"
        assert (await r.store.operations(t["id"]))[0]["consumed_run_id"] == second["id"]
        final = await second_worker.agent.aget_state(second["final_checkpoint"])
        assert not final.values.get("todos")

    database(check)


def test_background_readonly_current_run_and_child_block_write_shell_and_other_workspace(
    database, tmp_path
):
    (tmp_path / "a.txt").write_text("reading")
    child_root = tmp_path / "child"
    child_root.mkdir()

    async def check(r):
        setup(r, tmp_path, workspaces=[Workspace(id="child", root=str(child_root))])
        t = await task(r)
        responses = [
            call("start_background_shell", {"command": "sleep 3; echo done"}),
            call(
                "edit_file",
                {"path": "b.txt", "new_text": "bad", "create": True},
                "write",
            ),
            call("shell", {"command": "echo bad > bad.txt"}, "shell"),
            call("delegate_read", {"task": "read a.txt"}, "delegate"),
            call("read_file", {"path": "a.txt"}, "child-read"),
            AIMessage("read report"),
            AIMessage("waiting"),
        ]
        model = SyntheticModel(responses=responses)
        first = await execute(r, Worker(r, model=model))
        assert first["status"] == "completed", first
        texts = "\n".join(
            m.text for m in model.history[-1] if isinstance(m, ToolMessage)
        )
        assert texts.count("BACKGROUND_WORKSPACE_READ_ONLY") == 2
        assert not (tmp_path / "b.txt").exists() and not (tmp_path / "bad.txt").exists()
        await r.store.submit("other", profile="workspace_read", workspace_id="child")
        assert await r.store.claim(uuid4()) is None
        # An unrelated chat run can still use the released Agent slot.
        await r.store.submit("chat")
        chat = await r.store.claim(uuid4())
        assert chat and chat["workspace_id"] is None
        await r.store.finish(chat["id"], chat["lease_token"], error_code="TEST")
        await r.store.cancel_task(t["id"])
        await poll(r, t["id"])

    database(check)


@pytest.mark.parametrize("nonzero", [False, True])
def test_background_current_run_consumes_result_without_extra_run(
    database, tmp_path, nonzero
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        index = 0

        async def reply(messages):
            nonlocal index
            index += 1
            if index == 1:
                return call(
                    "start_background_shell",
                    {"command": "echo bounded; exit " + ("2" if nonzero else "0")},
                )
            if index == 2:
                ops = await poll(r, t["id"])
                return call(
                    "background_result", {"operation_id": str(ops[0]["id"])}, "result"
                )
            return AIMessage("Handled result")

        result = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert result["status"] == "completed", result
        assert (await r.store.task(t["id"]))["status"] == "completed"
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 1

    database(check)


def test_background_start_batch_rejected_before_any_side_effect(database, tmp_path):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        batch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "start_background_shell",
                    "args": {"command": "touch forbidden"},
                    "id": "start",
                },
                {"name": "read_file", "args": {"path": "a.txt"}, "id": "read"},
            ],
        )
        run = await execute(
            r, Worker(r, model=SyntheticModel(responses=[batch, AIMessage("handled")]))
        )
        assert run["status"] == "completed", run
        assert await r.store.operations(t["id"]) == []
        assert not (tmp_path / "forbidden").exists()

    database(check)


@pytest.mark.parametrize(
    "stop", ["timeout", "operation_cancel", "task_cancel", "run_failure"]
)
def test_background_deadline_cancellation_and_failure_kill_container(
    database, tmp_path, stop
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)

        async def reply(messages):
            if not any(isinstance(m, ToolMessage) for m in messages):
                return call(
                    "start_background_shell",
                    {
                        "command": 'setsid /bin/sh -c "sleep 1.8; echo escaped > escaped" & sleep 100 & wait'
                    },
                )
            if stop == "run_failure":
                raise RuntimeError("synthetic crash")
            return AIMessage("waiting")

        run = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        op = (await r.store.operations(t["id"], raw=True))[0]
        if stop == "timeout":
            async with r.pool.connection() as c:
                await c.execute(
                    "UPDATE agent.background_operations SET deadline_at=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (op["id"],),
                )
        elif stop == "operation_cancel":
            await r.store.cancel_operation(t["id"], op["id"])
        elif stop == "task_cancel":
            await r.store.cancel_task(t["id"])
        ops = await poll(r, t["id"])
        assert ops[0]["cleanup_state"] == "released"
        assert await DockerClient().inspect(op["container_name"]) is None
        await asyncio.sleep(2)
        assert not (tmp_path / "escaped").exists()
        state = (await r.store.task(t["id"]))["status"]
        if stop == "task_cancel":
            assert state == "cancelled"
        elif stop == "run_failure":
            assert state == "failed"
        else:
            assert state in ("queued", "waiting_background")
        assert run["status"] == ("failed" if stop == "run_failure" else "completed")

    database(check)


def test_background_output_flood_bounded_and_longer_than_shell_timeout(
    database, tmp_path
):
    async def check(r):
        import os

        setup(
            r,
            tmp_path,
            execution=ExecutionSettings(image=os.environ["AGENT_TEST_EXECUTION_IMAGE"]),
        )
        t = await task(r)
        first = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell",
                            {
                                "command": "sleep 30.2; python -c \"print('界'*1000000)\""
                            },
                        ),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        assert first["status"] == "completed", first
        op = (await poll(r, t["id"]))[0]
        assert op["state"] == "succeeded", op
        assert op["result"]["truncated"]
        assert len(op["result"]["output"].encode()) <= 16384
        assert op["result"]["total_bytes"] > 1000000

    database(check)


def test_background_dispatch_ack_loss_and_start_ack_loss_are_not_replayed(
    database, tmp_path, monkeypatch
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        original = DockerClient.call
        starts = 0

        async def lost(self, *args, **kwargs):
            nonlocal starts
            result = await original(self, *args, **kwargs)
            if args[0] == "start" and "agent-bg-" in str(
                (await self.inspect(args[1]))["Name"]
            ):
                starts += 1
                raise OSError("start acknowledgement lost")
            return result

        monkeypatch.setattr(DockerClient, "call", lost)
        run = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell", {"command": "echo once >> count"}
                        )
                    ]
                ),
            ),
        )
        assert run["status"] == "failed"
        async with r.pool.connection() as c:
            await c.execute("UPDATE agent.background_operations SET monitor_until=NULL")
        await poll(r, t["id"])
        assert starts == 1
        assert (
            not (tmp_path / "count").exists()
            or (tmp_path / "count").read_text().count("once") == 1
        )
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_background_unconfirmed_cleanup_holds_workspace_and_capacity(
    database, tmp_path, monkeypatch
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call("start_background_shell", {"command": "echo finished"}),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        original = DockerClient.call

        async def lost(self, *args, **kwargs):
            if args[0] == "rm":
                raise OSError("daemon unreachable")
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(DockerClient, "call", lost)
        for _ in range(30):
            for op in await r.store.claim_operations():
                await monitor_operation(r.store, r.registry, op)
            ops = await r.store.operations(t["id"])
            if ops[0]["cleanup_state"] == "failed":
                break
            await asyncio.sleep(0.05)
        assert ops[0]["cleanup_state"] == "failed"
        await r.store.maintain()
        assert (await r.store.task(t["id"]))["status"] == "failed"
        await r.store.submit("blocked", profile="workspace_read", workspace_id="source")
        assert await r.store.claim(uuid4()) is None
        monkeypatch.setattr(DockerClient, "call", original)
        await poll(r, t["id"])
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_background_worker_loop_restart_dependency_api_end_to_end(database, tmp_path):
    import httpx
    from agent_service.app import app

    async def check(r):
        setup(r, tmp_path)
        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://127.0.0.1"
        ) as client:
            parent = (
                await client.post("/v1/tasks", json={"task": "dependency"})
            ).json()
            created = (
                await client.post(
                    "/v1/tasks",
                    json={
                        "task": "background goal",
                        "profile": "task_work",
                        "workspace_id": "source",
                        "depends_on": [parent["task_id"]],
                    },
                )
            ).json()
            assert created["status"] == "waiting_dependencies"
            first = Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        AIMessage("dependency result"),
                        call(
                            "start_background_shell",
                            {
                                "command": "sleep 2; echo completed > result.txt; echo verified"
                            },
                        ),
                        AIMessage("waiting for operation"),
                    ]
                ),
            )
            loop = asyncio.create_task(first.run())
            try:
                for _ in range(300):
                    state = (await r.store.task(created["id"]))["status"]
                    if state == "waiting_background" and all(
                        t.done() for t in first.tasks.values()
                    ):
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError(await r.store.task(created["id"]))
            finally:
                first.stopping.set()
                await loop
            assert (await r.store.task(created["id"]))["status"] == "waiting_background"
            second_model = SyntheticModel(
                responses=[AIMessage("final from recovered result")]
            )
            second = Worker(r, model=second_model)
            loop = asyncio.create_task(second.run())
            try:
                for _ in range(300):
                    if (await r.store.task(created["id"]))["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError(await r.store.task(created["id"]))
            finally:
                second.stopping.set()
                await loop
            assert (tmp_path / "result.txt").read_text().strip() == "completed"
            rows = (await r.store.runs(task_id=created["id"]))["items"]
            assert len(rows) == 2
            assert "verified" in str(second_model.history)
            assert (await client.get("/v1/tasks/" + created["id"])).json()[
                "answer"
            ] == "final from recovered result"

    database(check)


def test_background_db_dispatch_ack_loss_never_repeats_command(
    database, tmp_path, monkeypatch
):
    from contextlib import asynccontextmanager

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
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
                raise psycopg.OperationalError("dispatch commit acknowledgement lost")

        # Arm only during reserve; a later DB retry confirms the original dispatch UUID.
        reserve = r.store.reserve_operation

        async def reserve_with_lost_ack(*args, **kwargs):
            nonlocal armed
            armed = True
            try:
                return await reserve(*args, **kwargs)
            finally:
                armed = False

        monkeypatch.setattr(r.store, "locked", lost_commit)
        monkeypatch.setattr(r.store, "reserve_operation", reserve_with_lost_ack)
        run = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell", {"command": "echo once >> count"}
                        ),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        assert run["status"] == "completed", run
        ops = await poll(r, t["id"])
        assert lost and len(ops) == 1 and ops[0]["state"] == "succeeded"
        assert (tmp_path / "count").read_text() == "once\n"

    database(check)


def test_background_result_modified_by_hook_is_not_consumed(database, tmp_path):
    from langchain.agents.middleware import AgentMiddleware
    from agent_service.profiles import Extension

    class HideResult(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            result = await handler(request)
            if request.tool_call["name"] == "background_result":
                return result.model_copy(update={"content": "hidden by hook"})
            return result

    async def check(r):
        setup(
            r,
            tmp_path,
            extensions={
                "task_work": Extension(
                    version="hide", middleware_factory=lambda: (HideResult(),)
                )
            },
        )
        t = await task(r)
        i = 0

        async def reply(messages):
            nonlocal i
            i += 1
            if i == 1:
                return call("start_background_shell", {"command": "echo ready"})
            if i == 2:
                ops = await poll(r, t["id"])
                return call(
                    "background_result", {"operation_id": str(ops[0]["id"])}, "result"
                )
            return AIMessage("progress")

        run = await execute(r, Worker(r, model=SyntheticModel(reply=reply)))
        assert run["status"] == "completed", run
        assert (await r.store.operations(t["id"]))[0]["consumed_run_id"] is None
        assert (await r.store.task(t["id"]))["status"] == "queued"
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 1

    database(check)


def test_background_only_profile_reserves_workspace_and_is_task_only(
    database, tmp_path
):
    from contracts.errors import ApiError

    async def check(r):
        install(
            r,
            tmp_path,
            profiles=[Profile(id="only", tools=("start_background_shell",))],
        )
        with pytest.raises(ApiError) as error:
            await r.store.submit("standalone", profile="only", workspace_id="source")
        assert error.value.status == 422
        t, _ = await r.store.create_task(
            "background only", profile="only", workspace_id="source"
        )
        run = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call("start_background_shell", {"command": "echo only"}),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        assert run["status"] == "completed", run
        assert (await r.store.execution_resource(run["id"]))["state"] == "released"
        assert (await poll(r, t["id"]))[0]["state"] == "succeeded"

    database(check)


@pytest.mark.parametrize(
    "phase", ["reserved", "creating", "created", "starting", "started"]
)
def test_background_restart_never_replays_dispatch_phases(database, tmp_path, phase):
    from agent_service.background import start_docker
    from agent_service.mcp_config import digest

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        row = await r.store.claim(uuid4())
        op = await r.store.reserve_operation(
            row,
            "crash",
            "docker",
            digest({"command": "echo once >> phases"}),
            {
                "execution": r.registry.execution.model_dump(),
                "workspace_identity": row["submitted_binding"]["workspace_identity"],
            },
            workspace_root=str(tmp_path),
        )
        if phase != "reserved":
            if phase == "creating":
                await r.store.operation_update(
                    op, daemon_id=await DockerClient().daemon(), phase="creating"
                )
            else:
                await start_docker(r.store, r.registry, op, "echo once >> phases", ".")
                # Persisted phase can lag behind external progress after lost acknowledgements.
                await r.store.operation_update(op, phase=phase)
        await r.store.release_monitor(op, 0)
        for item in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, item)
        await r.store.finish(row["id"], row["lease_token"], error_code="LEASE_EXPIRED")
        if phase == "creating":
            after = await r.store.operation(t["id"], op["id"], raw=True)
            assert after["state"] == "unknown" and after["cleanup_state"] != "released"
            assert not (tmp_path / "phases").exists()
        else:
            ops = await poll(r, t["id"])
            assert ops[0]["cleanup_state"] == "released"
            assert (
                not (tmp_path / "phases").exists()
                or (tmp_path / "phases").read_text().count("once") == 1
            )

    database(check)


def test_background_capacity_independent_global_and_per_task_start_limit(
    database, tmp_path
):
    from agent_service.mcp_config import digest
    from langchain_core.tools import ToolException

    async def check(r):
        setup(r, tmp_path)
        await task(r)
        row = await r.store.claim(uuid4())
        binding = {
            "execution": r.registry.execution.model_dump(),
            "workspace_identity": row["submitted_binding"]["workspace_identity"],
        }
        op = await r.store.reserve_operation(
            row, "one", "docker", digest("one"), binding, workspace_root=str(tmp_path)
        )
        with pytest.raises(ToolException, match="CAPACITY"):
            await r.store.reserve_operation(
                row,
                "two",
                "docker",
                digest("two"),
                binding,
                workspace_root=str(tmp_path),
            )
        await r.store.operation_update(
            op,
            state="failed",
            result={"known": "not started"},
            cleanup_state="released",
        )
        await r.store.release_monitor(op, 0)
        with pytest.raises(ToolException, match="CAPACITY"):
            await r.store.reserve_operation(
                row,
                "three",
                "docker",
                digest("three"),
                binding,
                workspace_root=str(tmp_path),
            )
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST_END")

    database(check, Settings(max_task_operations=1, max_active_operations=1))


def test_background_four_operations_five_runs_limit_and_known_failures(
    database, tmp_path
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        for n in range(4):
            run = await execute(
                r,
                Worker(
                    r,
                    model=SyntheticModel(
                        responses=[
                            call(
                                "start_background_shell",
                                {"command": f"echo attempt-{n}; exit 1"},
                                f"launch-{n}",
                            ),
                            AIMessage(f"progress {n}"),
                        ]
                    ),
                ),
            )
            assert run["status"] == "completed", run
            await poll(r, t["id"])
        final_model = SyntheticModel(
            responses=[
                call(
                    "start_background_shell",
                    {"command": "touch forbidden"},
                    "over-limit",
                ),
                AIMessage("final report"),
            ]
        )
        run = await execute(r, Worker(r, model=final_model))
        assert run["status"] == "completed", run
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 5
        assert len(await r.store.operations(t["id"])) == 4
        assert not (tmp_path / "forbidden").exists()
        assert "BACKGROUND_CAPACITY_EXCEEDED" in str(final_model.history)
        assert (await r.store.task(t["id"]))["status"] == "completed"

    database(check)


def test_background_two_tasks_global_capacity_does_not_consume_agent_slot(database):
    from agent_service.mcp_config import digest
    from langchain_core.tools import ToolException

    async def check(r):
        for n in range(3):
            await r.store.create_task(str(n))
        rows = [await r.store.claim(uuid4()) for _ in range(3)]
        operations = []
        for row in rows[:2]:
            operations.append(
                await r.store.reserve_operation(
                    row, "remote", "mcp", digest("args"), {"test": "not-dispatched"}
                )
            )
        with pytest.raises(ToolException, match="CAPACITY"):
            await r.store.reserve_operation(
                rows[2], "remote", "mcp", digest("args"), {"test": "not-dispatched"}
            )
        # Reserved but undispatched operations have a known non-effect and can be released.
        for op in operations:
            await r.store.operation_update(
                op,
                state="failed",
                result={"known": "not dispatched"},
                cleanup_state="released",
            )
            await r.store.release_monitor(op, 0)
        for row in rows:
            await r.store.finish(row["id"], row["lease_token"], error_code="TEST_END")

    database(check, Settings(max_active_runs=3))


@pytest.mark.parametrize("change", ["daemon", "configuration"])
def test_background_identity_change_retains_occupancy_and_never_mutates_wrong_resource(
    database, tmp_path, monkeypatch, change
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        run = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call("start_background_shell", {"command": "sleep 50"}),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        assert run["status"] == "completed", run
        original = DockerClient.daemon
        oldcfg = r.registry.execution
        if change == "daemon":

            async def changed(self):
                return "another-daemon"

            monkeypatch.setattr(DockerClient, "daemon", changed)
        else:
            r.registry.execution = oldcfg.model_copy(update={"cpus": 0.5})
        for op in await r.store.claim_operations():
            await monitor_operation(r.store, r.registry, op)
        await r.store.maintain()
        op = (await r.store.operations(t["id"]))[0]
        assert op["state"] == "unknown" and op["cleanup_state"] == "failed"
        assert (await r.store.task(t["id"]))["status"] == "failed"
        monkeypatch.setattr(DockerClient, "daemon", original)
        r.registry.execution = oldcfg
        await poll(r, t["id"])
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_task_cancellation_with_failed_run_resource_cleanup_never_reports_cancelled(
    database, tmp_path
):
    from agent_service.execution import recover_resources

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        row = await r.store.claim(uuid4())
        await r.store.cancel_task(t["id"])
        await r.store.resource_update(
            row["id"],
            row["lease_token"],
            state="cleanup_failed",
            cleanup_error="synthetic daemon loss",
        )
        await r.store.finish(row["id"], row["lease_token"], cancelled=True)
        result = await r.store.task(t["id"])
        assert (
            result["status"] == "failed"
            and result["error_code"] == "RESOURCE_CLEANUP_FAILED"
        )
        assert (await r.store.execution_resource(row["id"]))[
            "state"
        ] == "cleanup_failed"
        await recover_resources(r.store, r.registry.execution)
        await r.store.maintain()
        assert (await r.store.task(t["id"]))["status"] == "failed"

    database(check)


def test_background_monitor_lease_replacement_rejects_stale_observer(
    database, tmp_path
):
    from agent_service.external_operations import ExternalFailure
    from agent_service.mcp_config import digest

    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        row = await r.store.claim(uuid4())
        op = await r.store.reserve_operation(
            row,
            "lease",
            "docker",
            digest("not dispatched"),
            {
                "execution": r.registry.execution.model_dump(),
                "workspace_identity": row["submitted_binding"]["workspace_identity"],
            },
            workspace_root=str(tmp_path),
        )
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.background_operations SET monitor_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (op["id"],),
            )
        claimed = await asyncio.gather(
            r.store.claim_operations(), r.store.claim_operations()
        )
        assert sum(len(items) for items in claimed) == 1
        with pytest.raises(ExternalFailure, match="MONITOR_LEASE_LOST"):
            await r.store.operation_update(
                op, state="succeeded", result={"forbidden": "stale"}
            )
        fresh = next(items[0] for items in claimed if items)
        await monitor_operation(r.store, r.registry, fresh)
        after = await r.store.operation(t["id"], op["id"], raw=True)
        assert after["state"] == "failed" and after["cleanup_state"] == "released"
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST_END")

    database(check)


@pytest.mark.parametrize("late", ["deadline", "cancel"])
def test_background_completed_receipt_survives_late_observation_or_cancellation(
    database, tmp_path, late
):
    async def check(r):
        setup(r, tmp_path)
        t = await task(r)
        run = await execute(
            r,
            Worker(
                r,
                model=SyntheticModel(
                    responses=[
                        call(
                            "start_background_shell", {"command": "echo known success"}
                        ),
                        AIMessage("waiting"),
                    ]
                ),
            ),
        )
        assert run["status"] == "completed", run
        op = (await r.store.operations(t["id"], raw=True))[0]
        client = DockerClient()
        for _ in range(100):
            if not (await client.inspect(op["container_id"]))["State"]["Running"]:
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.2)
        if late == "deadline":
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.background_operations SET deadline_at=clock_timestamp()-interval '0.1 second' WHERE id=%s",
                    (op["id"],),
                )
        else:
            await r.store.cancel_operation(t["id"], op["id"])
        after = (await poll(r, t["id"]))[0]
        assert after["state"] == "succeeded", after
        assert "known success" in after["result"]["output"]

    database(check)


@pytest.mark.parametrize("output_bytes", [512, 16384])
def test_background_large_result_is_fully_delivered_in_current_run(
    database, tmp_path, output_bytes
):
    import os

    async def check(r):
        setup(
            r,
            tmp_path,
            execution=ExecutionSettings(
                image=os.environ["AGENT_TEST_EXECUTION_IMAGE"],
                output_bytes=output_bytes,
                output_lines=4 if output_bytes == 512 else 200,
            ),
        )
        t = await task(r)
        i = 0

        async def reply(messages):
            nonlocal i
            i += 1
            if i == 1:
                return call(
                    "start_background_shell",
                    {"command": "python -c \"print('x'*20000)\""},
                )
            if i == 2:
                op = (await poll(r, t["id"]))[0]
                return call(
                    "background_result", {"operation_id": str(op["id"])}, "read-result"
                )
            return AIMessage("large result handled")

        model = SyntheticModel(reply=reply)
        run = await execute(r, Worker(r, model=model))
        assert run["status"] == "completed", run
        op = (await r.store.operations(t["id"]))[0]
        result_message = next(
            m
            for m in model.history[-1]
            if isinstance(m, ToolMessage) and m.name == "background_result"
        )
        assert op["result"]["output"] in result_message.content
        assert op["consumed_run_id"] == run["id"]
        assert (await r.store.task(t["id"]))["status"] == "completed"
        assert len((await r.store.runs(task_id=t["id"]))["items"]) == 1

    database(check)
