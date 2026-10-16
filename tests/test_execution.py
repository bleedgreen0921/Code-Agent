"""Synthetic agents, real PostgreSQL and real Docker. Missing dependencies are failures."""

import asyncio
from dataclasses import replace
import hashlib
import os
import subprocess
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent_service.config import Settings
from agent_service.execution import DockerClient, ExecutionScope, recover_resources
from agent_service.execution_config import ExecutionSettings
from agent_service.execution_errors import CleanupFailed, EnvironmentUnavailable
from agent_service.profiles import Extension, Profile, Registry, RuntimeFile, Workspace
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_core import execute
from tests.test_profiles_policy import call


def install(r, root, *, execution=None, profiles=(), extensions=None, workspaces=()):
    image = os.environ["AGENT_TEST_EXECUTION_IMAGE"]
    cfg = execution or ExecutionSettings(image=image)
    if os.getuid() == 0:
        cfg = cfg.model_copy(update={"user": "65534:65534"})
        root.chmod(0o777)
    r.registry = r.store.registry = Registry(
        RuntimeFile(
            version=1,
            execution=cfg,
            workspaces=[Workspace(id="source", root=str(root)), *workspaces],
            profiles=list(profiles),
        ),
        extensions=extensions,
    )


async def scope_for(r, root, profile="workspace_write", *, execution=None):
    install(r, root, execution=execution)
    await r.store.submit("test", profile=profile, workspace_id="source")
    row = await r.store.claim(uuid4())
    row["execution_tools"] = r.registry.profiles[profile].tools
    scope = ExecutionScope(r.store, row, r.registry.execution)
    await scope.prepare()
    return scope


async def finish_scope(scope):
    assert await scope.close()
    await scope.store.finish(
        scope.row["id"], scope.row["lease_token"], error_code="TEST_FINISHED"
    )


def test_execution_config_closed_immutable_and_fingerprints(tmp_path):
    for values in (
        {"image": "python:latest"},
        {"user": "0:0"},
        {"network_enabled": True},
        {"output_bytes": 32768},
        {"command_timeout": float("nan")},
    ):
        with pytest.raises(ValueError):
            ExecutionSettings(**values)
    registry = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="w", root=str(tmp_path))],
            profiles=[Profile(id="subset", tools=("read_file",))],
        )
    )
    for name in ("workspace_read", "workspace_write", "subset"):
        with pytest.raises(ValueError):
            registry.bind(name)
    assert "shell" not in registry.profiles["workspace_read"].tools
    changed = Registry(
        RuntimeFile(
            version=1,
            workspaces=[Workspace(id="w", root=str(tmp_path))],
            execution=ExecutionSettings(command_timeout=1.0),
        )
    )
    assert registry.bind("workspace_read", "w") != changed.bind("workspace_read", "w")
    assert registry.bind() == changed.bind()


def test_read_agent_searches_and_reads_without_host_tools(database, tmp_path):
    (tmp_path / "a.txt").write_text("needle\nsecond\n")

    async def check(r):
        install(r, tmp_path)
        model = SyntheticModel(
            responses=[
                call("glob_search", {"pattern": "*.txt"}),
                call(
                    "grep_search",
                    {"pattern": "needle", "output_mode": "content"},
                    "grep",
                ),
                call("read_file", {"path": "a.txt"}, "read"),
                AIMessage("found"),
            ]
        )
        await r.store.submit("search", profile="workspace_read", workspace_id="source")
        worker = Worker(r, model=model)
        result = await execute(r, worker)
        assert result["status"] == "completed", result
        final = await worker.agent.aget_state(result["final_checkpoint"])
        messages = [m for m in final.values["messages"] if isinstance(m, ToolMessage)]
        assert "/a.txt" in messages[0].text and "/a.txt:1:needle" in messages[1].text
        assert "1: needle" in messages[2].text
        assert (
            messages[2].artifact["sha256"]
            == hashlib.sha256(b"needle\nsecond\n").hexdigest()
        )
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"
        assert not any("resources" in key for key in final.values)

    database(check)


def test_write_agent_hash_edit_shell_nonzero_correction_and_output_limits(
    database, tmp_path
):
    (tmp_path / "a.txt").write_text("old\n")

    async def check(r):
        install(r, tmp_path)

        def reply(messages):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if not tools:
                return call("read_file", {"path": "a.txt"})
            if len(tools) == 1:
                return call(
                    "edit_file",
                    {
                        "path": "a.txt",
                        "old_text": "old",
                        "new_text": "new",
                        "expected_hash": tools[0].artifact["sha256"],
                    },
                    "edit",
                )
            if len(tools) == 2:
                return call("shell", {"command": "grep new a.txt && false"}, "bad")
            if len(tools) == 3:
                assert (
                    tools[-1].status == "error" and tools[-1].artifact["exit_code"] == 1
                )
                return call(
                    "shell",
                    {
                        "command": "python -c \"import sys; sys.stdout.write('a'*1000000); sys.stderr.write('err\\n'*20000)\""
                    },
                    "flood",
                )
            assert tools[-1].artifact["truncated"]
            return AIMessage("verified")

        await r.store.submit("edit", profile="workspace_write", workspace_id="source")
        worker = Worker(r, model=SyntheticModel(reply=reply))
        result = await execute(r, worker)
        assert result["status"] == "completed", result
        assert (tmp_path / "a.txt").read_text() == "new\n"
        final = await worker.agent.aget_state(result["final_checkpoint"])
        for m in final.values["messages"]:
            if isinstance(m, ToolMessage):
                assert len(m.text.encode()) <= 16384 and len(m.text.splitlines()) <= 200
        events = await r.store.timeline(result["id"])
        assert [x["status"] for x in events["events"]][-2:] == [
            "environment_released",
            "completed",
        ]

    database(check)


@pytest.mark.parametrize("profile", ["workspace_read", "workspace_write"])
def test_real_container_isolation_limits_and_readonly_mount(
    database, tmp_path, profile, monkeypatch
):
    async def check(r):
        monkeypatch.setenv("AGENT_MODEL_KEY", "SHOULD_NOT_ENTER_CONTAINER")
        scope = await scope_for(r, tmp_path, profile)
        try:
            obj = await scope.client.inspect(scope.container_id)
            host, config = obj["HostConfig"], obj["Config"]
            assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"]
            assert host["Memory"] == 512 * 1024 * 1024 and host["PidsLimit"] == 128
            assert host["NanoCpus"] == 1000000000
            assert config["User"] == scope.settings.effective_user()
            assert not any(
                "SHOULD_NOT" in v or "DATABASE_URL" in v for v in config["Env"]
            )
            assert len(obj["Mounts"]) == 1 and obj["Mounts"][0]["Source"] == str(
                tmp_path
            )
            result = await scope.file_call(
                "edit_file", {"path": "created", "create": True, "new_text": "ok"}
            )
            if profile == "workspace_read":
                assert "error" in result and not (tmp_path / "created").exists()
            else:
                assert (
                    "error" not in result and (tmp_path / "created").read_text() == "ok"
                )
        finally:
            await finish_scope(scope)

    database(check)


def test_file_semantics_conflicts_and_no_partial_writes(database, tmp_path):
    (tmp_path / "text").write_text("same same")
    (tmp_path / "binary").write_bytes(b"\xff")
    (tmp_path / "big").write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    (tmp_path / "outside").symlink_to("/etc/passwd")
    (tmp_path / "directory").mkdir()
    os.mkfifo(tmp_path / "pipe")

    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            for path in (
                "../etc/passwd",
                "outside",
                "directory",
                "pipe",
                "binary",
                "big",
            ):
                result = await scope.file_call("read_file", {"path": path})
                assert "error" in result, (path, result)
            read = await scope.file_call("read_file", {"path": "text"})
            for args in (
                {"path": "text", "create": True, "new_text": "overwrite"},
                {
                    "path": "text",
                    "old_text": "same",
                    "new_text": "bad",
                    "expected_hash": read["sha256"],
                },
                {
                    "path": "text",
                    "old_text": "same same",
                    "new_text": "bad",
                    "expected_hash": "stale",
                },
                {"path": "directory", "create": True, "new_text": "bad"},
            ):
                assert "error" in await scope.file_call("edit_file", args)
            assert (tmp_path / "text").read_text() == "same same"
            assert not list(tmp_path.glob(".agent-edit-*"))
            (tmp_path / "text").write_text("external change")
            assert "error" in await scope.file_call(
                "edit_file",
                {
                    "path": "text",
                    "old_text": "same same",
                    "new_text": "bad",
                    "expected_hash": read["sha256"],
                },
            )
            assert (tmp_path / "text").read_text() == "external change"
            assert "error" not in await scope.file_call(
                "edit_file", {"path": "new", "create": True, "new_text": "created"}
            )
        finally:
            await finish_scope(scope)

    database(check)


def test_empty_chat_does_not_contact_docker(database, monkeypatch):
    async def broken(*args, **kwargs):
        raise AssertionError("chat contacted Docker")

    monkeypatch.setattr(DockerClient, "call", broken)

    async def check(r):
        await r.store.submit("chat")
        assert (await execute(r))["status"] == "completed"

    database(check)


def test_environment_missing_image_fails_before_model(database, tmp_path):
    async def check(r):
        install(r, tmp_path, execution=ExecutionSettings())
        await r.store.submit("read", profile="workspace_read", workspace_id="source")
        model = SyntheticModel()
        result = await execute(r, Worker(r, model=model))
        assert (
            result["error_code"] == "EXECUTION_ENVIRONMENT_UNAVAILABLE"
            and not model.history
        )
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(check)


def test_command_timeout_is_fatal_hooks_cannot_swallow_and_no_restart(
    database, tmp_path
):
    from langchain.agents.middleware import AgentMiddleware, ToolErrorMiddleware

    class Swallow(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            try:
                return await handler(request)
            except Exception:
                return ToolMessage("ignored", tool_call_id=request.tool_call["id"])

    async def check(r):
        install(
            r,
            tmp_path,
            execution=ExecutionSettings(
                image=os.environ["AGENT_TEST_EXECUTION_IMAGE"], command_timeout=2.0
            ),
            profiles=(Profile(id="timed", tools=("shell",)),),
            extensions={
                "timed": Extension(
                    middleware_factory=lambda: (
                        Swallow(),
                        ToolErrorMiddleware(lambda exc, req: "ignored"),
                    )
                )
            },
        )
        await r.store.submit("timeout", profile="timed", workspace_id="source")
        model = SyntheticModel(
            responses=[
                call(
                    "shell",
                    {"command": "echo changed > before; sleep 20; echo late > after"},
                ),
                AIMessage("wrong"),
            ]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["error_code"] == "COMMAND_TIMEOUT", result
        assert (
            len(model.history) == 1
            and (tmp_path / "before").exists()
            and not (tmp_path / "after").exists()
        )
        resource = await r.store.execution_resource(result["id"])
        assert (
            resource["state"] == "released"
            and await DockerClient().inspect(resource["container_id"]) is None
        )

    database(check)


def test_cancel_shell_cleans_background_and_preserves_changes(database, tmp_path):
    async def check(r):
        install(r, tmp_path)
        await r.store.submit("cancel", profile="workspace_write", workspace_id="source")
        model = SyntheticModel(
            responses=[
                call(
                    "shell",
                    {
                        "command": "(sleep 3; echo leaked > leaked) & echo started > started; sleep 30"
                    },
                ),
                AIMessage("late"),
            ]
        )
        worker = Worker(r, model=model)
        row = await r.store.claim(worker.id)
        task = asyncio.create_task(worker.execute(row))
        try:
            async with asyncio.timeout(10):
                while not (tmp_path / "started").exists():
                    await asyncio.sleep(0.02)
            await r.store.cancel(row["id"])
            await task
            assert (await r.store.get(row["id"]))["status"] == "cancelled"
            assert (await r.store.execution_resource(row["id"]))["state"] == "released"
            assert not (tmp_path / "leaked").exists() and len(model.history) == 1
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    database(check, replace(Settings(), heartbeat_seconds=0.03, lease_seconds=2.0))


def test_workspace_overlap_alias_serialization_and_capacity_after_terminal(
    database, tmp_path
):
    child = tmp_path / "child"
    child.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path)
    other = tmp_path.parent / (tmp_path.name + "-other")
    other.mkdir()

    async def check(r):
        install(
            r,
            tmp_path,
            workspaces=(
                Workspace(id="child", root=str(child)),
                Workspace(id="alias", root=str(alias)),
                Workspace(id="other", root=str(other)),
            ),
        )
        rows = [
            (await r.store.submit("x", profile="workspace_read", workspace_id=w))[0]
            for w in ("source", "alias", "child", "other")
        ]
        first = await r.store.claim(uuid4())
        second = await r.store.claim(uuid4())
        assert first["id"] == rows[0]["id"] and second["id"] == rows[3]["id"]
        assert await r.store.claim(uuid4()) is None
        await r.store.finish(
            first["id"], first["lease_token"], error_code="TEST_FAILED"
        )
        assert (
            await r.store.claim(uuid4()) is None
        )  # terminal resource still consumes capacity
        await r.store.resource_update(
            first["id"], first["lease_token"], state="released"
        )
        await r.store.finish(
            second["id"], second["lease_token"], error_code="TEST_FAILED"
        )
        assert (await r.store.claim(uuid4()))["id"] == rows[1]["id"]

    database(check)


def test_recovery_after_worker_loss_never_replays_agent(database, tmp_path):
    async def check(r):
        scope = await scope_for(r, tmp_path)
        identity = scope.container_id
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                (scope.row["id"],),
            )
        model = SyntheticModel()
        worker = Worker(r, model=model)
        await worker.tick()
        assert not model.history and not worker.tasks
        assert (await r.store.get(scope.row["id"]))["error_code"] == "LEASE_EXPIRED"
        assert (await r.store.execution_resource(scope.row["id"]))[
            "state"
        ] == "released"
        assert await DockerClient().inspect(identity) is None

    database(check)


def test_uncertain_create_and_daemon_mismatch_preserve_occupancy(database, tmp_path):
    async def check(r):
        install(r, tmp_path)
        await r.store.submit("unknown", profile="workspace_read", workspace_id="source")
        row = await r.store.claim(uuid4())
        await r.store.resource_update(
            row["id"],
            row["lease_token"],
            daemon_id=await DockerClient().daemon(),
            phase="creating",
        )
        await r.store.finish(row["id"], row["lease_token"], error_code="TEST_CRASH")
        await recover_resources(r.store, r.registry.execution)
        resource = await r.store.execution_resource(row["id"])
        assert (
            resource["state"] == "cleanup_failed"
            and "unknown" in resource["cleanup_error"]
        )
        # Inject persisted identity corruption; ordinary updates cannot rebind resources.
        async with r.pool.connection() as conn:
            await conn.execute(
                "UPDATE agent.execution_resources SET daemon_id='different',phase='created' WHERE run_id=%s",
                (row["id"],),
            )
        await recover_resources(r.store, r.registry.execution)
        resource = await r.store.execution_resource(row["id"])
        assert (
            resource["state"] == "cleanup_failed"
            and "mismatch" in resource["cleanup_error"]
        )
        await r.store.submit("blocked", profile="workspace_read", workspace_id="source")
        assert await r.store.claim(uuid4()) is None

    database(check)


def test_cleanup_failure_blocks_success_until_recovery(database, tmp_path, monkeypatch):
    from agent_service import execution as execution_module

    original = execution_module.cleanup_resource

    async def broken(*args):
        raise CleanupFailed("synthetic cleanup acknowledgement loss")

    async def check(r):
        install(r, tmp_path)
        await r.store.submit("done", profile="workspace_read", workspace_id="source")
        monkeypatch.setattr(execution_module, "cleanup_resource", broken)
        result = await execute(r)
        assert (
            result["error_code"] == "RESOURCE_CLEANUP_FAILED"
            and result["answer"] is None
        )
        assert (await r.store.execution_resource(result["id"]))[
            "state"
        ] == "cleanup_failed"
        monkeypatch.setattr(execution_module, "cleanup_resource", original)
        await recover_resources(r.store, r.registry.execution)
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"
        assert (await r.store.get(result["id"]))["status"] == "failed"

    database(check)


def test_success_waits_for_cleanup_and_cancellation_wins_publication(
    database, tmp_path, monkeypatch
):
    from agent_service import execution as execution_module

    original = execution_module.cleanup_resource
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args):
        started.set()
        await release.wait()
        return await original(*args)

    async def check(r):
        install(r, tmp_path)
        row, _ = await r.store.submit(
            "race", profile="workspace_read", workspace_id="source"
        )
        worker = Worker(r, model=SyntheticModel())
        claim = await r.store.claim(worker.id)
        monkeypatch.setattr(execution_module, "cleanup_resource", delayed)
        task = asyncio.create_task(worker.execute(claim))
        await asyncio.wait_for(started.wait(), 10)
        try:
            assert (await r.store.get(row["id"]))["status"] == "running"
            assert not await r.store.finish(
                row["id"], claim["lease_token"], answer="early", checkpoint={"x": 1}
            )
            await r.store.cancel(row["id"])
            release.set()
            await task
            assert (await r.store.get(row["id"]))["status"] == "cancelled"
        finally:
            release.set()
            await task

    database(check, replace(Settings(), heartbeat_seconds=0.03, lease_seconds=2.0))


def test_shell_state_is_run_local_and_restart_rejected(database, tmp_path):
    async def check(r):
        install(r, tmp_path)
        conv = await r.store.create_conversation("state", "workspace_write", "source")
        model = SyntheticModel(
            responses=[
                call("shell", {"command": "cd /tmp; export RUN_ONLY=present"}),
                call(
                    "shell", {"command": 'printf \'%s %s\' "$PWD" "$RUN_ONLY"'}, "state"
                ),
                AIMessage("first"),
                call("shell", {"restart": True}, "restart"),
                call(
                    "shell",
                    {"command": 'printf \'%s %s\' "$PWD" "${RUN_ONLY:-empty}"'},
                    "fresh",
                ),
                AIMessage("second"),
            ]
        )
        worker = Worker(r, model=model)
        await r.store.submit("first", conversation_id=conv["id"])
        first = await execute(r, worker)
        assert first["status"] == "completed", first
        await r.store.submit("second", conversation_id=conv["id"])
        second = await execute(r, worker)
        assert second["status"] == "completed", second
        final = await worker.agent.aget_state(second["final_checkpoint"])
        tools = [m for m in final.values["messages"] if isinstance(m, ToolMessage)]
        assert tools[1].text == "/tmp present"
        assert tools[2].status == "error" and "disabled" in tools[2].text
        assert tools[3].text == "/workspace empty"
        resources = [
            await r.store.execution_resource(row["id"]) for row in (first, second)
        ]
        assert resources[0]["container_id"] != resources[1]["container_id"]

    database(check)


def test_atomic_write_publication_failure_keeps_original(tmp_path, monkeypatch):
    from execution import helper

    monkeypatch.setattr(helper, "ROOT", tmp_path)
    target = tmp_path / "text"
    target.write_text("original")
    original = target.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("synthetic atomic replace failure")

    monkeypatch.setattr(helper.os, "replace", fail)
    with pytest.raises(OSError, match="atomic replace"):
        helper.file_operation(
            "edit_file",
            {
                "path": "text",
                "old_text": "original",
                "new_text": "updated",
                "expected_hash": hashlib.sha256(original).hexdigest(),
            },
            1024,
        )
    assert target.read_bytes() == original and not list(tmp_path.glob(".agent-edit-*"))


def test_workspace_permissions_final_hook_arguments_and_actual_objects(
    database, tmp_path
):
    from dataclasses import replace as dc_replace
    from langchain.agents.middleware import AgentMiddleware

    (tmp_path / "good").write_text("allowed")

    class Rewrite(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            return await handler(
                request.override(
                    tool_call={**request.tool_call, "args": {"path": "../etc/passwd"}}
                )
            )

    class Substitute(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            return await handler(dc_replace(request, tool=request.tool.model_copy()))

    async def check(r):
        for name, hook in (("rewrite", Rewrite), ("substitute", Substitute)):
            install(
                r,
                tmp_path,
                profiles=(Profile(id=name, tools=("read_file",)),),
                extensions={name: Extension(middleware_factory=lambda: (hook(),))},
            )
            await r.store.submit("read", profile=name, workspace_id="source")
            worker = Worker(
                r,
                model=SyntheticModel(
                    responses=[call("read_file", {"path": "good"}), AIMessage("done")]
                ),
            )
            result = await execute(r, worker)
            assert result["status"] == "completed", result
            state = await worker.agent.aget_state(result["final_checkpoint"])
            tool = state.values["messages"][-2]
            assert tool.status == "error"
            assert (
                "Invalid workspace path" if name == "rewrite" else "PERMISSION_DENIED"
            ) in tool.text

    database(check)


def test_same_run_tools_execute_serially(database, tmp_path):
    original = ExecutionScope.file_call
    active = 0
    peak = 0

    async def observed(self, *args):
        nonlocal active, peak
        active += 1
        peak = max(active, peak)
        try:
            await asyncio.sleep(0.02)
            return await original(self, *args)
        finally:
            active -= 1

    async def check(r):
        from unittest.mock import patch

        (tmp_path / "text").write_text("content")
        install(r, tmp_path)
        calls = AIMessage(
            "",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "text"},
                    "id": str(i),
                    "type": "tool_call",
                }
                for i in range(2)
            ],
        )
        await r.store.submit(
            "parallel", profile="workspace_read", workspace_id="source"
        )
        with patch.object(ExecutionScope, "file_call", observed):
            result = await execute(
                r, Worker(r, model=SyntheticModel(responses=[calls, AIMessage("done")]))
            )
        assert result["status"] == "completed" and peak == 1

    database(check)


def test_file_tool_cancellation_closes_container(database, tmp_path, monkeypatch):
    started = asyncio.Event()

    async def waiting(self, name, args):
        # A synthetic delayed helper still runs as a real Docker exec process.
        task = asyncio.create_task(
            self.client.call(
                "exec",
                "-i",
                self.container_id,
                "/usr/local/bin/python",
                "-I",
                "-c",
                "import time; time.sleep(20)",
                timeout=30,
            )
        )
        try:
            while not self.client.processes:
                await asyncio.sleep(0.01)
            started.set()
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def check(r):
        install(r, tmp_path)
        (tmp_path / "text").write_text("content")
        row, _ = await r.store.submit(
            "cancel file", profile="workspace_read", workspace_id="source"
        )
        worker = Worker(
            r, model=SyntheticModel(responses=[call("read_file", {"path": "text"})])
        )
        claim = await r.store.claim(worker.id)
        monkeypatch.setattr(ExecutionScope, "file_call", waiting)
        task = asyncio.create_task(worker.execute(claim))
        await asyncio.wait_for(started.wait(), 10)
        await r.store.cancel(row["id"])
        await asyncio.wait_for(task, 10)
        assert (await r.store.get(row["id"]))["status"] == "cancelled"
        resource = await r.store.execution_resource(row["id"])
        assert (
            resource["state"] == "released"
            and await DockerClient().inspect(resource["container_id"]) is None
        )

    database(check, replace(Settings(), heartbeat_seconds=0.03, lease_seconds=2.0))


def test_create_ack_loss_recovery_and_late_start_cannot_recreate(
    database, tmp_path, monkeypatch
):
    original = DockerClient.call
    identity = None

    async def lost_ack(self, *args, **kwargs):
        nonlocal identity
        result = await original(self, *args, **kwargs)
        if args[0] == "create":
            identity = result[1].decode().strip()
            raise OSError("create commit acknowledgement lost")
        return result

    async def check(r):
        install(r, tmp_path)
        monkeypatch.setattr(DockerClient, "call", lost_ack)
        await r.store.submit("create", profile="workspace_read", workspace_id="source")
        model = SyntheticModel()
        result = await execute(r, Worker(r, model=model))
        assert (
            result["error_code"] == "EXECUTION_ENVIRONMENT_UNAVAILABLE"
            and not model.history
        )
        resource = await r.store.execution_resource(result["id"])
        assert resource["state"] == "released" and resource["container_id"] == identity
        code, _, _ = await DockerClient().call("start", identity, check=False)
        assert code and await DockerClient().inspect(identity) is None

    database(check)


def test_remove_ack_loss_retains_capacity_and_recovery_confirms_absence(
    database, tmp_path, monkeypatch
):
    original = DockerClient.call

    async def lost_ack(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        if args[0] == "rm":
            raise OSError("remove acknowledgement lost")
        return result

    async def check(r):
        install(r, tmp_path)
        monkeypatch.setattr(DockerClient, "call", lost_ack)
        await r.store.submit("remove", profile="workspace_read", workspace_id="source")
        result = await execute(r)
        assert result["error_code"] == "RESOURCE_CLEANUP_FAILED"
        resource = await r.store.execution_resource(result["id"])
        assert resource["state"] == "cleanup_failed"
        assert await DockerClient().inspect(resource["container_id"]) is None
        await r.store.submit("blocked", profile="workspace_read", workspace_id="source")
        assert await r.store.claim(uuid4()) is None
        monkeypatch.setattr(DockerClient, "call", original)
        await recover_resources(r.store, r.registry.execution)
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"
        assert await r.store.claim(uuid4()) is not None

    database(check)


def test_docker_unreachable_retains_resource_until_daemon_returns(
    database, tmp_path, monkeypatch
):
    async def check(r):
        scope = await scope_for(r, tmp_path)
        original = DockerClient.daemon

        async def unavailable(self):
            raise EnvironmentUnavailable("synthetic daemon unavailable")

        monkeypatch.setattr(DockerClient, "daemon", unavailable)
        assert not await scope.close()
        assert (await r.store.execution_resource(scope.row["id"]))[
            "state"
        ] == "cleanup_failed"
        await r.store.finish(
            scope.row["id"], scope.row["lease_token"], error_code="TEST_CRASH"
        )
        monkeypatch.setattr(DockerClient, "daemon", original)
        await recover_resources(r.store, r.registry.execution)
        assert (await r.store.execution_resource(scope.row["id"]))[
            "state"
        ] == "released"

    database(check)


def test_real_container_oom_is_fatal(database, tmp_path):
    async def check(r):
        install(
            r,
            tmp_path,
            execution=ExecutionSettings(
                image=os.environ["AGENT_TEST_EXECUTION_IMAGE"],
                memory_bytes=96 * 1024 * 1024,
            ),
        )
        await r.store.submit("oom", profile="workspace_write", workspace_id="source")
        model = SyntheticModel(
            responses=[
                call("shell", {"command": 'python -c "x=bytearray(400*1024*1024)"'}),
                AIMessage("wrong"),
            ]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["error_code"] == "EXECUTION_ENVIRONMENT_LOST", result
        assert len(model.history) == 1
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(check)


def test_worker_sigkill_recovery_cleans_without_replaying(database, tmp_path):
    import sys

    async def check(r):
        install(r, tmp_path)
        submitted, _ = await r.store.submit(
            "crash", profile="workspace_write", workspace_id="source"
        )
        script = """
import asyncio, sys
from pathlib import Path
from agent_service.resources import Resources
from agent_service.config import Settings
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_profiles_policy import call
from tests.test_execution import install
async def main():
    async with Resources(Settings(heartbeat_seconds=.05,lease_seconds=2.0)) as r:
        install(r,Path(sys.argv[1]))
        worker=Worker(r,model=SyntheticModel(responses=[call("shell",{"command":"echo ready > ready; sleep 30"})]))
        row=await r.store.claim(worker.id)
        await worker.execute(row)
asyncio.run(main())
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with asyncio.timeout(15):
                while not (tmp_path / "ready").exists():
                    assert child.poll() is None
                    await asyncio.sleep(0.03)
            resource = await r.store.execution_resource(submitted["id"])
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
            assert not model.history and not worker.tasks
            assert (await r.store.get(submitted["id"]))["error_code"] == "LEASE_EXPIRED"
            assert await DockerClient().inspect(resource["container_id"]) is None
            assert (await r.store.execution_resource(submitted["id"]))[
                "state"
            ] == "released"
        finally:
            if child.poll() is None:
                child.kill()
                await asyncio.to_thread(child.wait)

    database(check)


def test_container_absolute_deadline_survives_worker_absence(database, tmp_path):
    async def check(r):
        scope = await scope_for(r, tmp_path)
        try:
            # Set the watchdog's original absolute deadline through a short Run configuration.
            identity = scope.container_id
            async with asyncio.timeout(10):
                while (await scope.client.inspect(identity))["State"]["Running"]:
                    await asyncio.sleep(0.05)
            assert not (await scope.client.inspect(identity))["State"]["Running"]
        finally:
            assert await scope.close()

    database(
        check,
        replace(Settings(), run_timeout=2.0, heartbeat_seconds=0.05, lease_seconds=4.0),
    )


def test_file_command_timeout_is_fatal(database, tmp_path, monkeypatch):
    async def stalled(self, name, args):
        await self.client.call(
            "exec",
            "-i",
            self.container_id,
            "/usr/local/bin/python",
            "-I",
            "-c",
            "import time; time.sleep(20)",
            timeout=30,
        )

    async def check(r):
        install(
            r,
            tmp_path,
            execution=ExecutionSettings(
                image=os.environ["AGENT_TEST_EXECUTION_IMAGE"], command_timeout=1.0
            ),
        )
        (tmp_path / "text").write_text("kept")
        monkeypatch.setattr(ExecutionScope, "file_call", stalled)
        await r.store.submit(
            "file timeout", profile="workspace_read", workspace_id="source"
        )
        model = SyntheticModel(
            responses=[call("read_file", {"path": "text"}), AIMessage("wrong")]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["error_code"] == "COMMAND_TIMEOUT", result
        assert len(model.history) == 1
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(check)


def test_total_deadline_keeps_run_timeout_semantics(database, tmp_path):
    async def check(r):
        install(r, tmp_path)
        await r.store.submit(
            "total timeout", profile="workspace_write", workspace_id="source"
        )
        model = SyntheticModel(
            responses=[call("shell", {"command": "sleep 20"}), AIMessage("wrong")]
        )
        result = await execute(r, Worker(r, model=model))
        assert result["error_code"] == "RUN_TIMEOUT", result
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(
        check,
        replace(
            Settings(), run_timeout=14.0, heartbeat_seconds=0.05, lease_seconds=3.0
        ),
    )


def test_cleanup_release_commit_ack_loss_is_confirmed_without_replay(
    database, tmp_path, monkeypatch
):
    async def check(r):
        install(r, tmp_path)
        original = r.store.resource_update
        lost = False

        async def ack_loss(*args, **kwargs):
            nonlocal lost
            result = await original(*args, **kwargs)
            if kwargs.get("state") == "released" and not lost:
                lost = True
                raise OSError("release commit acknowledgement lost")
            return result

        monkeypatch.setattr(r.store, "resource_update", ack_loss)
        model = SyntheticModel()
        await r.store.submit("finish", profile="workspace_read", workspace_id="source")
        result = await execute(r, Worker(r, model=model))
        assert result["status"] == "completed", result
        assert len(model.history) == 1 and lost
        assert (await r.store.execution_resource(result["id"]))["state"] == "released"

    database(check)


def test_cleanup_wait_renews_lease_and_lease_loss_blocks_publication(
    database, tmp_path, monkeypatch
):
    from agent_service import execution as execution_module

    original = execution_module.cleanup_resource
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed(*args):
        started.set()
        await release.wait()
        return await original(*args)

    async def check(r):
        install(r, tmp_path)
        model = SyntheticModel()
        submitted, _ = await r.store.submit(
            "cleanup lease", profile="workspace_read", workspace_id="source"
        )
        worker = Worker(r, model=model)
        claim = await r.store.claim(worker.id)
        monkeypatch.setattr(execution_module, "cleanup_resource", delayed)
        task = asyncio.create_task(worker.execute(claim))
        try:
            await asyncio.wait_for(started.wait(), 10)
            await asyncio.sleep(
                0.7
            )  # Exceeds the original lease; cleanup must keep renewing it.
            current = await r.store.get(submitted["id"])
            assert current["status"] == "running"
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (submitted["id"],),
                )
            await asyncio.sleep(0.1)
            release.set()
            await task
            result = await r.store.get(submitted["id"])
            assert result["error_code"] == "LEASE_EXPIRED" and result["answer"] is None
            assert (await r.store.execution_resource(submitted["id"]))[
                "state"
            ] == "released"
            assert len(model.history) == 1
        finally:
            release.set()
            await task

    database(check, replace(Settings(), heartbeat_seconds=0.03, lease_seconds=0.4))


def test_unreachable_docker_after_shell_start_still_closes_host_pipes(
    database, tmp_path, monkeypatch
):
    from agent_service.execution_adapter import ControlledShellMiddleware

    async def check(r):
        scope = await scope_for(r, tmp_path)
        shell = ControlledShellMiddleware(scope)
        resources = await asyncio.to_thread(shell._get_or_create_resources, {})
        command = await asyncio.to_thread(
            resources.session.execute, "echo ready", timeout=2.0
        )
        assert "ready" in command.output
        original = DockerClient.daemon

        async def unavailable(self):
            raise EnvironmentUnavailable("daemon unavailable during cleanup")

        monkeypatch.setattr(DockerClient, "daemon", unavailable)
        assert not await scope.close()
        assert resources.session._process is None
        assert (
            not resources.session._stdout_thread.is_alive()
            and not resources.session._stderr_thread.is_alive()
        )
        assert not scope.shell_processes and not scope.native_tasks
        monkeypatch.setattr(DockerClient, "daemon", original)
        await r.store.finish(
            scope.row["id"], scope.row["lease_token"], error_code="TEST_CRASH"
        )
        await recover_resources(r.store, r.registry.execution)
        assert (await r.store.execution_resource(scope.row["id"]))[
            "state"
        ] == "released"

    database(check)


def test_worker_loop_chat_with_configured_but_unavailable_docker(
    database, tmp_path, monkeypatch
):
    async def broken(*args, **kwargs):
        raise EnvironmentUnavailable("synthetic unavailable Docker")

    async def check(r):
        install(r, tmp_path)
        monkeypatch.setattr(DockerClient, "call", broken)
        submitted, _ = await r.store.submit("chat despite Docker outage")
        worker = Worker(r, model=SyntheticModel())
        task = asyncio.create_task(worker.run())
        try:
            async with asyncio.timeout(5):
                while (await r.store.get(submitted["id"]))["status"] != "completed":
                    await asyncio.sleep(0.01)
        finally:
            worker.stopping.set()
            await task

    database(check)


def test_final_tool_output_limit_includes_hook_wrapping(database, tmp_path):
    from langchain.agents.middleware import AgentMiddleware

    class Wrap(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            result = await handler(request)
            return result.model_copy(
                update={"content": result.text + "\nwrapper" * 1000}
            )

    async def check(r):
        (tmp_path / "text").write_text("content")
        install(
            r,
            tmp_path,
            execution=ExecutionSettings(
                image=os.environ["AGENT_TEST_EXECUTION_IMAGE"],
                output_bytes=512,
                output_lines=4,
            ),
            profiles=(Profile(id="wrapped", tools=("read_file",)),),
            extensions={"wrapped": Extension(middleware_factory=lambda: (Wrap(),))},
        )
        await r.store.submit("wrapped", profile="wrapped", workspace_id="source")
        worker = Worker(
            r,
            model=SyntheticModel(
                responses=[call("read_file", {"path": "text"}), AIMessage("done")]
            ),
        )
        result = await execute(r, worker)
        assert result["status"] == "completed", result
        final = await worker.agent.aget_state(result["final_checkpoint"])
        message = final.values["messages"][-2]
        assert len(message.text.encode()) <= 512 and len(message.text.splitlines()) <= 4
        assert message.artifact["truncated"]

    database(check)
