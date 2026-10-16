"""Native subgraphs with synthetic models, plus real Postgres/Docker regressions."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import httpx
import openai
import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from agent_service.config import Settings
from agent_service.delegation import (
    DelegateInput,
    DelegationController,
    DelegationDiagnostics,
    ReportedUsage,
    delegate_tool,
)
from agent_service.execution_config import ExecutionSettings
from agent_service.profiles import PermissionPolicy, Profile, Registry, RuntimeFile
from agent_service.runtime import RunContext, build_agent, run_config
from agent_service.workspace_tools import file_tools
from tests.models import SyntheticModel
from tests.test_profiles_policy import call


def constraints(messages):
    return json.loads(messages[0].text.split("Run constraints and progress:\n")[-1])


def harness(
    model, *, validators=None, middleware=(), settings=None, scope=None, schema=None
):
    saver = InMemorySaver()
    reads = []

    async def file_call(name, args):
        reads.append((name, args))
        return {"content": "1: needle", "sha256": "hash"}

    async def healthy():
        return None

    scope = scope or SimpleNamespace(
        settings=ExecutionSettings(),
        lock=asyncio.Lock(),
        check_open=lambda: None,
        healthy=healthy,
        remaining=lambda: 300,
        file_call=file_call,
    )
    settings = settings or Settings()
    read = next(t for t in file_tools(scope) if t.name == "read_file")
    placeholder = delegate_tool()
    policy = PermissionPolicy(
        {"read_file": read, "delegate_read": placeholder}, validators or {}
    )
    controller = DelegationController(
        model=model,
        checkpointer=saver,
        settings=settings,
        scope=scope,
        policy=policy,
        system_prompt="Trusted profile",
    )
    delegate = delegate_tool(controller)
    policy = PermissionPolicy(
        {"read_file": read, "delegate_read": delegate}, validators or {}
    )
    usage = ReportedUsage()
    graph = build_agent(
        model,
        checkpointer=saver,
        settings=settings,
        tools=[read, delegate],
        permission_policy=policy,
        middleware=[*middleware, DelegationDiagnostics(controller, usage, schema)],
        state_schema=schema,
    )
    cfg = run_config(str(uuid4()))
    cfg["callbacks"] = [usage]
    return graph, cfg, controller, reads, usage


async def invoke(graph, cfg, **state):
    return await graph.ainvoke(
        {"messages": [HumanMessage("private parent history")], **state},
        cfg,
        context=RunContext(cfg["configurable"]["thread_id"]),
        durability="sync",
    )


def test_native_subgraphs_isolate_history_permissions_namespaces_and_usage():
    async def check():
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "first task"}, "d1"),
                call("read_file", {"path": "ok"}, "read"),
                AIMessage("first finding"),
                call("delegate_read", {"task": "second task"}, "d2"),
                AIMessage("second finding"),
                AIMessage("parent answer"),
            ]
        )
        graph, cfg, controller, reads, usage = harness(model)
        state = await invoke(graph, cfg)
        assert state["messages"][-1].text == "parent answer"
        assert len(controller.started) == 2 and len(reads) == 1
        assert len(state["delegations"]) == 2
        namespaces = {r["namespace"] for r in state["delegations"]}
        assert len(namespaces) == 2 and all(namespaces)
        for record in state["delegations"]:
            cp = await graph.checkpointer.aget_tuple(record["final_checkpoint"])
            assert cp.metadata["delegation_id"] == record["delegation_id"]
            assert cp.metadata["parent_run_id"] == cfg["configurable"]["thread_id"]
            assert cp.metadata["original_tool_call_id"] in {"d1", "d2"}
        assert [r["model_calls"] for r in state["delegations"]] == [2, 1]
        assert [r["tool_calls"].get("__all__", 0) for r in state["delegations"]] == [
            1,
            0,
        ]
        children = [m for m in model.history if constraints(m)["agent_depth"] == 1]
        assert {tuple(constraints(m)["tools"]) for m in children} == {("read_file",)}
        assert all(
            not any("private parent history" in x.text for x in m) for m in children
        )
        assert [m.text for m in children[-1][1:]] == ["second task"]
        assert all("Trusted profile" in m[0].text for m in children)
        assert all("1: needle" not in m.text for m in state["messages"])
        assert usage.missing == 6 and state["reported_usage"]["complete"] is False
        assert [
            r["reported_usage"]["missing_reports"] for r in state["delegations"]
        ] == [2, 1]
        root = await graph.aget_state(cfg)
        assert root.config["configurable"]["checkpoint_ns"] == ""

    asyncio.run(check())


@pytest.mark.parametrize("task", ["", "   ", "中" * 2731])
def test_delegate_task_utf8_limit(task):
    with pytest.raises(ValidationError):
        DelegateInput(task=task)


def test_custom_delegation_requires_workspace_reader_and_chat_unchanged(tmp_path):
    with pytest.raises(ValueError, match="requires a workspace"):
        Registry(
            RuntimeFile(
                version=1, profiles=[Profile(id="bad", tools=("delegate_read",))]
            )
        )
    registry = Registry()
    assert registry.profiles["chat"].tools == ()
    for name in ("workspace_read", "workspace_write"):
        assert "delegate_read" in registry.profiles[name].tools


@pytest.mark.parametrize("mixed", [False, True])
def test_batch_rejection_does_not_start_child_and_normal_tools_continue(mixed):
    async def check():
        requests = [
            call("delegate_read", {"task": "task"}, "d1").tool_calls[0],
            call(
                "read_file" if mixed else "delegate_read",
                {"path": "ok"} if mixed else {"task": "task"},
                "d2",
            ).tool_calls[0],
        ]
        model = SyntheticModel(
            responses=[AIMessage("", tool_calls=requests), AIMessage("done")]
        )
        graph, cfg, controller, reads, _ = harness(model)
        state = await invoke(graph, cfg)
        assert not controller.started and len(reads) == int(mixed)
        errors = [
            m
            for m in state["messages"]
            if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert len(errors) == (1 if mixed else 2)
        assert all("SUBAGENT_BATCH_REJECTED" in m.text for m in errors)

    asyncio.run(check())


def test_hook_retarget_cannot_bypass_batch_entry_check():
    class Retarget(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            if request.tool_call["id"] == "retarget":
                target = next(
                    t for t in request.runtime.tools if t.name == "delegate_read"
                )
                request = request.override(
                    tool=target,
                    tool_call={
                        **request.tool_call,
                        "name": "delegate_read",
                        "args": {"task": "must reject"},
                    },
                )
            return await handler(request)

    async def check():
        model = SyntheticModel(
            responses=[
                AIMessage(
                    "",
                    tool_calls=[
                        call("read_file", {"path": "a"}, "retarget").tool_calls[0],
                        call("read_file", {"path": "b"}, "ordinary").tool_calls[0],
                    ],
                ),
                AIMessage("done"),
            ]
        )
        graph, cfg, controller, reads, _ = harness(model, middleware=[Retarget()])
        state = await invoke(graph, cfg)
        assert not controller.started and len(reads) == 1
        assert any("SUBAGENT_BATCH_REJECTED" in m.text for m in state["messages"])

    asyncio.run(check())


@pytest.mark.parametrize("failure", ["quota", "model", "invalid"])
def test_child_local_failures_consume_start_and_parent_can_finish(failure):
    async def check():
        child_calls = 0

        def reply(messages):
            nonlocal child_calls
            if constraints(messages)["agent_depth"] == 1:
                child_calls += 1
                if failure == "model":
                    raise openai.RateLimitError(
                        "secret",
                        response=httpx.Response(
                            429, request=httpx.Request("POST", "http://secret")
                        ),
                        body=None,
                    )
                if failure == "quota":
                    return call("read_file", {"path": "ok"}, str(child_calls))
                return AIMessage("")
            last = messages[-1]
            if isinstance(last, ToolMessage):
                return AIMessage("parent recovered")
            return call("delegate_read", {"task": "investigate"}, "d")

        graph, cfg, controller, reads, _ = harness(SyntheticModel(reply=reply))
        state = await invoke(graph, cfg)
        assert (
            state["messages"][-1].text == "parent recovered"
            and len(controller.started) == 1
        )
        code = {
            "quota": "QUOTA_EXCEEDED",
            "model": "MODEL_FAILED",
            "invalid": "INVALID_OUTPUT",
        }[failure]
        assert any("SUBAGENT_" + code in m.text for m in state["messages"])
        assert all("secret" not in m.text for m in state["messages"])

    asyncio.run(check())


def test_child_validators_denial_and_recursive_or_write_tools_unavailable():
    async def check():
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "investigate"}, "d"),
                call("read_file", {"path": "denied"}, "read"),
                call("shell", {"command": "touch x"}, "shell"),
                call("delegate_read", {"task": "nested"}, "nested"),
                call("edit_file", {"path": "x", "new_text": "x"}, "write"),
                AIMessage("permission checked"),
                AIMessage("done"),
            ]
        )
        graph, cfg, controller, reads, _ = harness(
            model, validators={"read_file": lambda args, ctx: args.path != "denied"}
        )
        state = await invoke(graph, cfg)
        assert not reads and len(controller.started) == 1
        cp = await graph.checkpointer.aget_tuple(
            state["delegations"][0]["final_checkpoint"]
        )
        errors = [
            m
            for m in cp.checkpoint["channel_values"]["messages"]
            if isinstance(m, ToolMessage)
        ]
        assert len(errors) == 4 and all(m.status == "error" for m in errors)

    asyncio.run(check())


def test_duplicate_ids_and_hook_dropped_diagnostics_do_not_reset_authority():
    class Drop(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            result = await handler(request)
            if hasattr(result, "update"):
                return replace(
                    result,
                    update={"messages": result.update["messages"], "delegations": []},
                )
            return result

    async def check():
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "a"}, "same"),
                AIMessage("a"),
                call("delegate_read", {"task": "repeat"}, "same"),
                call("delegate_read", {"task": "b"}, "other"),
                AIMessage("b"),
                call("delegate_read", {"task": "third"}, "third"),
                AIMessage("done"),
            ]
        )
        graph, cfg, controller, _, _ = harness(model, middleware=[Drop()])
        state = await invoke(graph, cfg)
        assert len(controller.started) == len(state["delegations"]) == 2
        assert any("SUBAGENT_DUPLICATE_CALL" in m.text for m in state["messages"])
        assert any("SUBAGENT_LIMIT_REACHED" in m.text for m in state["messages"])

    asyncio.run(check())


def test_postgres_docker_delegation_trace_and_fresh_conversation_state(
    database, tmp_path
):
    from tests.test_execution import install
    from tests.test_core import execute
    from tests.test_context import skill_file
    from agent_service.profiles import SkillsConfig, Workspace
    from agent_service.worker import Worker
    from fastapi import FastAPI
    from agent_service.api import router

    workspace, skills = tmp_path / "workspace", tmp_path / "skills"
    workspace.mkdir()
    (workspace / "text").write_text("evidence")
    skill_file(skills)

    async def check(r):
        install(r, workspace)
        r.registry = r.store.registry = Registry(
            RuntimeFile(
                version=1,
                execution=r.registry.execution,
                workspaces=[Workspace(id="source", root=str(workspace))],
                skills=SkillsConfig(directories=(str(skills),)),
            )
        )
        conv = await r.store.create_conversation(
            "delegation", "workspace_write", "source"
        )

        def reported(text):
            return AIMessage(
                text,
                usage_metadata={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
                response_metadata={"model_name": "synthetic"},
            )

        model = SyntheticModel(
            responses=[
                call(
                    "write_todos",
                    {"todos": [{"content": "investigate", "status": "in_progress"}]},
                    "todo",
                ),
                call("delegate_read", {"task": "find evidence"}, "d1"),
                call("read_file", {"path": "text"}, "r"),
                call("read_skill", {"name": "review"}, "s"),
                reported("first report"),
                call("delegate_read", {"task": "independent check"}, "d2"),
                reported("second report"),
                reported("final"),
                call("delegate_read", {"task": "next Run"}, "d1"),
                reported("fresh report"),
                reported("next final"),
            ]
        )
        worker = Worker(r, model=model)
        await r.store.submit("investigate", conversation_id=conv["id"])
        first = await execute(r, worker)
        assert first["status"] == "completed", first
        assert first["final_checkpoint"]["configurable"]["checkpoint_ns"] == ""
        root = await worker.agent.aget_state(first["final_checkpoint"])
        assert len(root.values["delegations"]) == 2
        usage = root.values["reported_usage"]
        assert usage["models"]["synthetic"]["total_tokens"] == 15
        assert usage["missing_reports"] == 5 and not usage["complete"]
        records = root.values["delegations"]
        assert [
            x["reported_usage"]["models"]["synthetic"]["total_tokens"] for x in records
        ] == [5, 5]
        app = FastAPI()
        app.state.resources = r
        app.include_router(router)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            all_trace = (
                await client.get(f"/v1/runs/{first['id']}/trace", params={"limit": 200})
            ).json()
            spaces = {
                cp["config"]["configurable"]["checkpoint_ns"]
                for cp in all_trace["checkpoints"]
            }
            assert spaces == {"", *(rec["namespace"] for rec in records)}
            child_trace = (
                await client.get(
                    f"/v1/runs/{first['id']}/trace",
                    params={"namespace": records[0]["namespace"]},
                )
            ).json()
            assert child_trace["checkpoints"] and all(
                cp["config"]["configurable"]["checkpoint_ns"] == records[0]["namespace"]
                for cp in child_trace["checkpoints"]
            )
        await r.store.submit("next task", conversation_id=conv["id"])
        second = await execute(r, worker)
        assert second["status"] == "completed", second
        final = await worker.agent.aget_state(second["final_checkpoint"])
        assert len(final.values["delegations"]) == 1 and not final.values.get("todos")
        assert final.values["thread_tool_call_count"]["__all__"] == 1
        assert all(
            rec["delegation_id"] != final.values["delegations"][0]["delegation_id"]
            for rec in records
        )
        children = [m for m in model.history if constraints(m)["agent_depth"] == 1]
        assert all(constraints(m)["current_run_todos"] == [] for m in children)
        assert all(constraints(m)["workspace_access"] == "read" for m in children)
        assert [m.text for m in children[-1][1:]] == ["next Run"]
        assert (await r.store.execution_resource(second["id"]))["state"] == "released"

    database(check)


@pytest.mark.parametrize("stage", ["model", "read"])
@pytest.mark.parametrize("control", ["cancel", "timeout", "lease", "stop"])
def test_child_wait_propagates_run_controls_and_cleans_shared_container(
    database, tmp_path, monkeypatch, stage, control
):
    from agent_service.execution import ExecutionScope, DockerClient
    from agent_service.worker import Worker
    from tests.test_execution import install

    async def check(r):
        install(r, tmp_path)
        started, stopped = asyncio.Event(), asyncio.Event()

        async def waiting(self, name, args):
            started.set()
            try:
                await self.client.call(
                    "exec",
                    "-i",
                    self.container_id,
                    "/usr/local/bin/python",
                    "-I",
                    "-c",
                    "import time; time.sleep(30)",
                    timeout=35,
                )
            finally:
                stopped.set()

        async def reply(messages):
            if constraints(messages)["agent_depth"] == 0:
                return call("delegate_read", {"task": "wait"}, "delegate")
            if stage == "read":
                return call("read_file", {"path": "text"}, "read")
            started.set()
            try:
                await asyncio.sleep(30)
                return AIMessage("late")
            finally:
                stopped.set()

        if stage == "read":
            monkeypatch.setattr(ExecutionScope, "file_call", waiting)
        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        row, _ = await r.store.submit(
            "control", profile="workspace_write", workspace_id="source"
        )
        claim = await r.store.claim(worker.id)
        task = asyncio.create_task(worker.execute(claim))
        try:
            await asyncio.wait_for(started.wait(), 10)
            if control == "cancel":
                await r.store.cancel(row["id"])
            elif control == "stop":
                task.cancel()
            elif control == "lease":
                async with r.pool.connection() as conn:
                    await conn.execute(
                        "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                        (row["id"],),
                    )
            outcome = await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), 12
            )
            assert outcome == [None] or (
                control == "stop" and isinstance(outcome[0], asyncio.CancelledError)
            ), outcome
            result = await r.store.get(row["id"])
            if control == "cancel":
                assert result["status"] == "cancelled"
            else:
                assert (
                    result["error_code"]
                    == {
                        "timeout": "RUN_TIMEOUT",
                        "lease": "LEASE_EXPIRED",
                        "stop": "WORKER_STOPPED",
                    }[control]
                ), result
            assert stopped.is_set() and len(model.history) == 2
            resource = await r.store.execution_resource(row["id"])
            assert resource["state"] == "released", resource
            assert await DockerClient().inspect(resource["container_id"]) is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    database(
        check,
        replace(
            Settings(),
            heartbeat_seconds=0.03,
            lease_seconds=4,
            run_timeout=15 if control == "timeout" else 30,
        ),
    )


@pytest.mark.parametrize(
    "failure", ["command-timeout", "container-lost", "oom", "unknown"]
)
def test_child_read_environment_failures_are_fatal_despite_parent_hook(
    database, tmp_path, monkeypatch, failure
):
    import os
    from agent_service.execution import ExecutionScope
    from agent_service.profiles import Extension
    from agent_service.worker import Worker
    from tests.test_execution import install
    from tests.test_core import execute

    class Swallow(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            try:
                return await handler(request)
            except Exception:
                return ToolMessage(
                    content="swallowed", tool_call_id=request.tool_call["id"]
                )

    async def check(r):
        cfg = ExecutionSettings(
            image=os.environ["AGENT_TEST_EXECUTION_IMAGE"],
            command_timeout=1 if failure == "command-timeout" else 30,
            memory_bytes=64 * 1024 * 1024,
        )
        install(
            r,
            tmp_path,
            execution=cfg,
            extensions={
                "workspace_write": Extension(middleware_factory=lambda: [Swallow()])
            },
        )
        original = ExecutionScope.file_call

        async def broken(self, name, args):
            if failure == "unknown":
                raise RuntimeError("unknown child failure")
            if failure == "container-lost":
                await self.client.call("rm", "-f", self.container_id)
                return await original(self, name, args)
            script = (
                "import time; time.sleep(20)"
                if failure == "command-timeout"
                else "a=[]\nwhile True: a.append(bytearray(8*1024*1024))"
            )
            code, _, _ = await self.client.call(
                "exec",
                "-i",
                self.container_id,
                "/usr/local/bin/python",
                "-I",
                "-c",
                script,
                timeout=30,
                check=False,
            )
            if code:
                from agent_service.execution_errors import EnvironmentLost

                raise EnvironmentLost("Execution helper was killed")
            await self.healthy()
            return {"content": "unreachable"}

        monkeypatch.setattr(ExecutionScope, "file_call", broken)
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "read"}, "d"),
                call("read_file", {"path": "text"}, "r"),
                AIMessage("must not publish"),
            ]
        )
        await r.store.submit("fatal", profile="workspace_write", workspace_id="source")
        row = await execute(r, Worker(r, model=model))
        assert (
            row["error_code"]
            == {
                "command-timeout": "COMMAND_TIMEOUT",
                "container-lost": "EXECUTION_ENVIRONMENT_LOST",
                "oom": "EXECUTION_ENVIRONMENT_LOST",
                "unknown": "EXECUTION_FAILED",
            }[failure]
        ), row
        assert len(model.history) == 2 and row["answer"] is None
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"

    database(check)


@pytest.mark.parametrize("ack", ["checkpoint", "diagnostic", "publication"])
def test_child_persistence_ack_loss_never_replays_models_or_tools(
    database, tmp_path, monkeypatch, ack
):
    from agent_service.worker import Worker
    from tests.test_execution import install
    from tests.test_core import execute

    async def check(r):
        install(r, tmp_path)
        (tmp_path / "text").write_text("data")
        put, finish = r.checkpointer.aput, r.store.finish
        triggered = False

        async def lost_put(config, checkpoint, metadata, new_versions):
            nonlocal triggered
            result = await put(config, checkpoint, metadata, new_versions)
            child = bool(config["configurable"].get("checkpoint_ns"))
            values = checkpoint["channel_values"]
            if not triggered and (
                (
                    ack == "checkpoint"
                    and child
                    and any(
                        isinstance(m, AIMessage) and m.text == "report"
                        for m in values.get("messages", [])
                    )
                )
                or (ack == "diagnostic" and not child and values.get("delegations"))
            ):
                triggered = True
                raise OSError("synthetic checkpoint acknowledgement loss")
            return result

        async def lost_finish(*args, **kwargs):
            nonlocal triggered
            result = await finish(*args, **kwargs)
            if not triggered:
                triggered = True
                raise OSError("synthetic publication acknowledgement loss")
            return result

        monkeypatch.setattr(r.checkpointer, "aput", lost_put)
        if ack == "publication":
            monkeypatch.setattr(r.store, "finish", lost_finish)
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "read"}, "d"),
                call("read_file", {"path": "text"}, "r"),
                AIMessage("report"),
                AIMessage("final"),
            ]
        )
        await r.store.submit("ack", profile="workspace_read", workspace_id="source")
        row = await execute(r, Worker(r, model=model))
        assert triggered
        assert row["status"] == ("completed" if ack == "publication" else "failed"), row
        assert len(model.history) == (4 if ack == "publication" else 3)
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"

    database(check)


def test_summary_does_not_reset_start_quota_and_usage_includes_summaries_once():
    async def check():
        parent = summaries = 0

        def reply(messages):
            nonlocal parent, summaries
            if "Context Extraction Assistant" in messages[0].text:
                summaries += 1
                answer = AIMessage("Previous investigations returned reports.")
            elif constraints(messages)["agent_depth"] == 1:
                answer = AIMessage("evidence " * 1500)
            else:
                parent += 1
                answer = (
                    call("delegate_read", {"task": "investigate"}, str(parent))
                    if parent < 4
                    else AIMessage("done")
                )
            return answer.model_copy(
                update={
                    "usage_metadata": {
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "total_tokens": 5,
                    },
                    "response_metadata": {"model_name": "synthetic"},
                }
            )

        model = SyntheticModel(reply=reply)
        graph, cfg, controller, _, usage = harness(
            model,
            settings=replace(
                Settings(),
                summary_trigger_tokens=180,
                summary_keep_tokens=40,
                summary_trim_tokens=1500,
            ),
        )
        state = await invoke(graph, cfg)
        assert len(controller.started) == 2 and summaries >= 1
        assert state["messages"][-1].text == "done"
        assert usage.report()["complete"]
        assert (
            usage.report()["models"]["synthetic"]["total_tokens"]
            == len(model.history) * 5
        )
        assert state["reported_usage"] == usage.report()

    asyncio.run(check())


def test_delegation_permission_and_invalid_input_do_not_consume_starts():
    async def check():
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "denied"}, "deny"),
                call("delegate_read", {"task": ""}, "invalid"),
                call("delegate_read", {"task": "allowed"}, "ok"),
                AIMessage("child"),
                AIMessage("parent"),
            ]
        )
        graph, cfg, controller, _, _ = harness(
            model, validators={"delegate_read": lambda args, ctx: args.task != "denied"}
        )
        state = await invoke(graph, cfg)
        assert controller.started == {"ok"}
        assert state["messages"][-1].text == "parent"

    asyncio.run(check())


def test_parent_quota_remains_independent_after_child_quota_failure():
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError

    async def check():
        parent_calls = child_calls = 0

        def reply(messages):
            nonlocal parent_calls, child_calls
            if constraints(messages)["agent_depth"] == 1:
                child_calls += 1
                return call("read_file", {"path": "ok"}, str(child_calls))
            parent_calls += 1
            return (
                call("delegate_read", {"task": "loop"}, "d")
                if parent_calls == 1
                else call("read_file", {"path": "ok"}, "p" + str(parent_calls))
            )

        graph, cfg, controller, reads, _ = harness(SyntheticModel(reply=reply))
        with pytest.raises(ToolCallLimitExceededError):
            await invoke(graph, cfg)
        assert controller.records[0]["error_code"] == "SUBAGENT_QUOTA_EXCEEDED"
        assert len(reads) == 15 and parent_calls == 11 and child_calls == 7

    asyncio.run(check())


def test_delegation_combines_custom_state_and_structured_response(database, tmp_path):
    from langchain.agents import AgentState
    from langchain.agents.structured_output import ProviderStrategy
    from pydantic import BaseModel
    from agent_service.profiles import Extension
    from agent_service.worker import Worker
    from tests.test_execution import install
    from tests.test_core import execute

    class CustomState(AgentState):
        business: str

    class Business(AgentMiddleware):
        state_schema = CustomState

        async def abefore_agent(self, state, runtime):
            return {"business": "root-only"}

    class Answer(BaseModel):
        result: str

    async def check(r):
        install(
            r,
            tmp_path,
            extensions={
                "workspace_write": Extension(
                    middleware_factory=lambda: [Business()],
                    state_schema=CustomState,
                    response_format=ProviderStrategy(Answer),
                )
            },
        )
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "analyze"}, "d"),
                AIMessage("plain child result"),
                AIMessage('{"result":"done"}'),
            ]
        )
        await r.store.submit(
            "structured", profile="workspace_write", workspace_id="source"
        )
        worker = Worker(r, model=model)
        row = await execute(r, worker)
        assert row["status"] == "completed" and json.loads(row["answer"]) == {
            "result": "done"
        }, row
        state = await worker.agent.aget_state(row["final_checkpoint"])
        assert state.values["business"] == "root-only"
        cp = await r.checkpointer.aget_tuple(
            state.values["delegations"][0]["final_checkpoint"]
        )
        assert "business" not in cp.checkpoint["channel_values"]
        assert "structured_response" not in cp.checkpoint["channel_values"]

    database(check)


def test_two_parent_runs_delegate_without_extra_slots_or_command_timeout(
    database, tmp_path
):
    from agent_service.worker import Worker
    from agent_service.profiles import Workspace
    from tests.test_execution import install

    async def check(r):
        roots = [tmp_path / "a", tmp_path / "b"]
        for root in roots:
            root.mkdir()
        install(
            r,
            roots[0],
            execution=ExecutionSettings(
                image=__import__("os").environ["AGENT_TEST_EXECUTION_IMAGE"],
                command_timeout=0.1,
            ),
            workspaces=(Workspace(id="other", root=str(roots[1])),),
        )
        ready, release = asyncio.Event(), asyncio.Event()
        count = 0

        async def reply(messages):
            nonlocal count
            if constraints(messages)["agent_depth"] == 1:
                count += 1
                if count == 2:
                    ready.set()
                await release.wait()
                await asyncio.sleep(0.2)
                return AIMessage("report")
            if isinstance(messages[-1], ToolMessage):
                return AIMessage("done")
            return call("delegate_read", {"task": "independent"}, "d")

        worker = Worker(r, model=SyntheticModel(reply=reply))
        for name in ("source", "other"):
            await r.store.submit(
                "concurrent", profile="workspace_write", workspace_id=name
            )
        await worker.tick()
        try:
            await asyncio.wait_for(ready.wait(), 10)
            assert len(worker.tasks) == 2
            async with r.pool.connection() as conn:
                assert (
                    await (
                        await conn.execute("SELECT count(*) AS n FROM agent.runs")
                    ).fetchone()
                )["n"] == 2
                assert (
                    await (
                        await conn.execute(
                            "SELECT count(*) AS n FROM agent.execution_resources"
                        )
                    ).fetchone()
                )["n"] == 2
            release.set()
            await asyncio.wait_for(asyncio.gather(*worker.tasks.values()), 15)
            for identity in worker.tasks:
                assert (await r.store.get(identity))["status"] == "completed"
                assert (await r.store.execution_resource(identity))[
                    "state"
                ] == "released"
        finally:
            release.set()
            for task in worker.tasks.values():
                task.cancel()
            await asyncio.gather(*worker.tasks.values(), return_exceptions=True)

    database(check)


def test_worker_sigkill_during_child_never_resumes_checkpoint(database, tmp_path):
    import subprocess
    import sys
    from agent_service.execution import DockerClient
    from agent_service.worker import Worker
    from agent_service.trace import trace_config
    from tests.test_execution import install

    async def check(r):
        install(r, tmp_path)
        submitted, _ = await r.store.submit(
            "crash child", profile="workspace_write", workspace_id="source"
        )
        script = """
import asyncio, json, sys
from pathlib import Path
from langchain_core.messages import AIMessage
from agent_service.resources import Resources
from agent_service.config import Settings
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_profiles_policy import call
from tests.test_execution import install
async def main():
    async with Resources(Settings(heartbeat_seconds=.05, lease_seconds=2)) as r:
        root = Path(sys.argv[1])
        install(r, root)
        async def reply(messages):
            context = json.loads(messages[0].text.split("Run constraints and progress:\\n")[-1])
            if context["agent_depth"] == 1:
                (root / "ready").write_text("child")
                await asyncio.sleep(30)
                return AIMessage("late")
            return call("delegate_read", {"task": "wait"}, "d")
        worker = Worker(r, model=SyntheticModel(reply=reply))
        row = await r.store.claim(worker.id)
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
            checkpoints = [
                cp async for cp in r.checkpointer.alist(trace_config(submitted["id"]))
            ]
            assert any(
                cp.config["configurable"]["checkpoint_ns"]
                and cp.metadata.get("original_tool_call_id") == "d"
                for cp in checkpoints
            )
            child.kill()
            await asyncio.to_thread(child.wait)
            resource = await r.store.execution_resource(submitted["id"])
            async with r.pool.connection() as conn:
                await conn.execute(
                    "UPDATE agent.runs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                    (submitted["id"],),
                )
            model = SyntheticModel()
            worker = Worker(r, model=model)
            await worker.tick()
            assert not model.history and not worker.tasks
            row = await r.store.get(submitted["id"])
            assert (
                row["error_code"] == "LEASE_EXPIRED" and row["final_checkpoint"] is None
            )
            assert await DockerClient().inspect(resource["container_id"]) is None
            assert (await r.store.execution_resource(row["id"]))["state"] == "released"
        finally:
            if child.poll() is None:
                child.kill()
                await asyncio.to_thread(child.wait)

    database(check)


@pytest.mark.parametrize("failure", ["quota", "sdk-timeout", "sdk-rate"])
def test_local_child_failure_keeps_real_parent_container_healthy_for_second_delegate(
    database, tmp_path, failure
):
    from agent_service.worker import Worker
    from tests.test_execution import install
    from tests.test_core import execute

    async def check(r):
        install(r, tmp_path)
        (tmp_path / "text").write_text("evidence")
        reads = 0

        def reply(messages):
            nonlocal reads
            if constraints(messages)["agent_depth"] == 1:
                if messages[1].text == "recover":
                    return AIMessage("recovered child")
                if failure == "quota":
                    reads += 1
                    return call("read_file", {"path": "text"}, str(reads))
                if failure == "sdk-timeout":
                    raise openai.APITimeoutError(
                        request=httpx.Request("POST", "http://secret")
                    )
                raise openai.RateLimitError(
                    "secret",
                    response=httpx.Response(
                        429, request=httpx.Request("POST", "http://secret")
                    ),
                    body=None,
                )
            if isinstance(messages[-1], ToolMessage):
                return (
                    AIMessage("done")
                    if messages[-1].tool_call_id == "recover"
                    else call("delegate_read", {"task": "recover"}, "recover")
                )
            return call("delegate_read", {"task": "fail"}, "fail")

        await r.store.submit(
            "recover", profile="workspace_write", workspace_id="source"
        )
        worker = Worker(r, model=SyntheticModel(reply=reply))
        row = await execute(r, worker)
        assert row["status"] == "completed", row
        state = await worker.agent.aget_state(row["final_checkpoint"])
        records = state.values["delegations"]
        assert (
            len(records) == 2
            and records[0]["status"] == "failed"
            and records[1]["status"] == "completed"
        )
        assert records[0]["error_code"] == (
            "SUBAGENT_QUOTA_EXCEEDED" if failure == "quota" else "SUBAGENT_MODEL_FAILED"
        )
        assert all("secret" not in m.text for m in state.values["messages"])
        assert state.values["thread_tool_call_count"]["__all__"] == 2
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"

    database(check)


def test_depth_guard_at_actual_entry_and_only_task_is_public():
    async def check():
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "nested"}, "d"),
                AIMessage("done"),
            ]
        )
        graph, cfg, controller, _, _ = harness(model)
        state = await graph.ainvoke(
            {"messages": [HumanMessage("test")]},
            cfg,
            context=RunContext(cfg["configurable"]["thread_id"], agent_depth=1),
        )
        assert not controller.started
        assert any("SUBAGENT_DEPTH_EXCEEDED" in m.text for m in state["messages"])
        assert set(delegate_tool().tool_call_schema.model_fields) == {"task"}
        with pytest.raises(ValidationError):
            DelegateInput(task="valid", profile="workspace_write")
        assert DelegateInput(task="a" * 8192).task

    asyncio.run(check())


def test_sdk_error_from_child_tool_is_not_misclassified_as_model_failure():
    async def check():
        error = openai.APITimeoutError(request=httpx.Request("POST", "http://secret"))

        async def file_call(name, args):
            raise error

        async def healthy():
            return None

        scope = SimpleNamespace(
            settings=ExecutionSettings(),
            lock=asyncio.Lock(),
            check_open=lambda: None,
            remaining=lambda: 300,
            healthy=healthy,
            file_call=file_call,
        )
        model = SyntheticModel(
            responses=[
                call("delegate_read", {"task": "read"}, "d"),
                call("read_file", {"path": "text"}, "r"),
                AIMessage("must not run"),
            ]
        )
        graph, cfg, controller, _, _ = harness(model, scope=scope)
        with pytest.raises(openai.APITimeoutError) as caught:
            await invoke(graph, cfg)
        assert caught.value is error and len(model.history) == 2
        assert len(controller.started) == 1

    asyncio.run(check())
