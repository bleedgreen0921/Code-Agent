"""Stage three: real native middleware, explicit snapshot and state boundaries."""

import asyncio
from dataclasses import replace
import json
import os

import pytest
from langchain.agents.middleware import AgentMiddleware, AgentState, TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import NotRequired

from agent_service.config import Settings
from agent_service.context import inherited_state, TODO_LIMITS
from agent_service.execution_config import ExecutionSettings
from agent_service.policy import ToolLifecycleMiddleware
from agent_service.profiles import (
    ConfigurationChanged,
    Extension,
    Profile,
    Registry,
    RuntimeFile,
    SkillsConfig,
    Workspace,
    fingerprint,
)
from agent_service.runtime import build_agent, RunContext, run_config
from agent_service.skills import load_skills, read_skill_tool
from agent_service.worker import Worker
from tests.models import SyntheticModel
from tests.test_core import execute
from tests.test_profiles_policy import call
from tests.test_runtime_bindings import install


def skill_file(
    root,
    name="review",
    body="PRIVATE SKILL BODY\n",
    description="Review code",
    metadata="",
):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / "SKILL.md"
    file.write_text(
        f"---\nname: {name}\ndescription: {description}\n{metadata}---\n{body}",
        encoding="utf-8",
    )
    return file


def config(root, *, profiles=(), execution=None, workspaces=()):
    return RuntimeFile(
        version=1,
        skills=SkillsConfig(directories=(str(root),)),
        profiles=list(profiles),
        execution=execution or ExecutionSettings(),
        workspaces=list(workspaces),
    )


def memory_profile(names=("review",), tools=("write_todos", "read_skill")):
    return Profile(id="memory", tools=tools, skills=names)


def constraints(messages):
    return json.loads(messages[0].text.split("Run constraints and progress:\n")[-1])


def test_skills_snapshot_bindings_and_legacy_chat_fingerprint(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    assert not load_skills((str(root),))
    empty = Registry(config(root))
    assert "write_todos" in empty.profiles["workspace_read"].tools
    assert "read_skill" not in empty.profiles["workspace_read"].tools
    assert empty.profiles["chat"].tools == ()
    # The new execution deadline policy is part of every profile, including chat.
    # Unrelated skills still must not change this fingerprint.
    old_profile = fingerprint(
        {
            "id": "chat",
            "system_prompt": None,
            "tools": [],
            "requires_workspace": False,
            "extension_version": "1",
            "tool_schemas": {},
            "reliability": empty.reliability.model_dump(),
        }
    )
    assert empty.bind()["profile_fingerprint"] == old_profile
    file = skill_file(root)
    skill_file(root, "other", "OTHER BODY\n")
    declarations = [memory_profile()]
    registry = Registry(config(root, profiles=declarations))
    binding = registry.bind("memory")
    file.write_text(file.read_text().replace("PRIVATE SKILL BODY", "CHANGED BODY"))
    assert registry.skills["review"].body == "PRIVATE SKILL BODY\n"
    registry.resolve(binding)
    with pytest.raises(ConfigurationChanged):
        Registry(config(root, profiles=declarations)).resolve(binding)
    file.write_text(file.read_text().replace("CHANGED BODY", "PRIVATE SKILL BODY"))
    skill_file(root, "other", "UNRELATED CHANGED\n")
    assert Registry(config(root, profiles=declarations)).bind("memory") == binding
    assert Registry(config(root)).bind() == empty.bind()
    with pytest.raises(TypeError):
        registry.skills["new"] = registry.skills["review"]
    for profile in [
        memory_profile(names=("unknown",)),
        Profile(id="memory", tools=("read_skill",)),
        Profile(id="memory", skills=("review",)),
    ]:
        with pytest.raises(ValueError):
            Registry(config(root, profiles=[profile]))


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "bad-yaml",
        "duplicate-key",
        "missing-name",
        "mismatch",
        "non-utf8",
        "escape-directory",
        "escape-file",
        "special",
        "long-body",
        "many-lines",
        "bad-description",
        "bad-compatibility",
        "bad-metadata",
        "missing-file",
    ],
)
def test_invalid_skill_catalog_fails_startup(tmp_path, failure):
    root = tmp_path / "skills"
    root.mkdir()
    file = skill_file(root)
    directories = (str(root),)
    if failure == "duplicate":
        other = tmp_path / "other"
        skill_file(other)
        directories += (str(other),)
    elif failure == "bad-yaml":
        file.write_text("---\nname: [\n---\nbody")
    elif failure == "duplicate-key":
        file.write_text("---\nname: review\nname: wrong\ndescription: test\n---\nbody")
    elif failure == "missing-name":
        file.write_text("---\ndescription: test\n---\nbody")
    elif failure == "mismatch":
        file.write_text("---\nname: wrong\ndescription: test\n---\nbody")
    elif failure == "non-utf8":
        file.write_bytes(b"\xff")
    elif failure == "escape-directory":
        file.unlink()
        file.parent.rmdir()
        outside = tmp_path / "outside"
        skill_file(outside)
        (root / "review").symlink_to(outside / "review", target_is_directory=True)
    elif failure == "escape-file":
        file.unlink()
        outside = tmp_path / "outside.md"
        outside.write_text("outside")
        file.symlink_to(outside)
    elif failure == "special":
        file.unlink()
        os.mkfifo(file)
    elif failure == "long-body":
        skill_file(root, body="中" * 6000)
    elif failure == "many-lines":
        skill_file(root, body="line\n" * 201)
    elif failure == "bad-description":
        skill_file(root, description="[]")
    elif failure == "bad-compatibility":
        skill_file(root, metadata="compatibility: []\n")
    elif failure == "bad-metadata":
        skill_file(root, metadata="metadata: {version: 1}\n")
    else:
        file.unlink()
    with pytest.raises(ValueError):
        load_skills(directories)


def test_skill_catalog_and_effective_output_limits(tmp_path):
    for n in range(9):
        skill_file(tmp_path, f"skill-{n}", description="d" * 1000)
    with pytest.raises(ValueError, match="catalog exceeds"):
        Registry(config(tmp_path))
    # A custom profile referencing only one of the registered skills fits its catalog,
    # but built-in workspace catalogs must also fit, so startup still fails.
    with pytest.raises(ValueError, match="catalog exceeds"):
        Registry(config(tmp_path, profiles=[memory_profile(names=("skill-0",))]))
    short = tmp_path / "skill-0" / "SKILL.md"
    short.write_text(short.read_text() + "z" * 600)
    with pytest.raises(ValueError, match="shorten"):
        load_skills((str(tmp_path),), output_bytes=512)


def test_native_todo_limits_parallel_rejection_and_custom_state():
    class State(AgentState):
        business: NotRequired[int]

    class Business(AgentMiddleware):
        state_schema = State

        async def abefore_model(self, state, runtime):
            return {"business": state.get("business", 0) + 1}

    class Answer(BaseModel):
        value: int

    async def check():
        good = [{"content": "inspect", "status": "in_progress"}]
        bad = [
            [{"content": "x", "status": "pending"}] * 21,
            [{"content": "x" * 201, "status": "pending"}],
            [{"content": "中" * 200, "status": "pending"}] * 20,
            [{"content": "x", "status": "invalid"}],
        ]
        model = SyntheticModel(
            responses=[
                call("write_todos", {"todos": good}),
                *[call("write_todos", {"todos": t}, str(i)) for i, t in enumerate(bad)],
                AIMessage("finished with pending work"),
            ]
        )
        graph = build_agent(
            model,
            middleware=[TodoListMiddleware(), Business()],
            state_schema=State,
            checkpointer=InMemorySaver(),
            output_limits=(512, 4),
        )
        result = await graph.ainvoke(
            {"messages": [HumanMessage("complex")]},
            run_config("todo"),
            context=RunContext("todo"),
        )
        assert result["todos"] == good and result["business"] == 6
        assert (await graph.aget_state(run_config("todo"))).values[
            "thread_tool_call_count"
        ]["__all__"] == 5
        assert result["messages"][-1].text == "finished with pending work"
        assert (
            len(
                [
                    m
                    for m in result["messages"]
                    if isinstance(m, ToolMessage) and m.status == "error"
                ]
            )
            == 4
        )
        assert constraints(model.history[0])["current_run_todos"] == []
        assert constraints(model.history[-1])["current_run_todos"] == good
        parallel = call("write_todos", {"todos": good}, "one")
        parallel.tool_calls += call("write_todos", {"todos": []}, "two").tool_calls
        model = SyntheticModel(responses=[parallel, AIMessage("rejected")])
        result = await build_agent(model, middleware=[TodoListMiddleware()]).ainvoke(
            {"messages": [HumanMessage("test")]}, context=RunContext("parallel")
        )
        assert not result.get("todos")
        assert (
            len(
                [
                    m
                    for m in result["messages"]
                    if isinstance(m, ToolMessage) and m.status == "error"
                ]
            )
            == 2
        )
        # Native schema combination also preserves structured output with Todo enabled.
        model = SyntheticModel(
            responses=[
                call("write_todos", {"todos": good}),
                call("Answer", {"value": 7}),
            ]
        )
        result = await build_agent(
            model,
            middleware=[TodoListMiddleware(), Business()],
            state_schema=State,
            response_format=ToolStrategy(Answer),
        ).ainvoke(
            {"messages": [HumanMessage("test")]}, context=RunContext("structured")
        )
        assert (
            result["structured_response"] == Answer(value=7)
            and result["todos"] == good
            and result["business"] == 2
        )
        assert inherited_state(result).keys() == {"messages"}

    asyncio.run(check())


def test_command_message_limit_preserves_native_fields_and_todos():
    todos = [{"content": "中" * 200, "status": "pending"}] * 20
    artifact = ["native arbitrary artifact"]
    msg = ToolMessage(
        "long\n" * 500,
        tool_call_id="call",
        name="write_todos",
        status="error",
        artifact=artifact,
    )
    original = Command(
        graph=Command.PARENT,
        goto="next",
        resume={"key": "value"},
        update={"messages": [msg], "todos": todos, "business": 42},
    )
    middleware = ToolLifecycleMiddleware((512, 4))
    limited = middleware.limit_result(original)
    message = limited.update["messages"][0]
    assert (
        message.tool_call_id == "call"
        and message.name == "write_todos"
        and message.status == "error"
    )
    assert message.artifact is artifact
    assert (
        limited.goto == original.goto
        and limited.graph == original.graph
        and limited.resume == original.resume
    )
    assert limited.update["todos"] is todos and limited.update["business"] == 42
    assert len(message.text.encode()) <= 512 and len(message.text.splitlines()) <= 4
    assert len(original.update["messages"][0].text) > len(message.text)


def test_todo_command_truncated_but_state_complete():
    async def check():
        todos = [{"content": "x" * 200, "status": "pending"}] * 20
        result = await build_agent(
            SyntheticModel(
                responses=[call("write_todos", {"todos": todos}), AIMessage("answer")]
            ),
            middleware=[TodoListMiddleware()],
            output_limits=(512, 4),
        ).ainvoke({"messages": [HumanMessage("test")]}, context=RunContext("limits"))
        assert result["todos"] == todos
        assert result["messages"][-2].text.endswith("[output truncated]")

    asyncio.run(check())


def test_direct_factory_generated_tool_duplicates_fail_before_model():
    todo = TodoListMiddleware()
    with pytest.raises(ValueError, match="Duplicate"):
        build_agent(SyntheticModel(), tools=todo.tools, middleware=[todo])


def test_skill_final_arguments_after_hooks_and_ephemeral_trusted_context(
    tmp_path, caplog
):
    file = skill_file(tmp_path)
    skill = load_skills((str(tmp_path),))["review"]
    reader = read_skill_tool({"review": skill})

    class Rewrite(AgentMiddleware):
        async def awrap_tool_call(self, request, handler):
            return await handler(
                request.override(
                    tool_call={**request.tool_call, "args": {"name": "other"}}
                )
            )

        async def awrap_model_call(self, request, handler):
            return await handler(
                request.override(system_message=SystemMessage("extension prompt"))
            )

    async def check():
        model = SyntheticModel(
            responses=[call("read_skill", {"name": "review"}), AIMessage("done")]
        )
        result = await build_agent(
            model,
            tools=[reader],
            middleware=[Rewrite()],
            system_prompt="TRUSTED",
            skills=[skill],
        ).ainvoke({"messages": [HumanMessage("test")]}, context=RunContext("skills"))
        assert (
            result["messages"][-2].status == "error"
            and "SKILL_NOT_AVAILABLE" in result["messages"][-2].text
        )
        for request in model.history:
            assert (
                "TRUSTED" in request[0].text and "extension prompt" in request[0].text
            )
            assert "PRIVATE SKILL BODY" not in request[0].text
            assert constraints(request)["skill_catalog"] == [
                {"name": "review", "description": "Review code"}
            ]
        assert all(not isinstance(m, SystemMessage) for m in result["messages"])
        file.unlink()
        model = SyntheticModel(
            responses=[call("read_skill", {"name": "review"}), AIMessage("done")]
        )
        result = await build_agent(model, tools=[reader], skills=[skill]).ainvoke(
            {"messages": [HumanMessage("test")]}, context=RunContext("snapshot")
        )
        assert result["messages"][-2].text == skill.body
        assert result["messages"][-2].artifact == {
            "name": "review",
            "sha256": skill.sha256,
        }

    with caplog.at_level("INFO"):
        asyncio.run(check())
    records = [r for r in caplog.records if r.msg == "model_context"]
    assert records and all(
        r.request_tokens_approx > r.history_tokens_approx > 0 for r in records
    )


def test_toml_skill_config_and_optional_metadata(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    skill_file(
        root,
        metadata="license: MIT\ncompatibility: Local code\nallowed-tools: shell edit_file\nmetadata: {author: test}\n",
    )
    path = tmp_path / "runtime.toml"
    path.write_text(f'''version = 1
[skills]
directories = ["{root}"]
[[profiles]]
id = "memory"
tools = ["read_skill"]
skills = ["review"]
''')
    monkeypatch.setenv("AGENT_RUNTIME_CONFIG", str(path))
    registry = Registry.from_env()
    assert registry.profiles["memory"].tools == ("read_skill",)
    assert (
        '"allowed-tools": "shell edit_file"' in registry.skills["review"].metadata_json
    )
    assert registry.profiles["memory"].skills == ("review",)


def test_postgres_summary_todos_skills_trace_and_messages_only_inheritance(
    database, tmp_path
):
    skill_file(tmp_path, body="PRIVATE INSTRUCTIONS " * 300 + "\n")
    todos = [{"content": "Inspect and verify", "status": "in_progress"}]
    final_todos = [{**todos[0], "status": "completed"}]

    async def check(r):
        registry = Registry(config(tmp_path, profiles=[memory_profile()]))
        install(r, registry)
        reads = summaries = main = 0

        def reply(messages):
            nonlocal reads, summaries, main
            if "Context Extraction Assistant" in messages[0].text:
                summaries += 1
                assert "Distinguish assumptions from verified facts" in messages[0].text
                return AIMessage("Verified prior progress; continue the current task.")
            current = constraints(messages)
            assert current["workspace_network"] == "none"
            assert current["tools"] == ["read_skill", "write_todos"]
            assert current["skill_catalog"] == [
                {"name": "review", "description": "Review code"}
            ]
            assert current["deadline_at"] != "None"
            assert "PRIVATE INSTRUCTIONS" not in messages[0].text
            # Native summary partitioning must preserve complete tool-call/result pairs.
            outstanding = set()
            for message in messages[1:]:
                if isinstance(message, AIMessage):
                    outstanding.update(t["id"] for t in message.tool_calls)
                elif isinstance(message, ToolMessage):
                    assert message.tool_call_id in outstanding
                    outstanding.remove(message.tool_call_id)
            assert not outstanding
            main += 1
            if main == 1:
                assert current["current_run_todos"] == []
                assert all("PRIVATE INSTRUCTIONS" not in m.text for m in messages)
                return call("write_todos", {"todos": todos}, "plan")
            if main <= 6:
                assert current["current_run_todos"] == todos
                reads += 1
                return call("read_skill", {"name": "review"}, f"read-{reads}")
            if main == 7:
                assert current["current_run_todos"] == todos
                return call("write_todos", {"todos": final_todos}, "complete")
            assert current["current_run_todos"] == final_todos
            return AIMessage("The verified answer")

        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        conversation = await r.store.create_conversation("context", "memory")
        first, _ = await r.store.submit(
            "Perform a complex review " * 80, conversation_id=conversation["id"]
        )
        result = await execute(r, worker)
        assert (
            result["status"] == "completed"
            and result["answer"] == "The verified answer"
        )
        assert reads == 5 and summaries >= 2 and main == 8
        assert await r.store.execution_resource(first["id"]) is None
        final = await worker.agent.aget_state(result["final_checkpoint"])
        assert final.values["todos"] == final_todos
        assert final.values["thread_model_call_count"] == 8
        assert final.values["thread_tool_call_count"]["__all__"] == 7
        assert not any(isinstance(m, SystemMessage) for m in final.values["messages"])
        histories = [cp async for cp in r.checkpointer.alist(run_config(first["id"]))]
        assert any(
            cp.checkpoint["channel_values"].get("todos") == todos for cp in histories
        )
        returned = [
            m
            for cp in histories
            for m in cp.checkpoint["channel_values"].get("messages", [])
            if isinstance(m, ToolMessage) and m.name == "read_skill"
        ]
        assert returned and all(
            m.text == registry.skills["review"].body for m in returned
        )
        snapshot = result["runtime_config"]
        assert snapshot["todo_limits"] == TODO_LIMITS and snapshot[
            "visible_skills"
        ] == [registry.skills["review"].manifest()]
        assert "PRIVATE INSTRUCTIONS" not in json.dumps(snapshot)
        # The existing endpoint exposes native state, with no new progress fields.
        import httpx
        from agent_service.app import app

        app.state.resources = r
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://127.0.0.1",
        ) as client:
            trace = (await client.get(f"/v1/runs/{first['id']}/trace")).json()
        assert any(
            cp["checkpoint"]["channel_values"].get("todos") == final_todos
            for cp in trace["checkpoints"]
        )
        # Drop read_skill and its allowlist between successful conversation turns.
        install(
            r,
            Registry(
                config(
                    tmp_path, profiles=[Profile(id="memory", tools=("write_todos",))]
                )
            ),
        )
        followup = SyntheticModel(
            responses=[
                call("read_skill", {"name": "review"}),
                AIMessage("No current access"),
            ]
        )
        await r.store.submit("next", conversation_id=conversation["id"])
        second = await execute(r, Worker(r, model=followup))
        assert second["status"] == "completed"
        for messages in followup.history:
            if "Context Extraction Assistant" not in messages[0].text:
                assert constraints(messages)["current_run_todos"] == []
                assert constraints(messages)["skill_catalog"] == []
        final = await worker.agent.aget_state(second["final_checkpoint"])
        assert not final.values.get("todos")
        assert final.values["thread_model_call_count"] == 2
        assert final.values["thread_tool_call_count"]["__all__"] == 1
        assert final.values["messages"][-2].status == "error"

    database(
        check,
        replace(
            Settings(),
            summary_trigger_tokens=700,
            summary_keep_tokens=200,
            summary_trim_tokens=2000,
        ),
    )


def test_worker_snapshot_disk_changes_and_configuration_change(database, tmp_path):
    file = skill_file(tmp_path)

    async def check(r):
        cfg = config(tmp_path, profiles=[memory_profile()])
        registry = Registry(cfg)
        install(r, registry)
        file.write_text(file.read_text().replace("PRIVATE SKILL BODY", "NEW VERSION"))
        model = SyntheticModel(
            responses=[call("read_skill", {"name": "review"}), AIMessage("done")]
        )
        worker = Worker(r, model=model)
        await r.store.submit("use frozen snapshot", profile="memory")
        row = await execute(r, worker)
        assert row["status"] == "completed"
        final = await worker.agent.aget_state(row["final_checkpoint"])
        assert final.values["messages"][-2].text == "PRIVATE SKILL BODY\n"
        await r.store.submit("queued old binding", profile="memory")
        install(r, Registry(cfg))
        model = SyntheticModel()
        row = await execute(r, Worker(r, model=model))
        assert row["error_code"] == "CONFIGURATION_CHANGED" and model.history == []
        assert row["runtime_config"] is None

    database(check)


def test_worker_generated_tools_bound_and_undeclared_or_duplicate_rejected(database):
    from langchain.tools import tool

    @tool
    async def native(value: int) -> str:
        """Return a trusted value."""
        return str(value)

    class Provider(AgentMiddleware):
        def __init__(self):
            self.tools = [native.model_copy()]

    async def check(r):
        install(
            r,
            Registry(
                RuntimeFile(
                    version=1, profiles=[Profile(id="custom", tools=("native",))]
                ),
                tools=[native],
                extensions={
                    "custom": Extension(middleware_factory=lambda: (Provider(),))
                },
            ),
        )
        model = SyntheticModel(responses=[call("native", {"value": 5}), AIMessage("5")])
        worker = Worker(r, model=model)
        await r.store.submit("generated", profile="custom")
        row = await execute(r, worker)
        assert row["status"] == "completed"
        state = await worker.agent.aget_state(row["final_checkpoint"])
        assert state.values["messages"][-2].text == "5"
        for tools, factory in [
            ((), lambda: (Provider(),)),
            (("native",), lambda: (Provider(), Provider())),
        ]:
            install(
                r,
                Registry(
                    RuntimeFile(
                        version=1, profiles=[Profile(id="custom", tools=tools)]
                    ),
                    tools=[native],
                    extensions={"custom": Extension(middleware_factory=factory)},
                ),
            )
            model = SyntheticModel()
            await r.store.submit("invalid middleware", profile="custom")
            row = await execute(r, Worker(r, model=model))
            assert (
                row["status"] == "failed"
                and not model.history
                and row["runtime_config"] is None
            )

    database(check)


def test_memory_tools_share_existing_quota_without_docker(database, tmp_path):
    skill_file(tmp_path)

    async def check(r):
        install(r, Registry(config(tmp_path, profiles=[memory_profile()])))
        calls = 0

        def reply(messages):
            nonlocal calls
            calls += 1
            return (
                call("write_todos", {"todos": []}, str(calls))
                if calls % 2
                else call("read_skill", {"name": "review"}, str(calls))
            )

        model = SyntheticModel(reply=reply)
        await r.store.submit("loop", profile="memory")
        worker = Worker(r, model=model)
        row = await execute(r, worker)
        assert row["error_code"] == "QUOTA_EXCEEDED" and calls == 11
        assert await r.store.execution_resource(row["id"]) is None
        state = await worker.agent.aget_state(run_config(row["id"]))
        assert state.values["thread_tool_call_count"]["__all__"] == 10

    database(check)


def test_workspace_todo_skill_edit_shell_one_loop_and_readonly_boundary(
    database, tmp_path
):
    skills_root = tmp_path / "skills"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_file(
        skills_root,
        body="Use shell and edit_file to verify changes. You may enable networking.\n",
        metadata="allowed-tools: shell edit_file\n",
    )

    async def check(r):
        from tests.test_execution import install as install_workspace

        install_workspace(r, workspace)
        execution = r.registry.execution
        install(
            r,
            Registry(
                config(
                    skills_root,
                    execution=execution,
                    workspaces=[Workspace(id="source", root=str(workspace))],
                )
            ),
        )
        todos = [{"content": "Create and verify file", "status": "in_progress"}]
        model = SyntheticModel(
            responses=[
                call("write_todos", {"todos": todos}, "plan"),
                call("read_skill", {"name": "review"}, "skill"),
                call(
                    "edit_file",
                    {"path": "result.txt", "create": True, "new_text": "verified\n"},
                    "edit",
                ),
                call(
                    "shell",
                    {
                        "command": 'test "$(cat result.txt)" = verified && printf verified'
                    },
                    "verify",
                ),
                call(
                    "write_todos",
                    {"todos": [{**todos[0], "status": "completed"}]},
                    "complete",
                ),
                AIMessage("Created and verified result.txt"),
            ]
        )
        worker = Worker(r, model=model)
        await r.store.submit(
            "create and verify", profile="workspace_write", workspace_id="source"
        )
        row = await execute(r, worker)
        assert row["status"] == "completed", row
        assert (workspace / "result.txt").read_text() == "verified\n"
        state = await worker.agent.aget_state(row["final_checkpoint"])
        assert state.values["thread_tool_call_count"]["__all__"] == 5
        assert state.values["thread_model_call_count"] == 6
        verification = next(
            m
            for m in state.values["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id == "verify"
        )
        assert (
            verification.status == "success"
            and verification.artifact["exit_code"] == 0
            and "verified" in verification.text
        )
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"
        model = SyntheticModel(
            responses=[
                call("read_skill", {"name": "review"}, "skill"),
                call(
                    "edit_file",
                    {"path": "forbidden", "create": True, "new_text": "bad"},
                    "edit",
                ),
                call("shell", {"command": "touch forbidden-shell"}, "shell"),
                AIMessage("Read-only tools enforced"),
            ]
        )
        worker = Worker(r, model=model)
        await r.store.submit(
            "follow skill", profile="workspace_read", workspace_id="source"
        )
        row = await execute(r, worker)
        assert row["status"] == "completed"
        state = await worker.agent.aget_state(row["final_checkpoint"])
        errors = [
            m
            for m in state.values["messages"]
            if isinstance(m, ToolMessage) and m.status == "error"
        ]
        assert (
            len(errors) == 2
            and not (workspace / "forbidden").exists()
            and not (workspace / "forbidden-shell").exists()
        )
        assert all(
            constraints(request)["workspace_access"] == "read"
            and constraints(request)["workspace_network"] == "none"
            for request in model.history
        )
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"

    database(check)


@pytest.mark.parametrize(
    "behavior", ["retry-success", "retry-exhausted", "timeout", "cancel"]
)
def test_native_summary_retry_timeout_cancel_preserves_cleanup_without_replay(
    database, tmp_path, behavior
):
    from tests.test_execution import install as install_workspace

    async def check(r):
        install_workspace(r, tmp_path)
        summary_started = asyncio.Event()
        attempts = main = 0

        async def reply(messages):
            nonlocal attempts, main
            if "Context Extraction Assistant" in messages[0].text:
                attempts += 1
                summary_started.set()
                if behavior in ("timeout", "cancel"):
                    await asyncio.sleep(30)
                if behavior == "retry-exhausted" or (
                    behavior == "retry-success" and attempts < 3
                ):
                    raise RuntimeError("Synthetic summary failure")
                return AIMessage("File changed once and shell verification completed.")
            main += 1
            if main == 1:
                return call(
                    "shell",
                    {
                        "command": "printf once >> changes.txt; printf 'payload%.0s' {1..1000}"
                    },
                    "once",
                )
            return AIMessage("Verified")

        model = SyntheticModel(reply=reply)
        worker = Worker(r, model=model)
        row, _ = await r.store.submit(
            "create once and report", profile="workspace_write", workspace_id="source"
        )
        claim = await r.store.claim(worker.id)
        task = asyncio.create_task(worker.execute(claim))
        try:
            await asyncio.wait_for(summary_started.wait(), 5)
            if behavior == "cancel":
                await r.store.cancel(row["id"])
            await asyncio.wait_for(task, 20)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        result = await r.store.get(row["id"])
        assert (tmp_path / "changes.txt").read_text() == "once"
        assert (await r.store.execution_resource(row["id"]))["state"] == "released"
        if behavior == "retry-success":
            assert result["status"] == "completed" and attempts == 3 and main == 2
        elif behavior == "retry-exhausted":
            assert (
                result["error_code"] == "EXECUTION_FAILED"
                and attempts == 3
                and main == 1
            )
        elif behavior == "timeout":
            assert result["error_code"] == "RUN_TIMEOUT" and attempts == 1 and main == 1
        else:
            assert result["status"] == "cancelled" and attempts == 1 and main == 1

    database(
        check,
        replace(
            Settings(),
            summary_trigger_tokens=150,
            summary_keep_tokens=40,
            summary_trim_tokens=1000,
            run_timeout=14.5 if behavior == "timeout" else 30,
            heartbeat_seconds=0.03,
            lease_seconds=4,
        ),
    )


def test_history_rejects_foreign_or_ambiguous_final_checkpoint(database, monkeypatch):
    async def check(r):
        conversation = await r.store.create_conversation("source")
        await r.store.submit("successful history", conversation_id=conversation["id"])
        first = await execute(r)
        get = r.store.get
        for mode in ("foreign-conversation", "missing-checkpoint", "wrong-thread"):
            monkeypatch.setattr(r.store, "get", get)
            target = (
                await r.store.create_conversation("foreign")
                if mode == "foreign-conversation"
                else conversation
            )
            await r.store.submit("next", conversation_id=target["id"])
            model = SyntheticModel()
            worker = Worker(r, model=model)
            claim = await r.store.claim(worker.id)
            claim["previous_run_id"] = first["id"]
            if mode != "foreign-conversation":

                async def invalid_final(run_id):
                    row = await get(run_id)
                    if run_id == first["id"]:
                        pointer = dict(row["final_checkpoint"]["configurable"])
                        if mode == "missing-checkpoint":
                            pointer.pop("checkpoint_id")
                        else:
                            pointer["thread_id"] = str(claim["id"])
                        row = {**row, "final_checkpoint": {"configurable": pointer}}
                    return row

                monkeypatch.setattr(r.store, "get", invalid_final)
            await worker.execute(claim)
            row = await get(claim["id"])
            assert row["error_code"] == "EXECUTION_FAILED" and model.history == []

    database(check)
