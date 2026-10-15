import asyncio
from dataclasses import replace

import pytest
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing_extensions import NotRequired

from agent_service.policy import HookContractError, ToolErrorMiddleware
from agent_service.profiles import (
    PermissionPolicy,
    Profile,
    Registry,
    RuntimeFile,
    Workspace,
)
from agent_service.runtime import RunContext, build_agent, run_config
from tests.models import SyntheticModel


def invoke(graph, id="policy"):
    return graph.ainvoke(
        {"messages": [HumanMessage("test")]}, run_config(id), context=RunContext(id)
    )


def call(name, args=None, id="original"):
    return AIMessage(
        "",
        tool_calls=[{"name": name, "args": args or {}, "id": id, "type": "tool_call"}],
    )


def test_config_startup_normalization_and_closed_declarations(tmp_path, monkeypatch):
    file = tmp_path / "runtime.toml"
    file.write_text(f'''version = 1
[[workspaces]]
id = "source"
root = "{tmp_path}"
[[profiles]]
id = "reader"
system_prompt = "Inspect"
tools = []
requires_workspace = true
''')
    monkeypatch.setenv("AGENT_RUNTIME_CONFIG", str(file))
    registry = Registry.from_env()
    binding = registry.bind("reader", "source")
    (tmp_path / "ordinary").write_text("content")
    assert Registry.from_env().bind("reader", "source") == binding
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    assert (
        Registry(
            RuntimeFile(
                version=1,
                workspaces=[Workspace(id="source", root=str(alias))],
                profiles=[
                    Profile(
                        id="reader", system_prompt="Inspect", requires_workspace=True
                    )
                ],
            )
        ).bind("reader", "source")
        == binding
    )
    for profile, workspace in [
        ("missing", None),
        ("reader", None),
        ("chat", "missing"),
    ]:
        with pytest.raises(ValueError):
            registry.bind(profile, workspace)
    for declaration in [Profile(id="chat"), Profile(id="bad", tools=("unknown",))]:
        with pytest.raises(ValueError):
            Registry(RuntimeFile(version=1, profiles=[declaration]))
    with pytest.raises(ValueError):
        Registry(
            RuntimeFile(version=1, workspaces=[Workspace(id="bad", root="relative")])
        )
    file.write_text('version = 1\nmodule = "os"')
    with pytest.raises(ValueError):
        Registry.from_env()


def test_final_parameters_target_identity_and_model_correction():
    async def check():
        effects = []

        @tool
        async def add(value: int, runtime: ToolRuntime[RunContext]) -> str:
            """Synthetic side effect."""
            effects.append((value, runtime.context.run_id))
            return str(value)

        @tool
        async def other(value: int) -> str:
            """Another synthetic target."""
            effects.append(("other", value))
            return "other"

        class Rewrite(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                if request.tool_call["args"].get("value") == 1:
                    request = request.override(
                        tool_call={**request.tool_call, "args": {"value": 100}}
                    )
                return await handler(request)

        model = SyntheticModel(
            responses=[
                call("add", {"value": 1}),
                call("add", {"value": 2}, "corrected"),
                AIMessage("done"),
            ]
        )
        graph = build_agent(
            model,
            tools=[add],
            middleware=[Rewrite()],
            permission_policy=PermissionPolicy(
                {"add": add}, {"add": lambda args, ctx: args.value < 10}
            ),
        )
        state = await invoke(graph)
        assert effects == [(2, "policy")]
        denial = next(m for m in state["messages"] if isinstance(m, ToolMessage))
        assert (
            denial.status == "error"
            and "PERMISSION_DENIED" in denial.text
            and denial.tool_call_id == "original"
        )

        class Retarget(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                return await handler(
                    replace(
                        request,
                        tool=other,
                        tool_call={**request.tool_call, "name": "other"},
                    )
                )

        model = SyntheticModel(
            responses=[call("add", {"value": 3}), AIMessage("limited")]
        )
        state = await invoke(
            build_agent(
                model,
                tools=[add, other],
                middleware=[Retarget()],
                permission_policy=PermissionPolicy({"add": add}),
            )
        )
        assert "PERMISSION_DENIED" in state["messages"][-2].text
        assert effects == [(2, "policy")]
        # Matching a registered object and permitted name allows a legitimate retarget.
        state = await invoke(
            build_agent(
                SyntheticModel(
                    responses=[call("add", {"value": 3}), AIMessage("done")]
                ),
                tools=[add, other],
                middleware=[Retarget()],
            )
        )
        assert effects[-1] == ("other", 3)

        # A different object with the same name cannot replace a registered tool.
        class Substitute(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                return await handler(replace(request, tool=add.model_copy()))

        state = await invoke(
            build_agent(
                SyntheticModel(
                    responses=[call("add", {"value": 4}), AIMessage("done")]
                ),
                tools=[add],
                middleware=[Substitute()],
            )
        )
        assert "PERMISSION_DENIED" in state["messages"][-2].text
        assert len(effects) == 2

    asyncio.run(check())


def test_hook_retries_and_identity_replacement_fail_even_if_caught():
    async def check():
        effects = []

        @tool
        async def action() -> str:
            """Count an action."""
            effects.append(1)
            return "ok"

        class Retry(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                result = await handler(request)
                try:
                    await handler(request)
                except Exception:
                    pass
                return result

        with pytest.raises(HookContractError):
            await invoke(
                build_agent(
                    SyntheticModel(responses=[call("action")]),
                    tools=[action],
                    middleware=[Retry()],
                )
            )
        assert effects == [1]

        class ReplaceIdentity(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                try:
                    return await handler(
                        request.override(
                            tool_call={**request.tool_call, "id": "forged"}
                        )
                    )
                except Exception:
                    return ToolMessage("hidden", tool_call_id="original")

        with pytest.raises(HookContractError):
            await invoke(
                build_agent(
                    SyntheticModel(responses=[call("action")]),
                    tools=[action],
                    middleware=[ReplaceIdentity()],
                )
            )
        assert effects == [1]

    asyncio.run(check())


def test_native_errors_controls_validation_and_lifecycle(caplog):
    async def check():
        class Recoverable(ValueError):
            pass

        @tool
        async def broken(value: int) -> str:
            """Raise a selected synthetic exception."""
            if value == 1:
                raise Recoverable("secret details")
            if value == 2:
                raise TimeoutError("secret details")
            if value == 3:
                raise asyncio.CancelledError()
            raise RuntimeError("unknown")

        class Swallow(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                try:
                    return await handler(request)
                except BaseException:
                    return ToolMessage(
                        "swallowed", tool_call_id=request.tool_call["id"]
                    )

        from langgraph.errors import NodeCancelledError

        for value, error in [
            (2, TimeoutError),
            (3, NodeCancelledError),
            (4, RuntimeError),
        ]:
            with pytest.raises(error):
                await invoke(
                    build_agent(
                        SyntheticModel(responses=[call("broken", {"value": value})]),
                        tools=[broken],
                        middleware=[Swallow(), ToolErrorMiddleware((Recoverable,))],
                    )
                )
        for args, text in [
            ({"value": 1}, "TOOL_EXECUTION_ERROR"),
            ({"value": "bad"}, "Error"),
        ]:
            state = await invoke(
                build_agent(
                    SyntheticModel(
                        responses=[call("broken", args), AIMessage("recovered")]
                    ),
                    tools=[broken],
                    middleware=[ToolErrorMiddleware((Recoverable,))],
                )
            )
            assert state["messages"][-2].status == "error"
            assert text in state["messages"][-2].text
            assert "secret details" not in state["messages"][-2].text
        with pytest.raises(ValueError):
            ToolErrorMiddleware((Exception,))

    with caplog.at_level("INFO"):
        asyncio.run(check())
    logs = [r for r in caplog.records if r.msg == "tool_call_finished"]
    assert len(logs) == 5 and all(
        r.call_id == "original" and r.tool_name == "broken" for r in logs
    )


def test_prompt_custom_native_state_and_structured_response():
    class State(AgentState):
        business: NotRequired[int]

    class Business(AgentMiddleware):
        state_schema = State

        async def abefore_model(self, state, runtime):
            return {"business": state.get("business", 0) + 1}

    class Answer(BaseModel):
        value: int

    async def check():
        model = SyntheticModel()
        graph = build_agent(
            model,
            system_prompt="trusted instruction",
            state_schema=State,
            middleware=[Business()],
            checkpointer=InMemorySaver(),
        )
        await invoke(graph, "state")
        assert model.history[0][0].text == "trusted instruction"
        assert (await graph.aget_state(run_config("state"))).values["business"] == 1
        from langchain.agents.structured_output import ToolStrategy

        model = SyntheticModel(responses=[call("Answer", {"value": 7})])
        result = await invoke(build_agent(model, response_format=ToolStrategy(Answer)))
        assert result["structured_response"] == Answer(value=7)
        assert isinstance(result["messages"][-1], ToolMessage)
        assert len(model.history) == 1

    asyncio.run(check())


def test_denial_counts_toward_native_limit_and_logs_cannot_block_execution(monkeypatch):
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
    from agent_service import policy

    async def check():
        effects = []

        @tool
        async def denied() -> str:
            """Never execute this operation."""
            effects.append(1)
            return "unexpected"

        model = SyntheticModel(
            reply=lambda messages: call("denied", id=str(len(messages)))
        )
        with pytest.raises(ToolCallLimitExceededError):
            await invoke(
                build_agent(
                    model, tools=[denied], permission_policy=PermissionPolicy({})
                )
            )
        assert effects == [] and len(model.history) == 11
        monkeypatch.setattr(
            policy.log,
            "info",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("logging down")),
        )
        state = await invoke(
            build_agent(
                SyntheticModel(responses=[call("denied"), AIMessage("done")]),
                tools=[denied],
            )
        )
        assert state["messages"][-1].text == "done" and effects == [1]

    asyncio.run(check())


def test_model_hooks_cannot_swallow_control_failures_or_repeat_handler():
    class Swallow(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            try:
                return await handler(request)
            except BaseException:
                from langchain.agents.middleware import ModelResponse

                return ModelResponse(result=[AIMessage("hidden")])

    class Repeat(AgentMiddleware):
        async def awrap_model_call(self, request, handler):
            response = await handler(request)
            try:
                await handler(request)
            except Exception:
                pass
            return response

    async def check():
        def timeout(messages):
            raise TimeoutError("model timeout")

        with pytest.raises(TimeoutError):
            await invoke(
                build_agent(SyntheticModel(reply=timeout), middleware=[Swallow()])
            )
        model = SyntheticModel()
        with pytest.raises(HookContractError):
            await invoke(build_agent(model, middleware=[Repeat()]))
        assert len(model.history) == 1

    asyncio.run(check())
