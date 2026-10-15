import asyncio
from dataclasses import replace

import pytest
from agent_service.config import Settings
from agent_service.runtime import RunContext, build_agent, run_config
from agent_service.worker import failure_code
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt

from tests.models import SyntheticModel


def test_empty_tools_and_standard_runtime_context():
    async def check():
        graph = build_agent(SyntheticModel())
        result = await graph.ainvoke(
            {"messages": [HumanMessage("hello")]},
            run_config("empty"),
            context=RunContext("empty"),
        )
        assert result["messages"][-1].text == "synthetic answer"
        seen = []

        @tool
        async def double(value: int, runtime: ToolRuntime[RunContext]) -> int:
            """Double an integer."""
            seen.append((value, runtime.context.run_id, runtime.store))
            return value * 2

        model = SyntheticModel(
            responses=[
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "double",
                            "args": {"value": 3},
                            "id": "t",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage("6"),
            ]
        )
        store = InMemoryStore()
        graph = build_agent(model, tools=[double], store=store)
        result = await graph.ainvoke(
            {"messages": [HumanMessage("double 3")]},
            run_config("tool"),
            context=RunContext("tool"),
        )
        assert seen == [(3, "tool", store)]
        assert any(
            isinstance(m, ToolMessage) and m.content == "6" for m in result["messages"]
        )
        # Parameter validation is supplied by the standard tool schema.
        with pytest.raises(ValueError):
            await double.ainvoke({"value": "invalid"})

    asyncio.run(check())


def test_tool_limit_and_max_concurrency():
    async def check():
        active = peak = calls = 0

        @tool
        async def wait(value: int) -> str:
            """Synthetic asynchronous tool."""
            nonlocal active, peak, calls
            calls += 1
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return str(value)

        def reply(messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage("done")
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "wait",
                        "args": {"value": i},
                        "id": str(i),
                        "type": "tool_call",
                    }
                    for i in range(6)
                ],
            )

        graph = build_agent(SyntheticModel(reply=reply), tools=[wait])
        await graph.ainvoke(
            {"messages": [HumanMessage("batch")]},
            run_config("batch"),
            context=RunContext("batch"),
        )
        assert calls == 6 and peak == 2
        calls = 0

        def loop(messages):
            return AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "wait",
                        "args": {"value": 1},
                        "id": str(len(messages)),
                        "type": "tool_call",
                    }
                ],
            )

        graph = build_agent(SyntheticModel(reply=loop), tools=[wait])
        with pytest.raises(ToolCallLimitExceededError) as caught:
            await graph.ainvoke(
                {"messages": [HumanMessage("loop")]},
                run_config("loop"),
                context=RunContext("loop"),
            )
        assert calls == 10
        assert failure_code(caught.value) == "QUOTA_EXCEEDED"
        # Exercise the native model limiter independently of the smaller tool budget.
        graph = create_agent(
            SyntheticModel(reply=loop),
            tools=[wait],
            middleware=[ModelCallLimitMiddleware(run_limit=16, exit_behavior="error")],
        )
        with pytest.raises(ModelCallLimitExceededError) as caught:
            await graph.ainvoke(
                {"messages": [HumanMessage("loop")]},
                {"recursion_limit": 128, "max_concurrency": 2},
            )
        assert failure_code(caught.value) == "QUOTA_EXCEEDED"

    asyncio.run(check())


def test_summary_preserves_recent_messages_and_native_interrupt_resume():
    async def check():
        model = SyntheticModel(
            responses=[AIMessage("summary of old messages"), AIMessage("final")]
        )
        cfg = replace(
            Settings(),
            summary_trigger_tokens=150,
            summary_keep_tokens=40,
            summary_trim_tokens=1000,
        )
        graph = build_agent(model, checkpointer=InMemorySaver(), settings=cfg)
        old = HumanMessage("old context " * 150)
        recent = HumanMessage("recent question")
        result = await graph.ainvoke(
            {"messages": [old, AIMessage("old reply"), recent]},
            run_config("summary"),
            context=RunContext("summary"),
        )
        assert len(model.history) == 2
        assert any("summary of old messages" in m.text for m in result["messages"])
        assert any(m.id == recent.id for m in result["messages"])
        assert all(m.id != old.id for m in result["messages"])
        assert any("recent question" in m.text for m in model.history[-1])

        @tool
        def paused() -> str:
            """Synthetic native interrupt."""
            return interrupt("continue?")

        model = SyntheticModel(
            responses=[
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "paused",
                            "args": {},
                            "id": "pause",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage("resumed"),
            ]
        )
        graph = build_agent(model, checkpointer=InMemorySaver(), tools=[paused])
        config = run_config("pause")
        state = await graph.ainvoke(
            {"messages": [HumanMessage("pause")]}, config, context=RunContext("pause")
        )
        assert state["__interrupt__"]
        state = await graph.ainvoke(
            Command(resume="yes"), config, context=RunContext("pause")
        )
        assert state["messages"][-1].text == "resumed"

    asyncio.run(check())
