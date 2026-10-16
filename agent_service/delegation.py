"""Run-owned, one-level read-only delegation through native tool/subgraph calls."""

import asyncio
import json
import logging
from dataclasses import replace
from uuid import uuid4
from agent_service.no_progress import NoProgress
from agent_service.failure import failure_info
from typing import Annotated, Any

import openai
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.tools import ToolRuntime
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, InjectedToolArg
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, field_validator

VERSION = "delegate-read-v1"
PROMPT_VERSION = "readonly-subagent-v1"
READ_TOOLS = frozenset({"glob_search", "grep_search", "read_file"})
CHILD_TOOLS = READ_TOOLS | {"read_skill"}
LIMITS = {
    "starts": 2,
    "depth": 1,
    "model_calls": 8,
    "tool_calls": 6,
    "task_bytes": 8192,
}
CHILD_PROMPT = (
    "You are a read-only investigation subagent. Complete only the delegated task. "
    "Use the available read/search tools to establish facts. Return concise findings, "
    "necessary file locations, and unresolved questions. Distinguish assumptions from "
    "verified results; never claim an operation was executed when it was not. "
    "Do not modify files, execute commands, or delegate further."
)
log = logging.getLogger(__name__)


def manifest():
    return {
        "version": VERSION,
        "prompt_version": PROMPT_VERSION,
        "tools": sorted(CHILD_TOOLS),
        "limits": dict(LIMITS),
    }


class DelegateInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str

    @field_validator("task")
    @classmethod
    def bounded_task(cls, value):
        if not value.strip() or len(value.encode("utf-8")) > LIMITS["task_bytes"]:
            raise ValueError("task must be nonempty and at most 8192 UTF-8 bytes")
        return value


class RuntimeDelegateInput(DelegateInput):
    # Explicit schemas must accept native injected args while hiding them from the model.
    runtime: Annotated[Any, InjectedToolArg] = None


def error_message(code, call_id):
    return ToolMessage(
        content=json.dumps(
            {
                "code": code,
                "message": {
                    "SUBAGENT_BATCH_REJECTED": "Delegate alone in a single tool call.",
                    "SUBAGENT_LIMIT_REACHED": "The Run delegation limit has been reached.",
                    "SUBAGENT_DUPLICATE_CALL": "This tool call ID has already started a delegation.",
                    "SUBAGENT_DEPTH_EXCEEDED": "Nested delegation is not permitted.",
                    "SUBAGENT_QUOTA_EXCEEDED": "The subagent exhausted its call quota.",
                    "SUBAGENT_MODEL_FAILED": "The subagent model request failed.",
                    "SUBAGENT_INVALID_OUTPUT": "The subagent did not produce valid final text.",
                    "SUBAGENT_NO_PROGRESS": "The subagent repeated the same correctable tool failure.",
                }[code],
            }
        ),
        name="delegate_read",
        tool_call_id=call_id,
        status="error",
        artifact={"failure_info": failure_info(code)}
        if code == "SUBAGENT_NO_PROGRESS"
        else None,
    )


def final_text(state):
    """Shared final assistant text contract for root and delegated graphs."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or last.tool_calls or not last.text.strip():
        raise InvalidFinalText("Graph did not produce a final assistant text")
    return last.text


class InvalidFinalText(RuntimeError):
    pass


class ReportedUsage(UsageMetadataCallbackHandler):
    """Native aggregation with a bounded completeness indicator, no call ledger."""

    def __init__(self):
        super().__init__()
        self.missing = 0
        self.model_failure = None

    def on_llm_end(self, response, **kwargs):
        super().on_llm_end(response, **kwargs)
        try:
            message = response.generations[0][0].message
            complete = bool(
                message.usage_metadata and message.response_metadata.get("model_name")
            )
        except (AttributeError, IndexError):
            complete = False
        if not complete:
            with self._lock:
                self.missing += 1

    def on_llm_error(self, error, **kwargs):
        with self._lock:
            self.missing += 1
            self.model_failure = error

    def report(self, *, aborted=False):
        with self._lock:
            # Model selection is server-owned; bound diagnostic payload regardless.
            items = list(self.usage_metadata.items())[:16]
            return {
                "models": {
                    str(name)[:200]: {
                        key: (
                            {str(k)[:64]: v for k, v in list(value.items())[:16]}
                            if isinstance(value, dict)
                            else value
                        )
                        for key, value in usage.items()
                        if key
                        in {
                            "input_tokens",
                            "output_tokens",
                            "total_tokens",
                            "input_token_details",
                            "output_token_details",
                        }
                    }
                    for name, usage in items
                },
                "complete": not (
                    aborted or self.missing or len(self.usage_metadata) > 16
                ),
                "missing_reports": self.missing,
            }


class DelegationState(AgentState):
    delegations: list[dict]
    reported_usage: dict


class DelegationDiagnostics(AgentMiddleware):
    def __init__(self, controller, usage, state_schema=None):
        self.controller, self.usage = controller, usage
        if state_schema is not None:
            # Keep trusted extension channels and native reducers intact.
            self.state_schema = type(
                "RunDelegationState", (state_schema, DelegationState), {}
            )
        else:
            self.state_schema = DelegationState

    async def aafter_agent(self, state, runtime):
        return {
            "delegations": list(self.controller.records),
            "reported_usage": self.usage.report(),
        }


class DelegationBatchMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        from agent_service.policy import active_call

        call = active_call.get()
        if request.tool_call["name"] == "delegate_read" and len(call.batch) != 1:
            return error_message("SUBAGENT_BATCH_REJECTED", call.call_id)
        return await handler(request)


def delegate_tool(controller=None):
    async def delegate_read(task: str, runtime: ToolRuntime):
        if controller is None:
            raise RuntimeError("Delegation declaration cannot execute")
        return await controller.invoke(task, runtime)

    return StructuredTool.from_function(
        coroutine=delegate_read,
        name="delegate_read",
        args_schema=RuntimeDelegateInput,
        description=(
            "Delegate one independent, focused read-only investigation. Supply the goal, "
            "necessary background and expected findings in task (at most 8 KiB UTF-8). "
            "Only the final text returns. Use directly available tools for simple operations. "
            "Call this tool alone; at most two delegations per Run, no nested delegation."
        ),
    )


class DelegationController:
    """Authoritative start reservations live outside model/checkpoint/Hook state."""

    def __init__(
        self,
        *,
        model,
        checkpointer,
        settings,
        scope,
        policy,
        skills=(),
        system_prompt=None,
        no_progress_enabled=False,
    ):
        self.model, self.checkpointer, self.settings = model, checkpointer, settings
        self.scope, self.policy, self.skills = scope, policy, tuple(skills)
        self.system_prompt = system_prompt
        self.no_progress_enabled = no_progress_enabled
        self.started = set()
        self.lock = asyncio.Lock()
        self.records = []

    def child_graph(self):
        from agent_service.workspace_tools import assemble
        from agent_service.skills import read_skill_tool
        from agent_service.profiles import PermissionPolicy
        from agent_service.runtime import build_agent
        from agent_service.execution_adapter import WorkspaceExecutionMiddleware

        names = set(self.policy.tools) & CHILD_TOOLS
        tools, _ = assemble(self.scope, names & READ_TOOLS)
        if "read_skill" in names:
            tools.append(read_skill_tool({s.name: s for s in self.skills}))
        policy = PermissionPolicy(
            {t.name: t for t in tools},
            {n: v for n, v in self.policy.validators.items() if n in names},
            self.policy.binding_fingerprint,
        )
        return build_agent(
            self.model,
            checkpointer=self.checkpointer,
            settings=self.settings,
            tools=tools,
            permission_policy=policy,
            system_prompt="\n\n".join(
                p for p in (self.system_prompt, CHILD_PROMPT) if p
            ),
            skills=self.skills,
            model_call_limit=LIMITS["model_calls"],
            tool_call_limit=LIMITS["tool_calls"],
            no_progress_enabled=self.no_progress_enabled,
            execution_middleware=[
                WorkspaceExecutionMiddleware(self.scope, names & READ_TOOLS)
            ],
            output_limits=(
                self.scope.settings.output_bytes,
                self.scope.settings.output_lines,
            ),
        )

    async def invoke(self, task, runtime):
        from agent_service.policy import active_call, HookContractError

        call = active_call.get()
        if (
            call is None
            or not call.entered
            or call.context is not runtime.context
            or runtime.tool_call_id != call.call_id
        ):
            raise HookContractError("Delegation requires the trusted tool boundary")
        # Validate again at the actual entry, after all trusted Hooks have run.
        DelegateInput(task=task)
        if len(call.batch) != 1:
            return error_message("SUBAGENT_BATCH_REJECTED", call.call_id)
        if runtime.context.agent_depth != 0:
            return error_message("SUBAGENT_DEPTH_EXCEEDED", call.call_id)
        async with self.lock:
            if call.call_id in self.started:
                return error_message("SUBAGENT_DUPLICATE_CALL", call.call_id)
            if len(self.started) >= LIMITS["starts"]:
                return error_message("SUBAGENT_LIMIT_REACHED", call.call_id)
            self.scope.check_open()
            self.started.add(call.call_id)
            return await self.run(task, runtime, call.call_id)

    async def run(self, task, runtime, call_id):
        identity = str(uuid4())
        context = replace(
            runtime.context,
            agent_id=identity,
            parent_agent_id=runtime.context.agent_id or runtime.context.run_id,
            agent_depth=1,
        )
        graph = self.child_graph()
        usage = ReportedUsage()
        from langchain_core.runnables.config import merge_configs

        config = merge_configs(
            runtime.config,
            {
                "callbacks": [usage],
                "metadata": {
                    "parent_run_id": context.run_id,
                    "delegation_id": identity,
                    "original_tool_call_id": call_id,
                    "agent_depth": 1,
                    "delegation_version": VERSION,
                },
            },
        )
        # Never pass the parent's checkpoint_id into a fresh subgraph invocation.
        config = {
            **config,
            "configurable": {
                k: v for k, v in config["configurable"].items() if k != "checkpoint_id"
            },
        }
        state, checkpoint, code = {}, None, None
        record = {"delegation_id": identity, "original_tool_call_id": call_id}
        try:
            async for mode, data in graph.astream(
                {"messages": [HumanMessage(task)]},
                config,
                context=context,
                stream_mode=["values", "checkpoints"],
                durability="sync",
            ):
                if mode == "values":
                    state = data
                elif mode == "checkpoints":
                    checkpoint = data
            if (
                not checkpoint
                or checkpoint["next"]
                or state.get("__interrupt__")
                or any(t.get("interrupts") for t in checkpoint.get("tasks", []))
            ):
                raise InvalidFinalText("Incomplete subgraph")
            answer = final_text(state)
            pointer = checkpoint["config"]
            if not pointer["configurable"].get("checkpoint_ns"):
                raise RuntimeError(
                    "Subgraph checkpoint requires an independent namespace"
                )
            saved = await self.checkpointer.aget_tuple(pointer)
            if (
                saved is None
                or saved.config["configurable"]["checkpoint_id"]
                != pointer["configurable"]["checkpoint_id"]
            ):
                raise RuntimeError("Subgraph final persistence unconfirmed")
        except (ModelCallLimitExceededError, ToolCallLimitExceededError):
            code = "SUBAGENT_QUOTA_EXCEEDED"
        except openai.APIError as exc:
            if usage.model_failure is not exc:
                raise
            code = "SUBAGENT_MODEL_FAILED"
        except InvalidFinalText:
            code = "SUBAGENT_INVALID_OUTPUT"
        except NoProgress:
            code = "SUBAGENT_NO_PROGRESS"
        except BaseException:
            log.info(
                "delegation_aborted",
                extra={
                    "run_id": context.run_id,
                    "delegation": {
                        **record,
                        "status": "aborted",
                        "last_checkpoint": checkpoint["config"] if checkpoint else None,
                        "model_calls": state.get("thread_model_call_count", 0),
                        "tool_calls": state.get("thread_tool_call_count", {}),
                        "reported_usage": usage.report(aborted=True),
                    },
                },
            )
            raise
        record.update(
            {
                "status": "failed" if code else "completed",
                "error_code": code,
                "namespace": checkpoint["config"]["configurable"].get("checkpoint_ns")
                if checkpoint
                else None,
                "final_checkpoint": checkpoint["config"]
                if checkpoint and not code
                else None,
                **(
                    {
                        "last_checkpoint": checkpoint["config"] if checkpoint else None,
                        "failure_info": failure_info("SUBAGENT_NO_PROGRESS"),
                    }
                    if code == "SUBAGENT_NO_PROGRESS"
                    else {}
                ),
                "model_calls": state.get("thread_model_call_count", 0),
                "tool_calls": state.get("thread_tool_call_count", {}),
                "reported_usage": usage.report(),
            }
        )
        self.records.append(record)
        message = (
            error_message(code, call_id)
            if code
            else ToolMessage(content=answer, name="delegate_read", tool_call_id=call_id)
        )
        return Command(
            update={"messages": [message], "delegations": list(self.records)}
        )
