"""Standard LangChain Agent factory, also usable directly for native interrupt/resume."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from agent_service.config import Settings
from agent_service.context import RunContextMiddleware, SUMMARY_PROMPT, validate_todos
from agent_service.profiles import PermissionPolicy
from agent_service.policy import (
    PermissionMiddleware,
    ToolLifecycleMiddleware,
    ExecutionControlMiddleware,
)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    task_id: str | None = None
    task_store: object = None
    result_delivery: object = None
    delivered_results: object = None
    conversation_id: str | None = None
    profile: str = "chat"
    workspace_id: str | None = None
    workspace_root: str | None = None
    workspace_identity: str | None = None
    binding_fingerprint: str | None = None
    deadline_at: datetime | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    agent_depth: int = 0


def model_from_resources(resources):
    url = os.environ["AGENT_MODEL_URL"].rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return ChatOpenAI(
        model=os.environ["AGENT_MODEL"],
        base_url=url,
        api_key=os.environ.get("AGENT_MODEL_KEY") or "local",
        timeout=resources.settings.model_timeout,
        max_tokens=resources.settings.max_output_tokens,
        max_retries=0,
        http_async_client=resources.http,
        http_client=resources.sync_http,
    )


def build_agent(
    model,
    *,
    checkpointer=None,
    tools: Sequence[BaseTool] = (),
    store: BaseStore | None = None,
    settings: Settings | None = None,
    system_prompt=None,
    middleware=(),
    state_schema=None,
    response_format=None,
    permission_policy: PermissionPolicy | None = None,
    execution_middleware=(),
    output_limits=None,
    skills=(),
    model_call_limit=16,
    tool_call_limit=10,
):
    from agent_service.delegation import DelegationBatchMiddleware
    from agent_service.mcp_runtime import MCPBatchMiddleware
    from agent_service.background import BackgroundBatchMiddleware

    cfg = settings or Settings.from_env()
    supplied = [
        *tools,
        *(
            t
            for m in (*middleware, *execution_middleware)
            for t in getattr(m, "tools", ())
        ),
    ]
    names = [t.name for t in supplied]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate runtime tool names")
    policy = permission_policy or PermissionPolicy({t.name: t for t in supplied})
    validators = dict(policy.validators)
    if "write_todos" in policy.tools:
        custom = validators.get("write_todos")
        validators["write_todos"] = lambda args, ctx: validate_todos(args, ctx) and (
            custom is None or custom(args, ctx) is not False
        )
    policy = PermissionPolicy(policy.tools, validators, policy.binding_fingerprint)
    return create_agent(
        model=model,
        tools=list(tools),
        context_schema=RunContext,
        checkpointer=checkpointer,
        store=store,
        system_prompt=system_prompt,
        state_schema=state_schema,
        response_format=response_format,
        middleware=[
            ToolLifecycleMiddleware(output_limits or (16384, 200)),
            ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="error"),
            SummarizationMiddleware(
                model,
                trigger=("tokens", cfg.summary_trigger_tokens),
                keep=("tokens", cfg.summary_keep_tokens),
                trim_tokens_to_summarize=cfg.summary_trim_tokens,
                summary_prompt=SUMMARY_PROMPT,
            ),
            DelegationBatchMiddleware(),
            MCPBatchMiddleware(policy),
            BackgroundBatchMiddleware(policy),
            *(m for m in middleware if not isinstance(m, ToolErrorMiddleware)),
            PermissionMiddleware(policy),
            *(m for m in middleware if isinstance(m, ToolErrorMiddleware)),
            *execution_middleware,
            RunContextMiddleware(policy, system_prompt=system_prompt, skills=skills),
            ExecutionControlMiddleware(),
        ],
    )


def run_config(run_id):
    return {
        "configurable": {"thread_id": str(run_id)},
        "max_concurrency": 2,
        "recursion_limit": 128,
    }
