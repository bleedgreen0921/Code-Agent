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
from agent_service.profiles import PermissionPolicy
from agent_service.policy import (
    PermissionMiddleware,
    ToolLifecycleMiddleware,
    ExecutionControlMiddleware,
)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    conversation_id: str | None = None
    profile: str = "chat"
    workspace_id: str | None = None
    workspace_root: str | None = None
    workspace_identity: str | None = None
    binding_fingerprint: str | None = None
    deadline_at: datetime | None = None


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
):
    cfg = settings or Settings.from_env()
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
            ToolLifecycleMiddleware(output_limits),
            ModelCallLimitMiddleware(run_limit=16, exit_behavior="error"),
            ToolCallLimitMiddleware(run_limit=10, exit_behavior="error"),
            SummarizationMiddleware(
                model,
                trigger=("tokens", cfg.summary_trigger_tokens),
                keep=("tokens", cfg.summary_keep_tokens),
                trim_tokens_to_summarize=cfg.summary_trim_tokens,
            ),
            *(m for m in middleware if not isinstance(m, ToolErrorMiddleware)),
            PermissionMiddleware(
                permission_policy or PermissionPolicy({t.name: t for t in tools})
            ),
            *(m for m in middleware if isinstance(m, ToolErrorMiddleware)),
            *execution_middleware,
            ExecutionControlMiddleware(),
        ],
    )


def run_config(run_id):
    return {
        "configurable": {"thread_id": str(run_id)},
        "max_concurrency": 2,
        "recursion_limit": 128,
    }
