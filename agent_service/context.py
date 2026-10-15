"""Ephemeral main-model context, native Todo validation, and history boundaries."""

import json
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

CONTEXT_VERSION = "context-v1"
TODO_VERSION = "todos-v1"
SUMMARY_VERSION = "code-summary-v1"
TODO_LIMITS = {"items": 20, "content_characters": 200, "serialized_bytes": 8192}
SUMMARY_PROMPT = (
    DEFAULT_SUMMARY_PROMPT
    + "\nPreserve the user goal, constraints, changed files, executed commands and their results, and unresolved issues. Distinguish assumptions from verified facts."
)
log = logging.getLogger(__name__)


def validate_todos(args, context):
    todos = args.todos
    return (
        len(todos) <= TODO_LIMITS["items"]
        and all(len(t["content"]) <= TODO_LIMITS["content_characters"] for t in todos)
        and len(json.dumps(todos, ensure_ascii=False).encode())
        <= TODO_LIMITS["serialized_bytes"]
    )


def inherited_state(values):
    """The sole cross-Run state allowlist. Never copy counters or business state."""
    return {"messages": list(values.get("messages", []))}


class RunContextMiddleware(AgentMiddleware):
    def __init__(self, policy, *, system_prompt=None, skills=()):
        self.policy = policy
        self.system_prompt = system_prompt
        self.skills = tuple(skills)

    async def awrap_model_call(self, request, handler):
        from agent_service.execution_config import WORKSPACE_TOOLS, WRITE_TOOLS
        from agent_service.skills import catalog

        ctx = request.runtime.context
        names = sorted(self.policy.tools)
        access = (
            "write"
            if set(names) & WRITE_TOOLS
            else "read"
            if set(names) & WORKSPACE_TOOLS
            else "none"
        )
        progress = request.state.get("todos", []) if "write_todos" in names else []
        context = {
            "context_policy": CONTEXT_VERSION,
            "profile": getattr(ctx, "profile", "chat"),
            "workspace_access": access,
            "tools": names,
            "workspace_network": "none",
            "deadline_at": str(getattr(ctx, "deadline_at", None)),
            "file_changes": "Changes persist on failure, cancellation, or timeout; no rollback.",
            "skill_catalog": json.loads(catalog(self.skills)),
            "current_run_todos": progress,
            "todo_interpretation": "Model-maintained progress, not completion evidence. Empty means no Todo for this Run. Historical lists are not current state. Always deliver a final answer.",
        }
        blocks = (
            list(request.system_message.content_blocks)
            if request.system_message
            else []
        )
        if self.system_prompt and (
            not request.system_message
            or self.system_prompt not in request.system_message.text
        ):
            blocks.insert(0, {"type": "text", "text": self.system_prompt})
        blocks.append(
            {
                "type": "text",
                "text": "\nRun constraints and progress:\n"
                + json.dumps(context, ensure_ascii=False),
            }
        )
        system = SystemMessage(content=blocks)
        updated = request.override(system_message=system)
        try:
            log.info(
                "model_context",
                extra={
                    "run_id": getattr(ctx, "run_id", None),
                    "history_tokens_approx": count_tokens_approximately(
                        request.messages
                    ),
                    "request_tokens_approx": count_tokens_approximately(
                        [system, *request.messages], tools=list(request.tools)
                    ),
                },
            )
        except Exception:
            pass
        return await handler(updated)
