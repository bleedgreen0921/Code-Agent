"""An explicit allowlist for persisted configuration; never serialize resources."""

from importlib.metadata import version
from urllib.parse import urlsplit

from agent_service.profiles import fingerprint, tool_schema
from agent_service.execution_config import IMPLEMENTATION_VERSION, WORKSPACE_TOOLS
from agent_service.context import CONTEXT_VERSION, TODO_LIMITS, SUMMARY_VERSION
from agent_service.skills import SKILL_VERSION
from agent_service.knowledge import manifest as knowledge_manifest


def safe_endpoint(endpoint):
    if not endpoint:
        return None
    parsed = urlsplit(str(endpoint))
    host = parsed.hostname
    if not host:
        return None
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port else "")


def runtime_snapshot(model, settings, registry, binding, middleware):
    from agent_service.delegation import manifest

    profile = registry.resolve(binding)
    return {
        "version": 1,
        **(
            registry.extensions[profile.id].manifest()
            if profile.id in registry.extensions
            else {}
        ),
        "reliability": registry.reliability.model_dump(),
        **(
            {"knowledge": knowledge_manifest()}
            if set(profile.tools) & {"knowledge_get", "knowledge_search"}
            else {}
        ),
        **(
            {
                "mcp": [
                    {
                        **registry.mcp_tools[n].manifest(),
                        "contract": registry.mcp_tools[n].definition,
                    }
                    for n in sorted(profile.tools)
                    if n in registry.mcp_tools
                ]
            }
            if set(profile.tools) & registry.mcp_tools.keys()
            else {}
        ),
        **({"delegation": manifest()} if "delegate_read" in profile.tools else {}),
        "context_policy_version": CONTEXT_VERSION,
        "todo_limits": TODO_LIMITS if "write_todos" in profile.tools else None,
        "visible_skills": [
            registry.skills[n].manifest() for n in sorted(profile.skills)
        ],
        "skill_implementation": SKILL_VERSION if profile.skills else None,
        "summary_prompt_version": SUMMARY_VERSION,
        "binding_fingerprint": binding["fingerprint"],
        "profile": profile.id,
        "profile_fingerprint": binding["profile_fingerprint"],
        "prompt_fingerprint": fingerprint(profile.system_prompt),
        "workspace_id": binding["workspace_id"],
        "workspace_identity": binding["workspace_identity"],
        "model": {
            "name": getattr(model, "model_name", model._llm_type),
            "endpoint": safe_endpoint(getattr(model, "openai_api_base", None)),
            "temperature": getattr(model, "temperature", None),
            "max_output_tokens": getattr(model, "max_tokens", None)
            or settings.max_output_tokens,
            "timeout_seconds": settings.model_timeout,
            "max_retries": 0,
        },
        "tools": {
            name: fingerprint(tool_schema(registry.tools[name]))
            for name in sorted(profile.tools)
        },
        "middleware": {
            "model_call_limit": 16,
            "tool_call_limit": 10,
            "tool_output_bytes": registry.execution.output_bytes,
            "tool_output_lines": registry.execution.output_lines,
            "max_concurrency": 2,
            "summary_trigger_tokens": settings.summary_trigger_tokens,
            "summary_keep_tokens": settings.summary_keep_tokens,
            "summary_trim_tokens": settings.summary_trim_tokens,
            "extensions": [
                f"{type(m).__module__}.{type(m).__qualname__}" for m in middleware
            ],
            "extension_version": registry.extensions.get(profile.id).version
            if profile.id in registry.extensions
            else "1",
        },
        "execution": (
            {
                **registry.execution.model_dump(),
                "user": registry.execution.effective_user(),
                "tool_implementation": IMPLEMENTATION_VERSION,
                "network": "none",
                "read_only_rootfs": True,
            }
            if set(profile.tools)
            & (WORKSPACE_TOOLS | {"start_background_shell"})
            else None
        ),
        "run_timeout_seconds": settings.run_timeout,
        "task_limits": {
            name: getattr(settings, name)
            for name in (
                "task_timeout",
                "background_timeout",
                "max_unfinished_tasks",
                "max_active_operations",
                "max_task_active_operations",
                "max_task_operations",
                "background_poll_seconds",
                "reconciliation_seconds",
            )
        },
        "framework": {
            name: version(name)
            for name in (
                "langchain",
                "langgraph",
                "langchain-openai",
                "langgraph-checkpoint-postgres",
            )
        },
    }
