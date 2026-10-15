"""An explicit allowlist for persisted configuration; never serialize resources."""

from importlib.metadata import version
from urllib.parse import urlsplit

from agent_service.profiles import fingerprint, tool_schema


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
    profile = registry.resolve(binding)
    return {
        "version": 1,
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
        "run_timeout_seconds": settings.run_timeout,
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
