"""Native async tool middleware with a guarded execution boundary."""

import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from agent_service.schema import local_schema
from langgraph.types import Command

from langchain.agents.middleware import (
    AgentMiddleware,
    ToolErrorMiddleware as NativeToolErrorMiddleware,
)
from langchain_core.messages import ToolMessage

log = logging.getLogger(__name__)


class HookContractError(RuntimeError):
    pass


class PermissionDenied(ValueError):
    """A trusted argument validator may deny a valid tool input."""


@dataclass
class Call:
    runtime: object
    context: object
    call_id: str
    name: str
    batch: tuple = ()
    entered: bool = False
    violation: BaseException | None = None
    failure: BaseException | None = None


active_call = ContextVar("active_tool_call", default=None)
active_model_call = ContextVar("active_model_call", default=None)


class ToolLifecycleMiddleware(AgentMiddleware):
    def __init__(self, output_limits=None):
        self.output_limits = output_limits
        self.batches = {}

    @staticmethod
    def batch_key(context):
        return (getattr(context, "run_id", None), getattr(context, "agent_id", None))

    async def aafter_model(self, state, runtime):
        # First middleware's after_model runs last, before concurrent tool tasks.
        # Capture once here: sibling tool Hooks may mutate shared message state.
        messages = state.get("messages", [])
        calls = getattr(messages[-1], "tool_calls", ()) if messages else ()
        self.batches[self.batch_key(runtime.context)] = tuple(
            MappingProxyType({"name": c["name"], "id": c["id"]}) for c in calls
        )

    def limit_result(self, result):
        from agent_service.output import limit_output

        if isinstance(result, ToolMessage):
            from agent_service.output import safe_content, bounded_artifact

            content, truncated = limit_output(
                safe_content(result.content), *self.output_limits
            )
            artifact = bounded_artifact(result.artifact, self.output_limits[0])
            if truncated and isinstance(artifact, dict):
                artifact = {**artifact, "truncated": True}
            result = result.model_copy(
                update={"content": content, "artifact": artifact}
            )
        elif isinstance(result, Command) and isinstance(result.update, dict):
            messages = result.update.get("messages")
            if isinstance(messages, ToolMessage):
                result = replace(
                    result,
                    update={**result.update, "messages": self.limit_result(messages)},
                )
            elif isinstance(messages, (list, tuple)):
                result = replace(
                    result,
                    update={
                        **result.update,
                        "messages": [self.limit_result(m) for m in messages],
                    },
                )
        return result

    async def awrap_model_call(self, request, handler):
        call = Call(request.runtime, request.runtime.context, "", "model")
        token = active_model_call.set(call)
        try:
            result = await handler(request)
            if call.violation is not None:
                raise call.violation
            if call.failure is not None:
                raise call.failure
            return result
        finally:
            active_model_call.reset(token)

    async def awrap_tool_call(self, request, handler):
        call = Call(
            request.runtime,
            request.runtime.context,
            request.tool_call["id"],
            request.tool_call["name"],
        )
        call.batch = self.batches.get(self.batch_key(request.runtime.context), ())
        token = active_call.set(call)
        started = time.monotonic()
        try:
            log.info(
                "tool_call_started",
                extra={
                    "run_id": getattr(call.context, "run_id", None),
                    "call_id": call.call_id,
                    "tool_name": call.name,
                },
            )
        except Exception:
            pass
        outcome = "exception"
        try:
            result = await handler(request)
            # Hooks cannot suppress control failures or repeat execution.
            if call.violation is not None:
                raise call.violation
            if call.failure is not None:
                raise call.failure
            if self.output_limits:
                result = self.limit_result(result)
            outcome = getattr(result, "status", type(result).__name__)
            return result
        except BaseException as exc:
            outcome = type(exc).__name__
            raise
        finally:
            active_call.reset(token)
            try:
                log.info(
                    "tool_call_finished",
                    extra={
                        "run_id": getattr(call.context, "run_id", None),
                        "call_id": call.call_id,
                        "tool_name": call.name,
                        "duration_seconds": time.monotonic() - started,
                        "result_type": outcome,
                    },
                )
            except Exception:
                pass  # Observation must not change execution semantics.


class PermissionMiddleware(AgentMiddleware):
    def __init__(self, policy):
        self.policy = policy
        self.schemas = {
            name: local_schema(tool.tool_call_schema)
            for name, tool in policy.tools.items()
            if isinstance(tool.tool_call_schema, dict)
        }

    async def awrap_tool_call(self, request, handler):
        call = active_call.get()
        if call is None:
            raise HookContractError("Missing tool lifecycle boundary")
        if call.entered:
            call.violation = HookContractError(
                "A tool handler may be entered only once"
            )
            raise call.violation
        call.entered = True
        if (
            request.runtime is not call.runtime
            or request.runtime.context is not call.context
            or request.tool_call["id"] != call.call_id
        ):
            call.violation = HookContractError("Hook replaced trusted tool identity")
            raise call.violation
        deadline = getattr(call.context, "deadline_at", None)
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            call.failure = TimeoutError("Tool execution deadline expired")
            raise call.failure
        call.name = request.tool_call["name"]
        policy = self.policy
        denied = (
            policy.tools.get(call.name) is not request.tool
            or request.tool is None
            or (
                policy.binding_fingerprint is not None
                and getattr(call.context, "binding_fingerprint", None)
                != policy.binding_fingerprint
            )
        )
        try:
            if not denied and call.name in self.schemas:
                try:
                    self.schemas[call.name].validate(
                        deepcopy(request.tool_call["args"])
                    )
                except Exception:
                    return ToolMessage(
                        content='{"code":"INVALID_TOOL_ARGUMENTS","message":"Arguments do not match the tool schema."}',
                        tool_call_id=call.call_id,
                        name=call.name,
                        status="error",
                    )
            if not denied and call.name in policy.validators:
                # The same tool schema used by the native executor parses arguments.
                # Invalid inputs are left to native validation/error formatting.
                from pydantic import ValidationError

                schema = request.tool.tool_call_schema
                try:
                    parsed = (
                        deepcopy(request.tool_call["args"])
                        if isinstance(schema, dict)
                        else schema.model_validate(request.tool_call["args"])
                    )
                except ValidationError:
                    parsed = None
                if parsed is not None:
                    denied = policy.validators[call.name](parsed, call.context) is False
        except PermissionDenied:
            denied = True
        except BaseException as exc:
            call.failure = exc
            raise
        if denied:
            return ToolMessage(
                content='{"code":"PERMISSION_DENIED","message":"Tool or arguments are not permitted for this run."}',
                tool_call_id=call.call_id,
                name=call.name,
                status="error",
            )
        try:
            return await handler(request)
        except BaseException as exc:
            call.failure = exc
            raise


class ToolErrorMiddleware(NativeToolErrorMiddleware):
    """Recover only explicitly listed execution errors; never control exceptions."""

    def __init__(self, recoverable):
        from langchain.agents.middleware.model_call_limit import (
            ModelCallLimitExceededError,
        )
        from langchain.agents.middleware.tool_call_limit import (
            ToolCallLimitExceededError,
        )
        from langgraph.errors import GraphBubbleUp

        from agent_service.execution_errors import ExecutionFailure

        controls = (
            ExecutionFailure,
            HookContractError,
            TimeoutError,
            ModelCallLimitExceededError,
            ToolCallLimitExceededError,
            GraphBubbleUp,
        )
        if any(
            not isinstance(t, type)
            or not issubclass(t, Exception)
            or any(issubclass(t, c) or issubclass(c, t) for c in controls)
            for t in recoverable
        ):
            raise ValueError(
                "Recoverable errors must exclude execution control exceptions"
            )
        recoverable = tuple(recoverable)

        def on_error(exc, request):
            if isinstance(exc, recoverable):
                return (
                    '{"code":"TOOL_EXECUTION_ERROR","message":"Tool execution failed."}'
                )
            return None

        super().__init__(on_error)


class ExecutionControlMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler):
        call = active_model_call.get()
        if (
            call.entered
            or request.runtime is not call.runtime
            or request.runtime.context is not call.context
        ):
            call.violation = HookContractError(
                "Hook repeated a model call or replaced trusted context"
            )
            raise call.violation
        call.entered = True
        try:
            return await handler(request)
        except BaseException as exc:
            call.failure = exc
            raise

    async def awrap_tool_call(self, request, handler):
        import asyncio
        from langchain.agents.middleware.model_call_limit import (
            ModelCallLimitExceededError,
        )
        from langchain.agents.middleware.tool_call_limit import (
            ToolCallLimitExceededError,
        )
        from langgraph.errors import GraphBubbleUp

        try:
            return await handler(request)
        except (
            asyncio.CancelledError,
            TimeoutError,
            GraphBubbleUp,
            ModelCallLimitExceededError,
            ToolCallLimitExceededError,
        ) as exc:
            call = active_call.get()
            call.failure = exc
            raise
