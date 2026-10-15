"""Native async tool middleware with a guarded execution boundary."""

import logging
import time
from copy import deepcopy
from datetime import datetime, timezone
from contextvars import ContextVar
from dataclasses import dataclass

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
    entered: bool = False
    violation: BaseException | None = None
    failure: BaseException | None = None


active_call = ContextVar("active_tool_call", default=None)
active_model_call = ContextVar("active_model_call", default=None)


class ToolLifecycleMiddleware(AgentMiddleware):
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

        controls = (
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
