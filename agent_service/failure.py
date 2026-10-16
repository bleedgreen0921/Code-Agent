"""Trusted, bounded failure facts. Recovery values describe advice, never authority."""

import asyncio
from typing import Literal

import psycopg
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FailureInfo(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    version: Literal[1] = 1
    code: str = Field(min_length=1, max_length=96, pattern=r"^[A-Z][A-Z0-9_]*$")
    category: Literal[
        "parameter",
        "permission",
        "transport",
        "timeout",
        "protocol",
        "execution",
        "quota",
        "configuration",
        "persistence",
        "cleanup",
        "verification",
    ]
    effect: Literal["not_dispatched", "possible_partial", "unknown", "not_applicable"]
    recovery: Literal[
        "correct_arguments", "reread", "retry_observation", "reconcile_original", "stop"
    ]
    message: str

    @model_validator(mode="after")
    def bounded(self):
        # Literal[int] alone accepts True in Pydantic; strict version is checked before parsing.
        if (
            len(self.message.encode("utf-8")) > 512
            or len(self.model_dump_json().encode("utf-8")) > 2048
        ):
            raise ValueError("Failure information exceeds its limit")
        return self

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, data):
        if (
            isinstance(data, dict)
            and "version" in data
            and type(data["version"]) is not int
        ):
            raise ValueError("Failure version must be an integer")
        return data


# Explicit membership only. Neither remote messages nor model-visible text classify errors.
SPECS = {}


def _register(codes, category, effect, recovery, message):
    for code in codes.split():
        SPECS[code] = (category, effect, recovery, message)


_register(
    "VERIFICATION_FAILED VERIFICATION_INCONCLUSIVE VERIFICATION_CONTRACT_ERROR VERIFICATION_EXECUTION_FAILED",
    "verification",
    "not_applicable",
    "stop",
    "Completion verification did not pass.",
)
_register(
    "NO_PROGRESS SUBAGENT_NO_PROGRESS",
    "execution",
    "not_applicable",
    "stop",
    "Repeated correctable tool failures prevented progress.",
)


_register(
    "INVALID_TOOL_ARGUMENTS FILE_ARGUMENTS_INVALID",
    "parameter",
    "not_dispatched",
    "correct_arguments",
    "Arguments do not match the operation contract.",
)
_register(
    "PERMISSION_DENIED BACKGROUND_WORKSPACE_READ_ONLY MCP_WRITE_BATCH_REJECTED BACKGROUND_START_BATCH_REJECTED BACKGROUND_REQUIRES_TASK_ROOT BACKGROUND_OPERATION_NOT_FOUND",
    "permission",
    "not_dispatched",
    "correct_arguments",
    "Operation was rejected before dispatch.",
)
_register(
    "FILE_VERSION_CONFLICT",
    "verification",
    "not_dispatched",
    "reread",
    "File version changed before publication; read the file again.",
)
_register(
    "SHELL_COMMAND_FAILED BACKGROUND_COMMAND_FAILED MCP_WRITE_ERROR MCP_TASK_EXECUTION_FAILED",
    "execution",
    "possible_partial",
    "stop",
    "Operation reported failure; partial effects may remain.",
)
_register(
    "BACKGROUND_NOT_STARTED",
    "execution",
    "not_dispatched",
    "stop",
    "Operation did not start.",
)
_register(
    "EXECUTION_FAILED TOOL_EXECUTION_ERROR FILE_OPERATION_FAILED",
    "execution",
    "unknown",
    "stop",
    "Execution failed; effects are not confirmed.",
)
_register(
    "COMMAND_TIMEOUT BACKGROUND_TIMEOUT RUN_TIMEOUT",
    "timeout",
    "unknown",
    "reconcile_original",
    "Execution deadline expired; effects are not confirmed.",
)
_register(
    "LEASE_EXPIRED WORKER_STOPPED EXECUTION_ENVIRONMENT_LOST BACKGROUND_LEASE_LOST MCP_LEASE_LOST",
    "execution",
    "unknown",
    "stop",
    "Execution authority ended; effects are not confirmed.",
)
_register(
    "QUEUE_TIMEOUT",
    "timeout",
    "not_applicable",
    "stop",
    "Run expired while queued; this does not describe tool effects.",
)
_register(
    "MODEL_TIMEOUT", "timeout", "not_applicable", "stop", "Model request timed out."
)
_register(
    "MODEL_AUTH_FAILED",
    "permission",
    "not_applicable",
    "stop",
    "Model authentication failed.",
)
_register(
    "MODEL_RATE_LIMITED QUOTA_EXCEEDED BACKGROUND_CAPACITY_EXCEEDED",
    "quota",
    "not_applicable",
    "stop",
    "Execution limit was reached.",
)
_register(
    "MODEL_CALL_FAILED", "transport", "not_applicable", "stop", "Model request failed."
)
_register(
    "HOOK_CONTRACT_ERROR GRAPH_INTERRUPTED",
    "protocol",
    "not_applicable",
    "stop",
    "Execution contract was not satisfied.",
)
_register(
    "PERSISTENCE_FAILED MCP_DISPATCH_UNCONFIRMED MCP_RESULT_UNCONFIRMED BACKGROUND_DISPATCH_UNCONFIRMED",
    "persistence",
    "unknown",
    "reconcile_original",
    "Persistence was not confirmed; reconcile the original record.",
)
_register(
    "RESOURCE_CLEANUP_FAILED BACKGROUND_CLEANUP_UNCONFIRMED",
    "cleanup",
    "unknown",
    "reconcile_original",
    "Original resource release is not confirmed.",
)
_register(
    "BACKGROUND_RESULT_UNKNOWN MCP_RESULT_UNKNOWN BACKGROUND_CREATE_UNCONFIRMED BACKGROUND_STOP_UNCONFIRMED MCP_CANCEL_UNCONFIRMED BACKGROUND_RECONCILIATION_FAILED",
    "transport",
    "unknown",
    "reconcile_original",
    "Original operation outcome is not confirmed.",
)
_register(
    "BACKGROUND_RECEIPT_INVALID MCP_TASK_RECEIPT_INVALID MCP_TASK_PROTOCOL_ERROR MCP_MESSAGE_TOO_LARGE MCP_PROTOCOL_ERROR",
    "protocol",
    "unknown",
    "reconcile_original",
    "Operation receipt did not satisfy the protocol contract.",
)
_register(
    "BACKGROUND_RESOURCE_IDENTITY_CHANGED MCP_TASK_IDENTITY_CHANGED BACKGROUND_REPLAY_DENIED MCP_REPLAY_DENIED BACKGROUND_MONITOR_LEASE_LOST BACKGROUND_STATE_CONFLICT",
    "verification",
    "unknown",
    "stop",
    "Original operation identity or authority does not match.",
)
_register(
    "CONFIGURATION_CHANGED EXECUTION_ENVIRONMENT_UNAVAILABLE BACKGROUND_CONFIGURATION_CHANGED BACKGROUND_DAEMON_CHANGED BACKGROUND_IMAGE_UNSUPPORTED BACKGROUND_WORKSPACE_CHANGED MCP_TASK_CONTRACT_CHANGED MCP_TASKS_UNSUPPORTED MCP_CREDENTIALS_INVALID",
    "configuration",
    "not_applicable",
    "stop",
    "Required execution configuration is unavailable or changed.",
)
_register(
    "MCP_CALL_FAILED MCP_CONNECTION_CLOSED MCP_CONNECTION_FAILED MCP_INTERACTION_UNSUPPORTED",
    "transport",
    "unknown",
    "stop",
    "External operation could not be confirmed.",
)

_register(
    "BACKGROUND_RESULT_NOT_READY",
    "execution",
    "not_dispatched",
    "retry_observation",
    "Original result is not ready for observation.",
)


_register(
    "BACKGROUND_OBSERVATION_EXHAUSTED BACKGROUND_OBSERVATION_INTERRUPTED BACKGROUND_OBSERVATION_TRANSIENT MCP_CANCEL_TRANSPORT_FAILED",
    "transport",
    "unknown",
    "reconcile_original",
    "Original resource observation or cancellation is not confirmed.",
)


def failure_info(code="EXECUTION_FAILED", *, effect=None, recovery=None):
    code = code if code in SPECS else "EXECUTION_FAILED"
    category, default_effect, default_recovery, message = SPECS[code]
    return FailureInfo(
        code=code,
        category=category,
        effect=effect or default_effect,
        recovery=recovery or default_recovery,
        message=message,
    ).model_dump()


def validated_info(value):
    return None if value is None else FailureInfo.model_validate(value).model_dump()


def control_types():
    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
    from langgraph.errors import GraphBubbleUp, NodeCancelledError
    from agent_service.execution_errors import ExecutionFailure
    from agent_service.policy import HookContractError

    return (
        asyncio.CancelledError,
        NodeCancelledError,
        GraphBubbleUp,
        ExecutionFailure,
        TimeoutError,
        ModelCallLimitExceededError,
        ToolCallLimitExceededError,
        HookContractError,
        psycopg.Error,
    )


def is_control(exc):
    return isinstance(exc, control_types()) or (
        isinstance(exc, BaseExceptionGroup)
        and any(is_control(e) for e in exc.exceptions)
    )


def control_leaf(exc):
    if isinstance(exc, BaseExceptionGroup):
        return next((control_leaf(e) for e in exc.exceptions if is_control(e)), exc)
    return exc


def exception_code(exc):
    import openai
    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
    from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
    from agent_service.execution_errors import ExecutionFailure
    from agent_service.profiles import ConfigurationChanged
    from agent_service.policy import HookContractError

    exc = control_leaf(exc)
    if isinstance(exc, HookContractError):
        return "HOOK_CONTRACT_ERROR"
    if isinstance(exc, psycopg.Error):
        return "PERSISTENCE_FAILED"
    if isinstance(exc, ExecutionFailure):
        return exc.code
    if isinstance(exc, ConfigurationChanged):
        return "CONFIGURATION_CHANGED"
    if isinstance(exc, (ModelCallLimitExceededError, ToolCallLimitExceededError)):
        return "QUOTA_EXCEEDED"
    if isinstance(exc, openai.APITimeoutError):
        return "MODEL_TIMEOUT"
    if isinstance(exc, TimeoutError):
        return "RUN_TIMEOUT"
    if isinstance(exc, openai.AuthenticationError):
        return "MODEL_AUTH_FAILED"
    if isinstance(exc, openai.RateLimitError):
        return "MODEL_RATE_LIMITED"
    if isinstance(exc, openai.APIError):
        return "MODEL_CALL_FAILED"
    return "EXECUTION_FAILED"
