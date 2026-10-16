"""Trusted recovery policy and narrow transport classifiers; never retry business I/O."""

import asyncio
import math
import random
from datetime import datetime, timezone
from contextvars import ContextVar

import psycopg
from psycopg_pool import PoolTimeout
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReliabilitySettings(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False
    )
    observation_max_failures: int = Field(default=5, ge=1)
    observation_window: float = Field(default=30, gt=0)
    observation_backoff_base: float = Field(default=1, gt=0)
    observation_backoff_cap: float = Field(default=8, gt=0)
    cleanup_backoff_base: float = Field(default=8, gt=0)
    cleanup_backoff_cap: float = Field(default=60, gt=0)
    publication_reserve: float = Field(default=2, gt=0)
    cancel_max_attempts: int = Field(default=3, ge=1, le=3)
    maintenance_max_failures: int = Field(default=5, ge=1)
    maintenance_window: float = Field(default=30, gt=0)
    maintenance_backoff_base: float = Field(default=1, gt=0)
    maintenance_backoff_cap: float = Field(default=8, gt=0)

    @model_validator(mode="after")
    def intervals(self):
        for prefix in ("observation", "cleanup", "maintenance"):
            if getattr(self, prefix + "_backoff_cap") < getattr(
                self, prefix + "_backoff_base"
            ):
                raise ValueError("Backoff cap must be at least its base")
        return self


def background_policy(registry, settings):
    return registry.reliability.model_dump() | {
        "poll_floor_seconds": settings.background_poll_seconds,
        "monitor_lease_seconds": settings.reconciliation_seconds,
    }


def backoff(failures, base, cap, *, uniform=random.uniform):
    return min(cap, base * 2 ** min(max(0, failures - 1), 30) * uniform(0.8, 1.2))


def transient_database(exc):
    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            transient_database(e) for e in exc.exceptions
        )
    if isinstance(exc, PoolTimeout):
        return True
    # Do not infer transience from arbitrary OperationalError messages. These states
    # identify connection loss/unavailability; authentication and SQL errors fail closed.
    state = getattr(exc, "sqlstate", None)
    return isinstance(exc, psycopg.OperationalError) and (
        state
        in {"08000", "08001", "08003", "08006", "08007", "57P01", "57P02", "57P03"}
        or (
            state is None
            and any(
                part in str(exc).lower()
                for part in (
                    "connection is closed",
                    "connection was closed",
                    "connection refused",
                    "connection reset",
                    "server closed the connection",
                    "could not connect",
                    "connection not open",
                    "commit ack",
                    "commit acknowledgement",
                )
            )
        )
    )


def transient_observation(exc):
    import httpx
    import httpx2
    import anyio
    from agent_service.execution_errors import EnvironmentUnavailable

    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            transient_observation(e) for e in exc.exceptions
        )
    if (
        type(exc) is RuntimeError
        and (
            str(exc)
            in {
                "Server session was closed unexpectedly",
                "Failed to initialize server session",
            }
            or str(exc).startswith("Client failed to connect:")
        )
        and exc.__cause__ is not None
    ):
        return transient_observation(exc.__cause__)
    if isinstance(exc, (httpx.RemoteProtocolError, httpx2.RemoteProtocolError)):
        return any(
            part in str(exc).lower()
            for part in (
                "server disconnected without sending a response",
                "peer closed connection",
                "incomplete chunked read",
                "connection closed",
            )
        )
    if isinstance(exc, EnvironmentUnavailable):
        # Docker CLI has no structured daemon transport errors. Match only explicit
        # connection diagnostics at the observation boundary, never its broad class.
        message = str(exc).lower()
        return any(
            s in message
            for s in (
                "cannot connect to the docker daemon",
                "connection refused",
                "connection reset by peer",
                "unexpected eof",
                "i/o timeout",
            )
        )
    return isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            anyio.EndOfStream,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx2.ConnectError,
            httpx2.ReadError,
            httpx2.ReadTimeout,
            httpx2.ConnectTimeout,
        ),
    )


observation_call_limit = ContextVar("observation_call_limit", default=None)


class ObservationFailure(Exception):
    """Only an explicitly bounded observation can create this marker."""


def transient_client_error(exc, client):
    from mcp.shared.exceptions import MCPError
    from mcp.types import CONNECTION_CLOSED, REQUEST_TIMEOUT

    if isinstance(exc, BaseExceptionGroup):
        return bool(exc.exceptions) and all(
            transient_client_error(e, client) for e in exc.exceptions
        )
    return transient_observation(exc) or (
        isinstance(exc, MCPError)
        and (
            exc.code == REQUEST_TIMEOUT
            or (
                exc.code == CONNECTION_CLOSED
                and getattr(client, "observation_transport", {}).get("transient")
                is True
            )
        )
    )


async def observe(awaitable, *, client=None):
    try:
        limit = observation_call_limit.get()
        if limit is None:
            return await awaitable
        async with asyncio.timeout(limit):
            return await awaitable
    except Exception as exc:
        if transient_client_error(exc, client):
            raise ObservationFailure() from None
        raise


def poll_interval(value):
    # PostgreSQL bigint, timedelta and local scheduling must all represent the value.
    if value is None:
        return None
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or value > 2_147_483_647
    ):
        from agent_service.external_operations import ExternalFailure

        raise ExternalFailure("MCP_TASK_PROTOCOL_ERROR")
    return math.ceil(value)


def remaining_execution(context, maximum=math.inf):
    """Shared deadline reader for models, tools, summaries and future validators."""
    if isinstance(context, dict):
        deadline = context.get("effective_deadline_at", context.get("deadline_at"))
        lease = context.get("_lease_until")
    else:
        deadline = getattr(context, "effective_deadline_at", None) or getattr(
            context, "deadline_at", None
        )
        lease = getattr(context, "execution_authority", None)
        lease = lease.get("_lease_until") if lease else None
    now = datetime.now(timezone.utc)
    authority = (
        context
        if isinstance(context, dict)
        else getattr(context, "execution_authority", None)
    )
    if authority and authority.get("_execution_closed"):
        raise TimeoutError("Execution has closed")
    if lease is not None and lease <= now:
        raise TimeoutError("Execution lease expired")
    # The Worker's local watchdog follows renewed lease confirmations while a call
    # is in progress. A captured old lease must not shorten a healthy long call.
    seconds = min(maximum, (deadline - now).total_seconds() if deadline else math.inf)
    if seconds <= 0:
        raise TimeoutError("Execution authority expired")
    return seconds
