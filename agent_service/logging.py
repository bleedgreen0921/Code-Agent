"""Shared bounded JSON logging. Tracebacks contain locations, never exception text."""

import json
import logging
import math
from pathlib import Path
from uuid import UUID

FIELDS = (
    "task_id",
    "run_id",
    "operation_id",
    "agent_id",
    "call_id",
    "tool_name",
    "duration_seconds",
    "result_type",
    "status",
    "error_code",
)
STATS = (
    "history_tokens_approx",
    "request_tokens_approx",
    "reported_usage",
    "delegation",
)
EVENTS = frozenset(
    "tool_call_started tool_call_finished run_transition task_transition background_transition run_context_budget run_context_prepared model_context delegation_aborted result_publication_unconfirmed lease_renewal_unconfirmed run_execution_failed run_reported_usage execution_environment_preflight_failed background_diagnostic_unconfirmed background_monitor_release_unconfirmed".split()
)


def text(value):
    value = value.encode("utf-8", "replace")[:256].decode("utf-8", "ignore")
    # JSON escaping can expand control characters sixfold. Bound that form too.
    while len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 258:
        value = value[:-1]
    return value


def clean(value, depth=0):
    if depth > 10:
        return "[depth limit]"
    if isinstance(value, (str, UUID)):
        return text(str(value))
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            text(k): clean(v, depth + 1)
            for k, v in list(value.items())[:100]
            if isinstance(k, str)
        }
    if isinstance(value, (list, tuple)):
        return [clean(v, depth + 1) for v in value[:100]]
    return "[unsupported value]"


def stack(exc, depth=0):
    frames = []
    tb = exc.__traceback__
    while tb is not None and len(frames) < 64:
        code = tb.tb_frame.f_code
        frames.append(
            {
                "file": text(str(Path(code.co_filename))),
                "function": text(code.co_name),
                "line": tb.tb_lineno,
            }
        )
        tb = tb.tb_next
    item = {"type": text(type(exc).__name__), "frames": frames}
    if isinstance(exc, BaseExceptionGroup) and depth < 4:
        item["exceptions"] = [stack(e, depth + 1) for e in exc.exceptions[:16]]
    # Do not traverse cause/context or stringify exceptions, frames or locals.
    return item


class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            payload = {
                "level": text(record.levelname),
                "logger": text(record.name),
                "message": record.msg
                if isinstance(record.msg, str) and record.msg in EVENTS
                else "log_event",
            }
            for key in FIELDS + STATS:
                if hasattr(record, key):
                    payload[key] = clean(getattr(record, key))
            if record.exc_info and isinstance(record.exc_info[1], BaseException):
                payload["exception"] = stack(record.exc_info[1])

            def encode():
                return json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )

            result = encode()
            # Deterministic trimming preserves all correlation fields and the error code.
            for key in (*reversed(STATS), "exception"):
                if len(result.encode("utf-8")) <= 16384:
                    break
                if key == "exception" and key in payload:
                    payload[key] = {
                        "type": payload[key]["type"],
                        "frames": payload[key]["frames"][-8:],
                        "truncated": True,
                    }
                else:
                    payload.pop(key, None)
                payload["truncated"] = True
                result = encode()
            return result
        except Exception:
            return '{"level":"ERROR","message":"log_serialization_failed"}'


def install_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
