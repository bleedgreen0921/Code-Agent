"""Opt-in native middleware; evidence lives only in trusted lifecycle memory."""

import hashlib
import json
import logging

from langchain.agents.middleware import AgentMiddleware
from agent_service.execution_errors import ExecutionFailure

CODES = ("INVALID_TOOL_ARGUMENTS", "PERMISSION_DENIED", "FILE_VERSION_CONFLICT")
THRESHOLD = 3
log = logging.getLogger(__name__)


def manifest():
    return {
        "version": 1,
        "codes": list(CODES),
        "threshold": THRESHOLD,
        "subagents": "isolated_return_failure",
    }


class NoProgress(ExecutionFailure):
    code = "NO_PROGRESS"


def argument_fingerprint(name, args, code):
    if code not in CODES:
        return None

    def valid(value):
        if type(value) is dict:
            return all(type(k) is str and valid(v) for k, v in value.items())
        if type(value) is list:
            return all(valid(v) for v in value)
        return value is None or type(value) in (str, int, float, bool)

    try:
        if not valid(args):
            return None
        payload = json.dumps(
            [name, args, code],
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None


class NoProgressMiddleware(AgentMiddleware):
    def __init__(self):
        self.order = ()
        self.results = {}
        self.last = None
        self.count = 0

    async def abefore_agent(self, state, runtime):
        self.order, self.results = (), {}
        self.last, self.count = None, 0

    def batch(self, calls):
        self.order = tuple(c["id"] for c in calls)
        self.results = {}
        if len(set(self.order)) != len(self.order):
            # Native call IDs cannot disambiguate these results. An ambiguous
            # batch is never evidence of a repeated trusted outcome.
            self.order, self.last, self.count = (), None, 0

    def record(self, call):
        self.results[call.call_id] = (
            call.trusted_fingerprint,
            call.name,
            call.trusted_code,
        )

    async def abefore_model(self, state, runtime):
        # Model-declared order, independent of completion order. Never read artifacts
        # or inherited checkpoint messages to manufacture trusted evidence.
        for identity in self.order:
            fingerprint, name, code = self.results.get(identity, (None, None, None))
            self.count = (
                self.count + 1
                if fingerprint and fingerprint == self.last
                else int(bool(fingerprint))
            )
            self.last = fingerprint
            if fingerprint:
                log.info(
                    "no_progress_observed",
                    extra={
                        "run_id": runtime.context.run_id,
                        "agent_id": runtime.context.agent_id,
                        "tool_name": name,
                        "error_code": code,
                        "repetition_count": self.count,
                    },
                )
        self.order, self.results = (), {}
        if self.count >= THRESHOLD:
            raise NoProgress("Repeated correctable tool failure")
