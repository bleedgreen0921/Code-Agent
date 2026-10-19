"""Same-graph repair submission gate and serialized candidate tool boundary."""

import asyncio
import json
from datetime import datetime, timedelta, timezone

import psycopg
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException

from agent_service.code_artifacts import CodeFailure
from agent_service.execution_errors import CommandTimeout
from agent_service.failure import is_control
from agent_service.policy import active_call
from agent_service.profiles import fingerprint


class RepairExecutionMiddleware(AgentMiddleware):
    def __init__(self, scope, names):
        self.scope, self.names = scope, set(names)

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] not in self.names:
            return await handler(request)
        scope = self.scope
        try:
            async with scope.lock:
                scope.check_open()
                if scope.write_closed:
                    raise CodeFailure(
                        "PERMISSION_DENIED", "Candidate control entry is closed"
                    )
                name, identity = request.tool_call["name"], request.tool_call["id"]
                request_digest = fingerprint(
                    {"name": name, "args": request.tool_call["args"]}
                )
                if identity in scope.tool_results:
                    saved_digest, saved_result = scope.tool_results[identity]
                    if saved_digest != request_digest:
                        raise CodeFailure(
                            "CHECK_RESULT_UNKNOWN",
                            "Tool identity was reused with different input",
                        )
                    return saved_result
                await scope.healthy()
                # Checks have their own per-check and Run deadlines.
                timeout = (
                    scope.remaining()
                    if name == "run_project_check"
                    else min(scope.remaining(), scope.settings.command_timeout)
                )
                async with asyncio.timeout(timeout):
                    result = await handler(request)
                    if name in ("shell", "edit_file"):
                        successful_write = name == "shell" or (
                            isinstance(result, ToolMessage)
                            and result.status == "success"
                        )
                        if successful_write:
                            revision = await scope.checkpoint("tool:" + identity)
                            if isinstance(result, ToolMessage):
                                result = result.model_copy(
                                    update={
                                        "artifact": {
                                            **(result.artifact or {}),
                                            "revision_id": str(revision["id"]),
                                            "content_digest": revision[
                                                "content_digest"
                                            ],
                                        }
                                    }
                                )
                await scope.healthy()
                scope.tool_results[identity] = (request_digest, result)
                return result
        except ToolException as exc:
            return ToolMessage(
                content=str(exc),
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )
        except TimeoutError as exc:
            scope.failure = CommandTimeout("Repair tool deadline expired")
            raise scope.failure from exc
        except BaseException as exc:
            if is_control(exc):
                call = active_call.get()
                if call:
                    call.failure = exc
                scope.failure = exc
                scope.closing = True
            raise


class RepairGateMiddleware(AgentMiddleware):
    def __init__(self, scope):
        self.scope = scope
        self.seen = {}
        self.scope.usage = {}

    async def abefore_model(self, state, runtime):
        self.scope.usage = self.usage(state)
        await self.scope.store.record_repair_usage(self.scope.row, self.scope.usage)

    def usage(self, state):
        scope = self.scope
        persistent = scope.delivery.persistent_usage if scope.delivery else {}
        return dict(
            model_calls=persistent.get(
                "model_calls", state.get("run_model_call_count", 0)
            ),
            tool_calls=persistent.get(
                "tool_calls", state.get("run_tool_call_count", {}).get("__all__", 0)
            ),
            check_calls=len(scope.checks),
            gate_rounds=scope.rounds,
            admitted_bytes=scope.candidate["admitted_bytes"],
            remaining_seconds=max(0, int(scope.remaining())),
        )

    def retry(self, feedback):
        return {
            "structured_response": None,
            "jump_to": "model",
            "messages": [
                HumanMessage(
                    content="Repair gate feedback (trusted controller): "
                    + json.dumps(feedback, ensure_ascii=False)
                )
            ],
        }

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        scope = self.scope
        scope.usage = self.usage(state)
        await scope.store.record_repair_usage(scope.row, scope.usage)
        ai = next(
            (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
        )
        if ai is None:
            raise CodeFailure("REPORT_INVALID", "Final model message missing")
        reports = [c for c in ai.tool_calls if c["name"] == "RepairDraftV1"]
        if reports and len(ai.tool_calls) != 1:
            # ToolStrategy consumes only report calls; acknowledge every other call
            # without executing any part of a mixed submission batch.
            existing = {
                m.tool_call_id
                for m in state["messages"][state["messages"].index(ai) + 1 :]
                if isinstance(m, ToolMessage)
            }
            errors = [
                ToolMessage(
                    content="Mixed report/tool batch rejected; no ordinary tools executed.",
                    tool_call_id=c["id"],
                    name=c["name"],
                    status="error",
                )
                for c in ai.tool_calls
                if c["id"] not in existing
            ]
            update = self.retry(
                {"reason": "Submit RepairDraftV1 alone in its own tool batch."}
            )
            update["messages"] = errors + update["messages"]
            return update
        if not reports:
            if not ai.tool_calls:
                return self.retry(
                    {
                        "reason": "A candidate requires a valid RepairDraftV1 submission; plain final text cannot complete repair."
                    }
                )
            return None
        draft = state.get("structured_response")
        if draft is None:
            # ToolStrategy already provided schema validation feedback.
            return {"structured_response": None, "jump_to": "model"}
        identity = ai.id or ("tool-call:" + reports[0]["id"])
        raw = draft.model_dump(mode="json")
        if identity in self.seen:
            saved_digest, saved_update = self.seen[identity]
            if saved_digest != fingerprint(raw):
                raise CodeFailure(
                    "REPORT_INVALID",
                    "Final message identity was reused with a different draft",
                )
            return saved_update
        attempt, created = await scope.store.reserve_repair_attempt(
            scope.row, scope.candidate["id"], identity, raw
        )
        if not created:
            if attempt["status"] == "running":
                raise CodeFailure(
                    "CHECK_RESULT_UNKNOWN",
                    "Gate execution interrupted; checks will not be replayed",
                )
            if attempt["status"] == "passed":
                scope.passed, scope.write_closed = attempt, True
                return {"jump_to": "end"}
            return self.retry(attempt["feedback"])
        scope.rounds += 1
        checks, revision = [], None
        async with scope.lock:
            scope.check_open()
            end = min(
                scope.row["effective_deadline_at"],
                datetime.now(timezone.utc)
                + timedelta(seconds=scope.code.settings.round_seconds),
            )
            failure = None
            try:
                scope.evidence.validate(draft)
                await scope.quiescent()
                revision = await scope.checkpoint("gate:" + str(attempt["id"]))
                scope.evidence.validate(draft)
                for check_id in scope.row["_code"]["check_ids"]:
                    checks.append(
                        await scope.run_check(
                            scope.code.checks[check_id],
                            "gate:" + str(attempt["id"]) + ":dev:" + check_id,
                            revision,
                            round_deadline=end,
                        )
                    )
                checks.append(
                    await scope.run_check(
                        scope.acceptance,
                        "gate:" + str(attempt["id"]) + ":acceptance",
                        revision,
                        purpose="acceptance",
                        round_deadline=end,
                    )
                )
                passed = all(
                    c["result"]["status"] == "passed" and c["result"]["complete"]
                    for c in checks
                )
                if passed and scope.delivery:
                    passed, merged_checks, failure = await scope.delivery.validate(
                        revision, checks
                    )
                    if not passed:
                        checks = checks + merged_checks
                if not passed and failure is None:
                    failure = (
                        "Required checks failed or could not confirm the candidate."
                    )
            except (ValueError, ToolException) as exc:
                passed = False
                failure = str(exc)
            usage = scope.usage = self.usage(state)
            feedback = dict(
                attempt_id=str(attempt["id"]),
                revision_id=str(revision["id"]) if revision else None,
                passed=passed,
                reason=failure,
                checks=checks,
                unfinished_checks=[
                    c
                    for c in scope.row["_code"]["check_ids"]
                    if c not in {v["check_id"] for v in checks}
                ]
                + (
                    []
                    if any(c["purpose"] == "acceptance" for c in checks)
                    else ["acceptance:" + scope.acceptance.id]
                ),
                remaining=dict(
                    model_calls=max(
                        0, scope.code.settings.model_calls - usage["model_calls"]
                    ),
                    tool_calls=max(
                        0, scope.code.settings.tool_calls - usage["tool_calls"]
                    ),
                    checks=max(0, scope.code.settings.check_calls - len(scope.checks)),
                    rounds=max(0, scope.code.settings.gate_rounds - scope.rounds),
                    seconds=usage["remaining_seconds"],
                ),
            )
            # Include real bounded diagnostics; model text never manufactures a pass.
            feedback["checks"] = []
            for check in checks:
                _, log = await scope.store.code_artifact(
                    scope.row["task_id"], check["log_artifact_id"]
                )
                _, result_data = await scope.store.code_artifact(
                    scope.row["task_id"], check["result_artifact_id"]
                )
                try:
                    events = json.loads(json.loads(result_data)["protocol"]).get(
                        "events", []
                    )
                except (ValueError, TypeError, KeyError):
                    events = []
                feedback["checks"].append(
                    {
                        **check,
                        "diagnostic": log[:2048].decode("utf-8", "replace"),
                        "failed_cases": [
                            e for e in events if e.get("outcome") != "passed"
                        ][:32],
                    }
                )
            for retry in range(3):
                try:
                    saved = await scope.store.finish_repair_attempt(
                        scope.row,
                        attempt["id"],
                        revision["id"] if revision else None,
                        [c["id"] for c in checks],
                        feedback,
                        passed,
                        usage,
                    )
                    break
                except (psycopg.Error, OSError, TimeoutError):
                    if retry == 2:
                        raise
            if passed:
                scope.write_closed = True
                scope.passed = saved
                scope.candidate["frozen_revision_id"] = revision["id"]
                update = {"jump_to": "end"}
            else:
                if (
                    scope.rounds >= scope.code.settings.gate_rounds
                    or len(scope.checks) >= scope.code.settings.check_calls
                    or scope.remaining() < 1
                ):
                    raise CodeFailure(
                        "REPAIR_BUDGET_EXHAUSTED", failure or "Repair budget exhausted"
                    )
                update = self.retry(feedback)
            self.seen[identity] = (fingerprint(raw), update)
            return update
