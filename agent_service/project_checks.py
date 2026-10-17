"""Controlled project checks: server classification never trusts stdout or exit alone."""

import json
from collections import Counter

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from agent_service.code_artifacts import CodeFailure
from agent_service.policy import active_call


class ProjectCheckInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    check_id: str = Field(min_length=1, max_length=128)


def classify_result(
    raw,
    *,
    exit_code,
    expected_cases=(),
    adapter=None,
    timed_out=False,
    oom=False,
    truncated=False,
):
    limitations = []
    counts = Counter()
    selected = executed = 0
    valid = False
    try:
        value = json.loads(raw)
        required = {
            "protocol",
            "adapter",
            "complete",
            "selected",
            "started",
            "finished",
            "events",
            "collection_errors",
            "exit_code",
        }
        if (
            type(value) is not dict
            or set(value) != required
            or type(value["protocol"]) is not int
            or value["protocol"] != 1
            or type(value["complete"]) is not bool
        ):
            raise ValueError("Invalid envelope")
        if adapter and value["adapter"] != adapter:
            raise ValueError("Adapter mismatch")
        if type(value["selected"]) is not int or not 0 <= value["selected"] <= 100000:
            raise ValueError("Invalid selection count")
        for key in ("started", "finished", "collection_errors", "events"):
            if type(value[key]) is not list or len(value[key]) > 100000:
                raise ValueError("Invalid event list")
        if any(
            type(case) is not str
            for case in value["started"]
            + value["finished"]
            + value["collection_errors"]
        ):
            raise ValueError("Invalid case ID")
        for event in value["events"]:
            if (
                type(event) is not dict
                or set(event) != {"case", "outcome"}
                or type(event["case"]) is not str
                or event["outcome"]
                not in (
                    "passed",
                    "failed",
                    "error",
                    "skipped",
                    "xfailed",
                    "xpassed",
                    "collection_error",
                )
            ):
                raise ValueError("Invalid test event")
        selected, executed = value["selected"], len(value["started"])
        counts = Counter(e["outcome"] for e in value["events"])
        valid = True
        if not value["complete"]:
            limitations.append("incomplete_protocol")
        if type(value["exit_code"]) is not int or value["exit_code"] != exit_code:
            limitations.append("exit_protocol_mismatch")
        if not selected or not executed:
            limitations.append("zero_tests")
        if (
            len(set(value["started"])) != executed
            or Counter(value["started"]) != Counter(value["finished"])
            or executed != selected
        ):
            limitations.append("incomplete_execution")
        event_cases = {event["case"] for event in value["events"]}
        if set(value["finished"]) - event_cases:
            limitations.append("missing_outcomes")
        if event_cases - set(value["started"]):
            limitations.append("unselected_events")
        if any(n > 1 for n in Counter(e["case"] for e in value["events"]).values()):
            limitations.append("multiple_outcomes")
        if value["collection_errors"] or counts["collection_error"]:
            limitations.append("collection_errors")
        if counts["skipped"]:
            limitations.append("skipped_cases")
        if set(expected_cases) - set(value["finished"]):
            limitations.append("missing_required_cases")
        if exit_code not in (0, 1):
            limitations.append("runner_exit")
        if exit_code == 1 and not (
            counts["failed"] or counts["error"] or counts["xpassed"]
        ):
            limitations.append("unexplained_failure_exit")
        if exit_code == 0 and (
            counts["failed"] or counts["error"] or counts["xpassed"]
        ):
            limitations.append("exit_protocol_mismatch")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        limitations.append("invalid_protocol")
    if timed_out:
        limitations.append("timeout")
    if oom:
        limitations.append("oom")
    if truncated:
        limitations.append("result_limit")
    failures = counts["failed"] + counts["error"] + counts["xpassed"]
    # Complete, valid protocol may still establish a failure under partial coverage.
    # Loader/collection failures do not establish actual assertion failures.
    status = (
        "failed"
        if valid and failures
        else "inconclusive"
        if limitations or not valid
        else "passed"
    )
    return {
        "status": status,
        "selected": selected,
        "executed": executed,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["error"],
        "skipped": counts["skipped"],
        "expected_failures": counts["xfailed"],
        "unexpected_successes": counts["xpassed"],
        "exit_code": exit_code,
        "complete": valid and not limitations,
        "limitations": sorted(set(limitations)),
    }


def project_check_tool(controller=None):
    async def execute(check_id):
        if controller is None:
            raise CodeFailure(
                "CODE_TASK_REQUIRED", "Check tool requires a code analysis task"
            )
        call = active_call.get()
        if call is None:
            raise CodeFailure("CHECK_RESULT_UNKNOWN", "Missing trusted tool identity")
        result = await controller.run(check_id, call.call_id)
        return json.dumps(result, ensure_ascii=False), result

    return StructuredTool.from_function(
        coroutine=execute,
        name="run_project_check",
        description="Execute one authorized registered check against the sealed snapshot. Only check_id is accepted; server controls command, image, paths, timeout and result classification. Failures and inconclusive results do not end analysis.",
        args_schema=ProjectCheckInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
