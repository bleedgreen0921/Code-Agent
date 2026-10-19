"""Head-only Python observations on the existing independent child-resource journal."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
import psycopg
from langchain_core.tools import ToolException
from agent_service.code_artifacts import CodeFailure
from agent_service.execution_errors import CleanupFailed, EnvironmentUnavailable
from agent_service.review_reports import execution_record


def classify_observation(
    raw, *, exit_code, script_digest, timed_out=False, oom=False, truncated=False
):
    limits = []
    result = {"status": "inconclusive", "details": "", "complete": False}
    try:
        value = json.loads(raw)
        if (
            type(value) is not dict
            or value.get("protocol") != 1
            or type(value.get("protocol")) is not int
            or value.get("runner_version") != "python-observation-v1"
            or value.get("script_digest") != script_digest
            or value.get("status") not in ("observed", "not_observed", "inconclusive")
            or type(value.get("complete")) is not bool
            or type(value.get("details")) is not str
            or type(value.get("limitations")) is not list
            or any(type(x) is not str for x in value["limitations"])
        ):
            raise ValueError("Invalid observation envelope")
        result.update(
            status=value["status"], details=value["details"], complete=value["complete"]
        )
        limits.extend(value["limitations"])
        if not value["complete"]:
            limits.append("incomplete_observation")
        if value["complete"] != (value["status"] != "inconclusive"):
            limits.append("invalid_observation_status")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        limits.append("invalid_protocol")
    if exit_code != 0:
        limits.append("runner_exit")
    if timed_out:
        limits.append("timeout")
    if oom:
        limits.append("oom")
    if truncated:
        limits.append("result_limit")
    if limits:
        result.update(status="inconclusive", complete=False)
    result.update(limitations=sorted(set(limits)), exit_code=exit_code)
    return result


class ReviewExecution:
    async def reproduce(self, call_id, hypothesis, script):
        if self.row["_code"]["stage"] != "review":
            raise ToolException("Review task required")
        if len(script.encode()) > 16384:
            raise ToolException("Script exceeds 16 KiB")
        from agent_service.code_execution import verify_snapshot

        async with self.check_lock:
            self.check_open()
            if self.remaining() < 1:
                raise ToolException("Insufficient remaining Review time")
            await asyncio.to_thread(
                verify_snapshot,
                self.snapshot_root,
                self.manifest,
                self.code.settings.model_dump(),
            )
            artifact = await self.store.put_code_artifact(
                self.row,
                "reproduction_script",
                script.encode(),
                metadata={
                    "hypothesis": hypothesis,
                    "runner_version": "python-observation-v1",
                    "content_digest": self.manifest["content_digest"],
                },
                maximum=16384,
            )
            record, created = await self.store.reserve_reproduction(
                self.row,
                call_id,
                hypothesis,
                artifact,
                self.snapshot_root,
                self.manifest["content_digest"],
            )
            if record is None:
                raise ToolException("Shared execution or reproduction budget exhausted")
            if not created:
                if record["result"] is None:
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN",
                        "Original reproduction requires reconciliation; it will not be replayed",
                    )
                value = execution_record(record)
                self.evidence.reproductions[value["id"]] = value
                return value
            resource = await self.store.code_resource(record["resource_id"])
            if (
                not resource
                or resource["execution_root"] != str(self.snapshot_root)
                or resource["snapshot_id"] != self.row["_snapshot_id"]
            ):
                raise CodeFailure(
                    "SNAPSHOT_INVALID", "Reproduction must mount the sealed head"
                )
            seconds = min(self.code.settings.check_seconds, self.remaining())
            deadline = min(
                self.row["effective_deadline_at"],
                datetime.now(timezone.utc) + timedelta(seconds=seconds),
            )
            scope = await self.child(
                "check", self.snapshot_root, resource=resource, deadline=deadline
            )
            raw, logs, exit_code, timed_out, oom = b"", b"", None, False, False
            settings = self.code.settings
            try:
                await scope.prepare()
                try:
                    exit_code, raw, logs = await scope.client.call(
                        "exec",
                        "-i",
                        scope.container_id,
                        "/usr/local/bin/python",
                        "-I",
                        "-B",
                        "/opt/agent/review_runner.py",
                        stdin=json.dumps({"script": script}).encode(),
                        timeout=max(
                            0.01,
                            (deadline - datetime.now(timezone.utc)).total_seconds(),
                        ),
                        maximum=max(settings.result_bytes, settings.log_bytes) + 1,
                        check=False,
                        timeout_result=True,
                    )
                    timed_out = exit_code == 124
                except TimeoutError:
                    timed_out = True
                info = await scope.client.inspect(scope.container_id)
                if info is None:
                    raise EnvironmentUnavailable(
                        "Reproduction container disappeared before reconciliation"
                    )
                oom = bool(info["State"].get("OOMKilled"))
                if not timed_out and not info["State"]["Running"] and not oom:
                    raise EnvironmentUnavailable(
                        "Reproduction environment stopped unexpectedly"
                    )
            finally:
                if not await scope.close():
                    raise CleanupFailed("Reproduction cleanup unconfirmed")
            result = classify_observation(
                raw[: settings.result_bytes],
                exit_code=exit_code,
                script_digest=hashlib.sha256(script.encode()).hexdigest(),
                timed_out=timed_out,
                oom=oom,
                truncated=len(raw) > settings.result_bytes,
            )
            result["log_truncated"] = len(logs) > settings.log_bytes
            log = await self.store.put_code_artifact(
                self.row,
                "reproduction_log",
                logs[: settings.log_bytes],
                metadata={
                    "reproduction_id": str(record["id"]),
                    "truncated": result["log_truncated"],
                },
                maximum=settings.log_bytes,
            )
            saved_result = await self.store.put_code_artifact(
                self.row,
                "reproduction_result",
                json.dumps(
                    {
                        "summary": result,
                        "protocol": raw[: settings.result_bytes].decode(
                            "utf-8", "replace"
                        ),
                    }
                ).encode(),
                maximum=settings.result_bytes * 6 + 65536,
            )
            for attempt in range(3):
                try:
                    completed = await self.store.finish_reproduction(
                        record["id"], result, log["id"], saved_result["id"]
                    )
                    break
                except (psycopg.Error, OSError, TimeoutError):
                    if attempt == 2:
                        raise
            value = execution_record(completed)
            self.evidence.reproductions[value["id"]] = value
            return value

    async def read_execution(self, execution_id, view="result", offset=0, limit=50):
        from uuid import UUID

        try:
            identity = UUID(execution_id)
        except ValueError as exc:
            raise ToolException("Expected execution UUID") from exc
        async with self.store.pool.connection() as conn:
            record = await (
                await conn.execute(
                    "SELECT * FROM agent.code_check_runs WHERE id=%s AND run_id=%s",
                    (identity, self.row["id"]),
                )
            ).fetchone()
            if not record:
                record = await (
                    await conn.execute(
                        "SELECT * FROM agent.code_review_reproductions WHERE id=%s AND run_id=%s",
                        (identity, self.row["id"]),
                    )
                ).fetchone()
        if (
            not record
            or not record["result"]
            or record["snapshot_id"] != self.row["_snapshot_id"]
        ):
            raise ToolException("Completed execution in this Run required")
        value = execution_record(record)
        if view == "log":
            if not record["log_artifact_id"]:
                raise ToolException("Execution log unavailable")
            _, data = await self.store.code_artifact(
                self.row["task_id"], record["log_artifact_id"]
            )
            records = [
                {"line": n, "text": line}
                for n, line in enumerate(
                    data.decode("utf-8", "replace").splitlines(), 1
                )
            ]
            kind = "execution_log"
        else:
            if "check_id" not in record:
                details = record["result"].get("details", "")
                records = [
                    {"summary": value},
                    *[
                        {"details": details[i : i + 2048]}
                        for i in range(0, len(details), 2048)
                    ],
                ]
                return await self.evidence.deliver(
                    "execution_detail",
                    records[offset : offset + limit],
                    offset=offset,
                    total=len(records),
                    execution_id=execution_id,
                )
            _, data = await self.store.code_artifact(
                self.row["task_id"], record["result_artifact_id"]
            )
            protocol = json.loads(data).get("protocol", "")
            try:
                events = json.loads(protocol).get("events", [])
            except ValueError:
                events = []
            records = [{"summary": value}, *events]
            kind = "execution_detail"
        return await self.evidence.deliver(
            kind,
            records[offset : offset + limit],
            offset=offset,
            total=len(records),
            execution_id=execution_id,
        )
