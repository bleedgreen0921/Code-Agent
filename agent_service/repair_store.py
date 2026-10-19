"""Lease-bound immutable repair records and atomic candidate publication."""

import json
from uuid import uuid4

from psycopg.types.json import Jsonb

from agent_service.code_artifacts import CodeFailure
from agent_service.profiles import fingerprint
from contracts.errors import ApiError


class RepairStore:
    async def repair_candidate(self, task_id):
        metadata = await self.code_metadata(task_id)
        if not metadata:
            raise ApiError(404, "NOT_FOUND", "Code task not found")
        if metadata["stage"] != "repair":
            raise ApiError(422, "INVALID_REQUEST", "Repair task required")
        async with self.pool.connection() as conn:
            candidate = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_candidates WHERE task_id=%s ORDER BY created_at DESC LIMIT 1",
                    (task_id,),
                )
            ).fetchone()
            baseline_package = await (
                await conn.execute(
                    "SELECT id FROM agent.code_artifacts WHERE run_id=%s AND kind='repair_baseline' ORDER BY created_at LIMIT 1",
                    (candidate["run_id"] if candidate else None,),
                )
            ).fetchone()
            revisions = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_revisions WHERE candidate_id=%s ORDER BY created_at,id",
                    (candidate["id"] if candidate else None,),
                )
            ).fetchall()
        return dict(
            delivery_mode="candidate_only",
            delivery_status="not_applied",
            candidate={k: v for k, v in candidate.items() if k != "storage_key"}
            if candidate
            else None,
            revisions=[
                {k: v for k, v in r.items() if k != "storage_key"} for r in revisions
            ],
            baseline_artifact_id=baseline_package["id"] if baseline_package else None,
            diagnostic_artifact_id=metadata["diagnostic_artifact_id"],
        )

    async def repair_context_manifest(self, row, baseline_digest):
        ids = row["_code"]["request"]["context_task_ids"]
        async with self.pool.connection() as conn:
            contexts = await (
                await conn.execute(
                    """SELECT c.task_id,c.stage,c.report_artifact_id,
                c.snapshot_id,s.content_digest,a.storage_key,a.sha256,a.size_bytes
                FROM agent.code_tasks c JOIN agent.code_artifacts a ON a.id=c.report_artifact_id
                JOIN agent.code_snapshots s ON s.id=c.snapshot_id WHERE c.task_id=ANY(%s::uuid[])
                AND c.stage IN ('analysis','review') AND a.state='available' AND a.visibility='formal'""",
                    (ids,),
                )
            ).fetchall()
        result = []
        for context in contexts:
            report = json.loads(
                self.artifact_storage.read(
                    context["storage_key"], context["sha256"], context["size_bytes"]
                )
            )["report"]
            summary = report.get("summary") or report.get("scope", {}).get(
                "question", ""
            )
            result.append(
                {
                    "task_id": str(context["task_id"]),
                    "stage": context["stage"],
                    "report_artifact_id": str(context["report_artifact_id"]),
                    "snapshot_id": str(context["snapshot_id"]),
                    "content_digest": context["content_digest"],
                    "differs_from_baseline": context["content_digest"]
                    != baseline_digest,
                    "summary": summary[:512],
                    "notice": "Reconfirm every historical location, check and reproduction against current baseline.",
                }
            )
        if len(result) != len(ids):
            raise CodeFailure(
                "REPORT_INVALID", "Authorized formal context report unavailable"
            )
        return result

    async def create_repair_candidate(self, row, identity, acceptance, admitted):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            result = await (
                await conn.execute(
                    """INSERT INTO agent.code_repair_candidates
                (id,task_id,run_id,baseline_id,storage_key,acceptance,policy_digest,admitted_bytes)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        identity,
                        row["task_id"],
                        row["id"],
                        row["_snapshot_id"],
                        "candidates/" + identity.hex,
                        Jsonb(acceptance),
                        fingerprint(self.registry.repair.manifest),
                        admitted,
                    ),
                )
            ).fetchone()
            await conn.execute(
                """UPDATE agent.execution_resources SET candidate_id=%s,
                execution_root=%s WHERE run_id=%s AND state='reserved' """,
                (
                    identity,
                    str(self.artifact_storage.path(result["storage_key"]) / "source"),
                    row["id"],
                ),
            )
        return result

    async def save_repair_revision(self, row, revision):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            prior = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_revisions WHERE id=%s",
                    (revision["id"],),
                )
            ).fetchone()
            if prior:
                if any(prior[k] != v for k, v in revision.items()):
                    raise CodeFailure(
                        "ARTIFACT_INVALID", "Immutable revision registration differs"
                    )
                return prior
            candidate = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_candidates WHERE id=%s AND run_id=%s FOR UPDATE",
                    (revision["candidate_id"], row["id"]),
                )
            ).fetchone()
            if not candidate or candidate["status"] not in ("preparing", "writable"):
                raise CodeFailure("PERMISSION_DENIED", "Candidate writes are closed")
            if (
                candidate["admitted_bytes"] + revision["size_bytes"]
                > self.registry.repair.settings.storage_bytes
            ):
                raise CodeFailure(
                    "ARTIFACT_LIMIT", "Candidate content admission budget exhausted"
                )
            keys = tuple(revision)
            from psycopg import sql

            saved = await (
                await conn.execute(
                    sql.SQL(
                        "INSERT INTO agent.code_repair_revisions ({}) VALUES ({}) RETURNING *"
                    ).format(
                        sql.SQL(",").join(map(sql.Identifier, keys)),
                        sql.SQL(",").join(sql.Placeholder() for _ in keys),
                    ),
                    tuple(revision.values()),
                )
            ).fetchone()
            await conn.execute(
                """UPDATE agent.code_repair_candidates SET current_revision_id=%s,
                status='writable',admitted_bytes=admitted_bytes+%s WHERE id=%s""",
                (revision["id"], revision["size_bytes"], candidate["id"]),
            )
            return saved

    async def reserve_repair_attempt(self, row, candidate_id, message_id, draft):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            prior = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_attempts WHERE run_id=%s AND message_id=%s",
                    (row["id"], message_id),
                )
            ).fetchone()
            if prior:
                if prior["draft_digest"] != fingerprint(draft):
                    raise CodeFailure(
                        "REPORT_INVALID", "Repeated final message draft differs"
                    )
                return prior, False
            count = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM agent.code_repair_attempts WHERE run_id=%s",
                    (row["id"],),
                )
            ).fetchone()
            if count["n"] >= self.registry.repair.settings.gate_rounds:
                raise CodeFailure(
                    "REPAIR_BUDGET_EXHAUSTED", "Candidate gate round budget exhausted"
                )
            record = await (
                await conn.execute(
                    """INSERT INTO agent.code_repair_attempts
                (id,task_id,run_id,candidate_id,message_id,draft_digest,draft) VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        uuid4(),
                        row["task_id"],
                        row["id"],
                        candidate_id,
                        message_id,
                        fingerprint(draft),
                        Jsonb(draft),
                    ),
                )
            ).fetchone()
            return record, True

    async def finish_repair_attempt(
        self, row, identity, revision_id, checks, feedback, passed, usage
    ):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            attempt = await (
                await conn.execute(
                    "SELECT * FROM agent.code_repair_attempts WHERE id=%s AND run_id=%s FOR UPDATE",
                    (identity, row["id"]),
                )
            ).fetchone()
            status = "passed" if passed else "failed"
            if attempt["status"] != "running":
                if attempt["status"] != status or attempt["feedback"] != feedback:
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN",
                        "Gate completion acknowledgement differs",
                    )
                return attempt
            result = await (
                await conn.execute(
                    """UPDATE agent.code_repair_attempts SET status=%s,
                revision_id=%s,check_ids=%s,feedback=%s,finished_at=clock_timestamp() WHERE id=%s RETURNING *""",
                    (status, revision_id, Jsonb(checks), Jsonb(feedback), identity),
                )
            ).fetchone()
            await conn.execute(
                "UPDATE agent.code_repair_candidates SET usage=%s WHERE id=%s",
                (Jsonb(usage), attempt["candidate_id"]),
            )
            if passed:
                updated = await (
                    await conn.execute(
                        """UPDATE agent.code_repair_candidates SET status='passed',
                    frozen_revision_id=%s,report_digest=%s WHERE id=%s AND current_revision_id=%s
                    AND status='writable' RETURNING id""",
                        (
                            revision_id,
                            attempt["draft_digest"],
                            attempt["candidate_id"],
                            revision_id,
                        ),
                    )
                ).fetchone()
                if not updated:
                    raise CodeFailure(
                        "REPORT_INVALID", "Gate version no longer current"
                    )
            return result

    async def reserve_repair_check(
        self,
        row,
        candidate_id,
        revision,
        call_id,
        definition,
        purpose,
        acceptance_digest=None,
    ):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            prior = await (
                await conn.execute(
                    "SELECT * FROM agent.code_check_runs WHERE run_id=%s AND tool_call_id=%s",
                    (row["id"], call_id),
                )
            ).fetchone()
            if prior:
                if (
                    prior["revision_id"] != revision["id"]
                    or prior["definition"] != json.loads(json.dumps(definition))
                    or prior["purpose"] != purpose
                ):
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN", "Check identity binding differs"
                    )
                return prior, False
            count = await self.execution_count(conn, row)
            settings = self.registry.repair.settings
            reserve = len(row["_code"]["check_ids"]) + 1
            if count >= settings.check_calls:
                raise CodeFailure(
                    "REPAIR_BUDGET_EXHAUSTED", "Registered check budget exhausted"
                )
            if (
                purpose == "development"
                and not call_id.startswith("gate:")
                and count + reserve >= settings.check_calls
            ):
                from langchain_core.tools import ToolException

                raise ToolException(
                    "Remaining check quota is reserved for one required candidate gate; submit a draft."
                )
            root = self.artifact_storage.path(revision["storage_key"]) / "source"
            resource = await self.reserve_code_resource(conn, row, "check", root)
            await conn.execute(
                "UPDATE agent.code_execution_resources SET candidate_id=%s,revision_id=%s WHERE id=%s",
                (candidate_id, revision["id"], resource["id"]),
            )
            record = await (
                await conn.execute(
                    """INSERT INTO agent.code_check_runs
                (id,task_id,run_id,snapshot_id,resource_id,tool_call_id,check_id,definition,candidate_id,revision_id,purpose,acceptance_digest)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        uuid4(),
                        row["task_id"],
                        row["id"],
                        row["_snapshot_id"],
                        resource["id"],
                        call_id,
                        definition["id"],
                        Jsonb(definition),
                        candidate_id,
                        revision["id"],
                        purpose,
                        acceptance_digest,
                    ),
                )
            ).fetchone()
            return record, True

    async def record_repair_usage(self, row, usage):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                "UPDATE agent.code_repair_candidates SET usage=%s WHERE run_id=%s",
                (Jsonb(usage), row["id"]),
            )

    async def bind_repair_checkpoint(self, row, checkpoint):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            saved = await (
                await conn.execute(
                    """UPDATE agent.code_repair_candidates SET final_checkpoint=%s
                WHERE run_id=%s AND status='passed' RETURNING id""",
                    (Jsonb(checkpoint), row["id"]),
                )
            ).fetchone()
            if not saved:
                raise CodeFailure(
                    "REPORT_INVALID", "No passed candidate to bind to final checkpoint"
                )

    async def retain_repair_failure(self, row, reason):
        # This is diagnostic retention; it cannot publish a success or replay work.
        detail = await self.repair_candidate(row["task_id"])
        artifact = await self.put_code_artifact(
            row,
            "repair_diagnostic",
            json.dumps(
                dict(reason=reason, last_consistent_version=True, **detail),
                default=str,
                ensure_ascii=False,
            ).encode(),
            maximum=4 * 1024**2,
        )
        async with self.locked() as conn:
            await conn.execute(
                "UPDATE agent.code_repair_attempts SET status='inconclusive',feedback=%s,finished_at=clock_timestamp() WHERE run_id=%s AND status='running'",
                (Jsonb({"reason": reason, "replayed": False}), row["id"]),
            )
            await conn.execute(
                "UPDATE agent.code_tasks SET diagnostic_artifact_id=%s WHERE task_id=%s",
                (artifact["id"], row["task_id"]),
            )
            await conn.execute(
                "UPDATE agent.code_repair_candidates SET status=CASE WHEN %s='CANCELLED' THEN 'cancelled' ELSE 'failed' END WHERE run_id=%s AND status<>'published'",
                (reason, row["id"]),
            )

    async def publish_repair_report(self, conn, run, artifact_ids):
        from agent_service.repair_reports import markdown_repair, validate_published

        candidate = await (
            await conn.execute(
                "SELECT * FROM agent.code_repair_candidates WHERE run_id=%s AND status='passed' FOR UPDATE",
                (run["id"],),
            )
        ).fetchone()
        current_run = await (
            await conn.execute(
                "SELECT final_checkpoint FROM agent.runs WHERE id=%s", (run["id"],)
            )
        ).fetchone()
        if (
            not candidate
            or candidate["final_checkpoint"] != current_run["final_checkpoint"]
            or not candidate["final_checkpoint"]
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Candidate final checkpoint association differs"
            )
        artifacts = await (
            await conn.execute(
                "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND id=ANY(%s::uuid[]) AND state='available' FOR UPDATE",
                (run["id"], list(artifact_ids.values())),
            )
        ).fetchall()
        by_id = {str(a["id"]): a for a in artifacts}
        if (
            len(by_id) != 2
            or by_id.get(artifact_ids["report"], {}).get("kind") != "repair_report"
            or by_id.get(artifact_ids["markdown"], {}).get("kind") != "repair_markdown"
        ):
            raise CodeFailure("REPORT_INVALID", "Staged repair artifacts missing")

        def read(a):
            return self.artifact_storage.read(
                a["storage_key"], a["sha256"], a["size_bytes"]
            )

        document = json.loads(read(by_id[artifact_ids["report"]]))
        if (
            document["authority"].get("report_artifact_id") != artifact_ids["report"]
            or document["authority"].get("markdown_artifact_id")
            != artifact_ids["markdown"]
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Formal repair artifact identities differ"
            )
        await validate_published(self, conn, run, candidate, document)
        if markdown_repair(document["report"], document["authority"]).encode() != read(
            by_id[artifact_ids["markdown"]]
        ):
            raise CodeFailure("REPORT_INVALID", "Repair Markdown differs")
        await conn.execute(
            "UPDATE agent.code_artifacts SET visibility='formal' WHERE run_id=%s AND state='available'",
            (run["id"],),
        )
        await conn.execute(
            "UPDATE agent.code_repair_candidates SET status='published' WHERE id=%s",
            (candidate["id"],),
        )
        await conn.execute(
            "UPDATE agent.code_tasks SET phase='completed',report_artifact_id=%s,markdown_artifact_id=%s WHERE task_id=%s",
            (artifact_ids["report"], artifact_ids["markdown"], run["task_id"]),
        )
