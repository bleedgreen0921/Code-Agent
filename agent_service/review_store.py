"""Review-specific persistence inside the existing task/lease/resource transactions."""

import json
from uuid import uuid4
from psycopg import sql
from psycopg.types.json import Jsonb
from agent_service.code_artifacts import CodeFailure
from agent_service.profiles import fingerprint
from contracts.errors import ApiError


class ReviewStore:
    async def save_review_range(self, row, document):
        data = json.dumps(document, ensure_ascii=False, sort_keys=True).encode()
        artifact = await self.put_code_artifact(
            row,
            "review_changes",
            data,
            maximum=self.registry.review.settings.diff_bytes * 24 + 4 * 1024**2,
        )
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                """INSERT INTO agent.code_review_ranges(id,task_id,run_id,snapshot_id,metadata,changes_artifact_id)
                VALUES(%s,%s,%s,%s,%s,%s)""",
                (
                    uuid4(),
                    row["task_id"],
                    row["id"],
                    row["_snapshot_id"],
                    Jsonb(document["range"]),
                    artifact["id"],
                ),
            )

    async def review_changes(self, task_id, limit=30, offset=0):
        task = await self.code_task(task_id)
        if task["stage"] != "review":
            raise ApiError(422, "INVALID_REQUEST", "Review task required")
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_review_ranges WHERE task_id=%s ORDER BY created_at DESC LIMIT 1",
                    (task_id,),
                )
            ).fetchone()
        if row is None:
            return {"items": [], "next_offset": None, "range": None}
        _, data = await self.code_artifact(task_id, row["changes_artifact_id"])
        document = json.loads(data)
        items = [
            {k: v for k, v in c.items() if k != "records"} for c in document["changes"]
        ]
        return {
            "items": items[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(items) else None,
            "range": row["metadata"],
            "limitations": document["limitations"],
        }

    async def execution_identity(self, conn, row, call_id, kind):
        other = (
            "code_check_runs" if kind == "reproduction" else "code_review_reproductions"
        )
        prior = await (
            await conn.execute(
                sql.SQL(
                    "SELECT 1 FROM agent.{} WHERE run_id=%s AND tool_call_id=%s"
                ).format(sql.Identifier(other)),
                (row["id"], call_id),
            )
        ).fetchone()
        if prior:
            raise CodeFailure(
                "CHECK_RESULT_UNKNOWN",
                "Tool call identity already belongs to another execution type",
            )

    async def execution_count(self, conn, row):
        record = await (
            await conn.execute(
                """SELECT
            (SELECT count(*) FROM agent.code_check_runs WHERE run_id=%s) +
            (SELECT count(*) FROM agent.code_review_reproductions WHERE run_id=%s) AS n""",
                (row["id"], row["id"]),
            )
        ).fetchone()
        return record["n"]

    async def reserve_reproduction(
        self, row, call_id, hypothesis, script_artifact, root, digest
    ):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await self.execution_identity(conn, row, call_id, "reproduction")
            prior = await (
                await conn.execute(
                    "SELECT * FROM agent.code_review_reproductions WHERE run_id=%s AND tool_call_id=%s",
                    (row["id"], call_id),
                )
            ).fetchone()
            if prior:
                if (
                    prior["script_digest"] != script_artifact["sha256"]
                    or prior["hypothesis"] != hypothesis
                ):
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN", "Original reproduction input differs"
                    )
                return prior, False
            n = await self.execution_count(conn, row)
            count = await (
                await conn.execute(
                    "SELECT count(*) AS n FROM agent.code_review_reproductions WHERE run_id=%s",
                    (row["id"],),
                )
            ).fetchone()
            settings = self.registry.review.settings
            if n >= settings.check_calls or count["n"] >= settings.reproduction_calls:
                return None, False
            resource = await self.reserve_code_resource(conn, row, "check", root)
            record = await (
                await conn.execute(
                    """INSERT INTO agent.code_review_reproductions
                (id,task_id,run_id,snapshot_id,resource_id,tool_call_id,hypothesis,script_artifact_id,script_digest,runner_version,content_digest)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        uuid4(),
                        row["task_id"],
                        row["id"],
                        row["_snapshot_id"],
                        resource["id"],
                        call_id,
                        hypothesis,
                        script_artifact["id"],
                        script_artifact["sha256"],
                        "python-observation-v1",
                        digest,
                    ),
                )
            ).fetchone()
            return record, True

    async def finish_reproduction(self, identity, result, log_id, result_id):
        async with self.locked() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_review_reproductions WHERE id=%s FOR UPDATE",
                    (identity,),
                )
            ).fetchone()
            if row["status"] != "running":
                if row["result_digest"] != fingerprint(result):
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN", "Stored reproduction result differs"
                    )
                return row
            return await (
                await conn.execute(
                    """UPDATE agent.code_review_reproductions SET status=%s,result=%s,result_digest=%s,
                log_artifact_id=%s,result_artifact_id=%s,finished_at=clock_timestamp() WHERE id=%s RETURNING *""",
                    (
                        result["status"],
                        Jsonb(result),
                        fingerprint(result),
                        log_id,
                        result_id,
                        identity,
                    ),
                )
            ).fetchone()

    async def publish_review_report(self, conn, run, artifact_ids):
        from agent_service.review_reports import load_ledger, markdown_review

        rows = await (
            await conn.execute(
                """SELECT * FROM agent.code_artifacts WHERE run_id=%s AND task_id=%s
            AND id=ANY(%s::uuid[]) AND state='available' FOR UPDATE""",
                (run["id"], run["task_id"], list(artifact_ids.values())),
            )
        ).fetchall()
        by_id = {str(r["id"]): r for r in rows}
        if (
            len(by_id) != 2
            or by_id.get(artifact_ids["report"], {}).get("kind") != "review_report"
            or by_id.get(artifact_ids["markdown"], {}).get("kind") != "review_markdown"
        ):
            raise CodeFailure("REPORT_INVALID", "Staged Review report unavailable")
        contents = {
            identity: self.artifact_storage.read(
                a["storage_key"], a["sha256"], a["size_bytes"]
            )
            for identity, a in by_id.items()
        }
        document = json.loads(contents[artifact_ids["report"]])
        ledger = await load_ledger(self, conn, run)
        report = ledger.validate(document["report"])
        expected = ledger.authority(report)
        coverage = ledger.coverage(report)
        coverage_rows = await (
            await conn.execute(
                "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND kind='review_coverage' AND state='available'",
                (run["id"],),
            )
        ).fetchall()
        if len(coverage_rows) != 1:
            raise CodeFailure("REPORT_INVALID", "Complete coverage artifact missing")
        saved_coverage = coverage_rows[0]
        coverage_value = json.loads(
            self.artifact_storage.read(
                saved_coverage["storage_key"],
                saved_coverage["sha256"],
                saved_coverage["size_bytes"],
            )
        )
        if coverage_value != coverage or saved_coverage["metadata"].get(
            "coverage_digest"
        ) != fingerprint(coverage):
            raise CodeFailure("REPORT_INVALID", "Persisted coverage records differ")
        authority = dict(document["authority"])
        if (
            authority.pop("report_artifact_id", None) != artifact_ids["report"]
            or authority.pop("markdown_artifact_id", None) != artifact_ids["markdown"]
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Formal Review artifact identity mismatch"
            )
        if (
            authority != expected
            or markdown_review(report, expected).encode()
            != contents[artifact_ids["markdown"]]
        ):
            raise CodeFailure(
                "REPORT_INVALID",
                "Review authority, coverage or rendering differs from persisted evidence",
            )
        await conn.execute(
            "UPDATE agent.code_artifacts SET visibility='formal' WHERE id=ANY(%s::uuid[])",
            (list(artifact_ids.values()),),
        )
        await conn.execute(
            "UPDATE agent.code_tasks SET phase='completed',report_artifact_id=%s,markdown_artifact_id=%s WHERE task_id=%s",
            (artifact_ids["report"], artifact_ids["markdown"], run["task_id"]),
        )
