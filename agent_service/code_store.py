"""Lease-bound persistence for isolated analysis. Admission uses the Task transaction."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from agent_service.code_artifacts import ArtifactStorage, CodeFailure
from agent_service.profiles import fingerprint
from contracts.code import CodeTaskCreate
from contracts.review import CodeReviewCreate
from agent_service.review_store import ReviewStore
from agent_service.repair_store import RepairStore
from contracts.repair import CodeRepairCreate
from contracts.errors import ApiError
from contracts.v1 import ErrorCode


def public_artifact(row):
    return {k: v for k, v in row.items() if k != "storage_key"}


class CodeStore(RepairStore, ReviewStore):
    @property
    def artifact_storage(self):
        return ArtifactStorage(
            next(
                (
                    r.settings.storage_root
                    for r in (
                        self.registry.code,
                        self.registry.review,
                        self.registry.repair,
                    )
                    if r.settings.enabled
                ),
                None,
            )
        )

    async def create_code_task(self, body, key=None):
        if not isinstance(body, CodeTaskCreate):
            body = (
                CodeRepairCreate
                if body.get("stage") == "repair"
                else CodeReviewCreate
                if body.get("stage") == "review"
                else CodeTaskCreate
            ).model_validate(body)
        registry = (
            self.registry.repair
            if body.stage == "repair"
            else self.registry.review
            if body.stage == "review"
            else self.registry.code
        )
        if not registry.settings.enabled:
            raise ApiError(
                422, ErrorCode.INVALID_REQUEST, "Requested code stage is disabled"
            )
        try:
            suite, checks = registry.select(body.workspace_id, body.check_suite_id)
            if body.stage == "repair":
                acceptance = registry.select_acceptance(
                    body.workspace_id, body.acceptance_suite_id
                )
                registry.validate_budget(
                    checks,
                    acceptance=acceptance,
                    run_seconds=min(
                        self.settings.run_timeout, self.settings.task_timeout
                    ),
                )
        except ValueError as exc:
            raise ApiError(422, ErrorCode.INVALID_REQUEST, str(exc)) from exc
        return await self.create_task(
            body.task,
            key,
            profile=registry.settings.profile,
            workspace_id=body.workspace_id,
            depends_on=body.context_task_ids,
            _code={
                "request": body.model_dump(mode="json"),
                "suite": suite,
                "checks": checks,
            },
        )

    async def code_task(self, task_id):
        task = await self.task(task_id)
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_tasks WHERE task_id=%s", (task_id,)
                )
            ).fetchone()
            if not row:
                raise ApiError(404, ErrorCode.NOT_FOUND, "Code task not found")
            snap = await (
                await conn.execute(
                    "SELECT * FROM agent.code_snapshots WHERE id=%s",
                    (row["snapshot_id"],),
                )
            ).fetchone()
            counts = await (
                await conn.execute(
                    "SELECT status,count(*) AS count FROM agent.code_check_runs WHERE task_id=%s GROUP BY status",
                    (task_id,),
                )
            ).fetchall()
        review = None
        reproduction_counts = []
        summary = {}
        if row["stage"] == "review":
            async with self.pool.connection() as conn:
                review = await (
                    await conn.execute(
                        "SELECT metadata FROM agent.code_review_ranges WHERE task_id=%s ORDER BY created_at DESC LIMIT 1",
                        (task_id,),
                    )
                ).fetchone()
                reproduction_counts = await (
                    await conn.execute(
                        "SELECT status,count(*) AS count FROM agent.code_review_reproductions WHERE task_id=%s GROUP BY status",
                        (task_id,),
                    )
                ).fetchall()
            if row["report_artifact_id"]:
                try:
                    _, data = await self.code_artifact(
                        task_id, row["report_artifact_id"]
                    )
                    authority = json.loads(data)["authority"]
                    summary = {
                        k: authority[k]
                        for k in ("finding_count", "coverage", "execution_notice")
                    }
                except ApiError as exc:
                    if exc.status != 410:
                        raise
        if row["stage"] == "repair":
            detail = await self.repair_candidate(task_id)
            summary.update(detail)
            summary["candidate_revision_id"] = (
                detail["candidate"]["current_revision_id"]
                if detail["candidate"]
                else None
            )
            summary["budget_usage"] = (
                detail["candidate"]["usage"] if detail["candidate"] else {}
            )
        return {
            **summary,
            "resolved_range": review["metadata"] if review else None,
            "reproductions": {r["status"]: r["count"] for r in reproduction_counts},
            "task_id": str(task_id),
            "status": task["status"],
            "stage": row["stage"],
            "phase": row["phase"]
            if task["status"] not in ("failed", "cancelled")
            else task["status"],
            "target": row["request"]["target"],
            "check_suite_id": row["check_suite_id"],
            "check_ids": row["check_ids"],
            "checks": {r["status"]: r["count"] for r in counts},
            "snapshot": {k: v for k, v in snap.items() if k != "storage_key"}
            if snap
            else None,
            "report_artifact_id": row["report_artifact_id"],
            "markdown_artifact_id": row["markdown_artifact_id"],
            "diagnostic_artifact_id": row["diagnostic_artifact_id"],
            "error_code": task["error_code"],
            "unresolved": task["unresolved"],
        }

    async def code_metadata(self, task_id):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM agent.code_tasks WHERE task_id=%s", (task_id,)
                )
            ).fetchone()

    async def code_phase(self, row, phase):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                "UPDATE agent.code_tasks SET phase=%s WHERE task_id=%s",
                (phase, row["task_id"]),
            )

    async def code_live(self, conn, row):
        live = await (
            await conn.execute(
                """SELECT 1 FROM agent.runs WHERE id=%s AND lease_token=%s
            AND status='running' AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp()""",
                (row["id"], row["lease_token"]),
            )
        ).fetchone()
        if not live:
            raise CodeFailure(
                "LEASE_EXPIRED", "Code execution lease is no longer valid"
            )

    async def code_page(self, task_id, kind, limit=30, cursor=None):
        await self.code_task(task_id)
        table = {
            "checks": "code_check_runs",
            "artifacts": "code_artifacts",
            "reproductions": "code_review_reproductions",
            "repair_attempts": "code_repair_attempts",
        }[kind]
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    sql.SQL("""SELECT * FROM agent.{} WHERE task_id=%s
                AND (%s::uuid IS NULL OR id<%s) ORDER BY id DESC LIMIT %s""").format(
                        sql.Identifier(table)
                    ),
                    (task_id, cursor, cursor, limit + 1),
                )
            ).fetchall()
        from agent_service.storage import page

        return page([public_artifact(r) for r in rows], limit)

    async def code_artifact(self, task_id, artifact_id):
        async with self.pool.connection() as conn, conn.transaction():
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_artifacts WHERE task_id=%s AND id=%s FOR SHARE",
                    (task_id, artifact_id),
                )
            ).fetchone()
            if not row:
                raise ApiError(404, ErrorCode.NOT_FOUND, "Artifact not found")
            if row["state"] != "available":
                raise ApiError(410, "ARTIFACT_EXPIRED", "Artifact content expired")
            try:
                data = self.artifact_storage.read(
                    row["storage_key"], row["sha256"], row["size_bytes"]
                )
            except CodeFailure as exc:
                raise ApiError(409, exc.code, str(exc)) from exc
            return public_artifact(row), data

    async def put_code_artifact(
        self, row, kind, data, *, metadata=None, artifact_id=None, maximum=None
    ):
        artifact_id = artifact_id or uuid4()
        maximum = maximum or self.registry.code_for(row).settings.result_bytes
        key, checksum = self.artifact_storage.write(artifact_id, data, maximum)
        for attempt in range(3):
            try:
                async with self.locked() as conn:
                    # Diagnostic artifacts can be retained after cancellation, but their Run
                    # lease identity must still match. Formal publication checks a live lease.
                    owner = await (
                        await conn.execute(
                            "SELECT 1 FROM agent.runs WHERE id=%s AND lease_token=%s",
                            (row["id"], row["lease_token"]),
                        )
                    ).fetchone()
                    if not owner:
                        raise CodeFailure("LEASE_EXPIRED", "Artifact ownership lost")
                    await conn.execute(
                        """INSERT INTO agent.code_artifacts(id,task_id,run_id,snapshot_id,kind,storage_key,sha256,size_bytes,metadata)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING""",
                        (
                            artifact_id,
                            row["task_id"],
                            row["id"],
                            row.get("_snapshot_id"),
                            kind,
                            key,
                            checksum,
                            len(data),
                            Jsonb(metadata or {}),
                        ),
                    )
                    result = await (
                        await conn.execute(
                            "SELECT * FROM agent.code_artifacts WHERE id=%s",
                            (artifact_id,),
                        )
                    ).fetchone()
                    if (
                        result["run_id"] != row["id"]
                        or result["sha256"] != checksum
                        or result["metadata"] != (metadata or {})
                    ):
                        raise CodeFailure(
                            "ARTIFACT_INVALID", "Artifact registration mismatch"
                        )
                return result
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise

    async def reserve_code_snapshot(self, row, target):
        sid = uuid4()
        key = "snapshots/" + sid.hex
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                """INSERT INTO agent.code_snapshots(id,task_id,run_id,state,target,policy_fingerprint,storage_key)
                VALUES(%s,%s,%s,'preparing',%s,%s,%s)""",
                (
                    sid,
                    row["task_id"],
                    row["id"],
                    Jsonb(target),
                    row["submitted_binding"]["fingerprint"],
                    key,
                ),
            )
            await conn.execute(
                "UPDATE agent.code_tasks SET snapshot_id=%s,phase='snapshot' WHERE task_id=%s",
                (sid, row["task_id"]),
            )
        return sid, self.artifact_storage.path(key)

    async def seal_code_snapshot(self, row, manifest, root, manifest_digest):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            snap = await (
                await conn.execute(
                    """UPDATE agent.code_snapshots SET state='sealed',commit_sha=%s,
                content_digest=%s,git_digest=%s,manifest_digest=%s,resolved_at=%s
                WHERE id=%s AND run_id=%s AND state='preparing' RETURNING id""",
                    (
                        manifest["commit_sha"],
                        manifest["content_digest"],
                        manifest["git_digest"],
                        manifest_digest,
                        datetime.fromtimestamp(manifest["resolved_at"], timezone.utc),
                        row["_snapshot_id"],
                        row["id"],
                    ),
                )
            ).fetchone()
            if not snap:
                raise CodeFailure("SNAPSHOT_INVALID", "Snapshot sealing failed")
            await conn.execute(
                """UPDATE agent.execution_resources SET execution_root=%s,snapshot_id=%s
                WHERE run_id=%s AND lease_token=%s AND state='reserved' """,
                (str(root), row["_snapshot_id"], row["id"], row["lease_token"]),
            )
            await conn.execute(
                "UPDATE agent.code_tasks SET phase=stage WHERE task_id=%s",
                (row["task_id"],),
            )

    async def reserve_code_resource(
        self, conn, row, purpose, root, *, resource_id=None
    ):
        rid = resource_id or uuid4()
        database = (
            await (await conn.execute("SELECT current_database() AS name")).fetchone()
        )["name"]
        name = (
            "agent-code-"
            + fingerprint(database)[:12]
            + "-"
            + row["id"].hex
            + "-"
            + rid.hex
        )
        return await (
            await conn.execute(
                """INSERT INTO agent.code_execution_resources
            (id,run_id,lease_token,worker_id,purpose,workspace_root,execution_root,snapshot_id,container_name)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                (
                    rid,
                    row["id"],
                    row["lease_token"],
                    row["worker_id"],
                    purpose,
                    self.registry.workspaces[row["workspace_id"]],
                    str(root),
                    row.get("_snapshot_id"),
                    name,
                ),
            )
        ).fetchone()

    async def code_resource(self, resource_id):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM agent.code_execution_resources WHERE id=%s",
                    (resource_id,),
                )
            ).fetchone()

    async def code_resource_update(
        self, resource_id, token, *, require_live=False, recovered=False, **changes
    ):
        if not changes or changes.keys() - {
            "daemon_id",
            "container_id",
            "state",
            "phase",
            "cleanup_error",
        }:
            raise ValueError("Invalid code resource update")
        async with self.locked() as conn:
            resource = await (
                await conn.execute(
                    "SELECT * FROM agent.code_execution_resources WHERE id=%s AND lease_token=%s FOR UPDATE",
                    (resource_id, token),
                )
            ).fetchone()
            if not resource or resource["state"] == "released":
                return None
            if require_live:
                await self.code_live(
                    conn, {"id": resource["run_id"], "lease_token": token}
                )
                if resource["state"] not in ("reserved", "active"):
                    return None
            if any(
                resource[k] is not None and resource[k] != changes[k]
                for k in ("daemon_id", "container_id")
                if k in changes
            ):
                return None
            return await (
                await conn.execute(
                    sql.SQL("""UPDATE agent.code_execution_resources SET {},updated_at=clock_timestamp(),
                released_at=CASE WHEN %s='released' THEN clock_timestamp() ELSE released_at END WHERE id=%s RETURNING *""").format(
                        sql.SQL(",").join(
                            sql.SQL("{}=%s").format(sql.Identifier(k)) for k in changes
                        )
                    ),
                    (*changes.values(), changes.get("state"), resource_id),
                )
            ).fetchone()

    async def code_resources(self, *, run_id=None, orphans=False):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT e.* FROM agent.code_execution_resources e JOIN agent.runs r ON r.id=e.run_id
                WHERE e.state<>'released' AND (%s::uuid IS NULL OR e.run_id=%s)
                AND (NOT %s OR r.status IN ('completed','failed','cancelled') OR r.lease_until<=clock_timestamp()) ORDER BY e.created_at""",
                    (run_id, run_id, orphans),
                )
            ).fetchall()

    async def reserve_code_check(self, row, call_id, check, root):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await self.execution_identity(conn, row, call_id, "check")
            prior = await (
                await conn.execute(
                    "SELECT * FROM agent.code_check_runs WHERE run_id=%s AND tool_call_id=%s",
                    (row["id"], call_id),
                )
            ).fetchone()
            if prior:
                return prior, False
            count = await self.execution_count(conn, row)
            if count >= self.registry.code_for(row).settings.check_calls:
                return None, False
            resource = await self.reserve_code_resource(conn, row, "check", root)
            result = await (
                await conn.execute(
                    """INSERT INTO agent.code_check_runs
                (id,task_id,run_id,snapshot_id,resource_id,tool_call_id,check_id,definition) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        uuid4(),
                        row["task_id"],
                        row["id"],
                        row["_snapshot_id"],
                        resource["id"],
                        call_id,
                        check.id,
                        Jsonb(check.manifest()),
                    ),
                )
            ).fetchone()
            return result, True

    async def finish_code_check(self, check_id, result, log_id, result_id):
        checksum = fingerprint(result)
        async with self.locked() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.code_check_runs WHERE id=%s FOR UPDATE",
                    (check_id,),
                )
            ).fetchone()
            if row["status"] != "running":
                if row["result_digest"] != checksum:
                    raise CodeFailure(
                        "CHECK_RESULT_UNKNOWN", "Stored check result differs"
                    )
                return row
            return await (
                await conn.execute(
                    """UPDATE agent.code_check_runs SET status=%s,result=%s,result_digest=%s,
                log_artifact_id=%s,result_artifact_id=%s,finished_at=clock_timestamp() WHERE id=%s RETURNING *""",
                    (
                        result["status"],
                        Jsonb(result),
                        checksum,
                        log_id,
                        result_id,
                        check_id,
                    ),
                )
            ).fetchone()

    async def stage_code_report(self, row, report, markdown, authority):
        if row.get("_code", {}).get("stage") == "review":
            coverage = row["_code_scope"].evidence.coverage(report)
            await self.put_code_artifact(
                row,
                "review_coverage",
                json.dumps(coverage, ensure_ascii=False, sort_keys=True).encode(),
                metadata={
                    "coverage_digest": fingerprint(coverage),
                    "range_id": authority["range"]["range_id"],
                },
                maximum=self.registry.review.settings.diff_bytes * 2 + 65536,
            )
        report_id, markdown_id = uuid4(), uuid4()
        authority = {
            **authority,
            "report_artifact_id": str(report_id),
            "markdown_artifact_id": str(markdown_id),
        }
        structured = await self.put_code_artifact(
            row,
            "repair_report"
            if row.get("_code", {}).get("stage") == "repair"
            else "review_report"
            if row.get("_code", {}).get("stage") == "review"
            else "analysis_report",
            json.dumps(
                {
                    **(
                        {
                            "version": 1,
                            "delivery_mode": "candidate_only",
                            "delivery_status": "not_applied",
                        }
                        if row.get("_code", {}).get("stage") == "repair"
                        else {}
                    ),
                    "report": report,
                    "authority": authority,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode(),
            artifact_id=report_id,
            maximum=self.registry.code_for(row).settings.report_bytes
            + (
                131072
                if row.get("_code", {}).get("stage") == "repair"
                else 65536
                if row.get("_code", {}).get("stage") == "review"
                else 32768
            ),
        )
        rendered = await self.put_code_artifact(
            row,
            "repair_markdown"
            if row.get("_code", {}).get("stage") == "repair"
            else "review_markdown"
            if row.get("_code", {}).get("stage") == "review"
            else "analysis_markdown",
            markdown.encode(),
            artifact_id=markdown_id,
            maximum=self.registry.code_for(row).settings.report_bytes * 3,
        )
        return {"report": str(structured["id"]), "markdown": str(rendered["id"])}

    async def publish_code_report(self, conn, run, artifact_ids):
        if run["profile"] == "code-repair-isolated":
            return await self.publish_repair_report(conn, run, artifact_ids)
        if run["profile"] == "code-review-isolated":
            try:
                return await self.publish_review_report(conn, run, artifact_ids)
            except (ValueError, KeyError, TypeError) as exc:
                raise CodeFailure(
                    "REPORT_INVALID", "Persisted Review report or evidence is invalid"
                ) from exc
        # Called inside the same transaction as the successful Run completion.
        rows = await (
            await conn.execute(
                """SELECT * FROM agent.code_artifacts WHERE run_id=%s AND task_id=%s
            AND id=ANY(%s::uuid[]) AND state='available' FOR UPDATE""",
                (run["id"], run["task_id"], list(artifact_ids.values())),
            )
        ).fetchall()
        if len(rows) != 2 or {r["kind"] for r in rows} != {
            "analysis_report",
            "analysis_markdown",
        }:
            raise CodeFailure("REPORT_INVALID", "Staged report artifacts unavailable")
        by_id = {str(a["id"]): a for a in rows}
        if (
            by_id[artifact_ids["report"]]["kind"] != "analysis_report"
            or by_id[artifact_ids["markdown"]]["kind"] != "analysis_markdown"
        ):
            raise CodeFailure("REPORT_INVALID", "Formal artifact kinds do not match")
        contents = {
            str(a["id"]): self.artifact_storage.read(
                a["storage_key"], a["sha256"], a["size_bytes"]
            )
            for a in rows
        }
        document = json.loads(contents[artifact_ids["report"]])
        authority = document["authority"]
        snapshot = await (
            await conn.execute(
                "SELECT * FROM agent.code_snapshots WHERE id=%s AND run_id=%s AND state='sealed' FOR SHARE",
                (authority["snapshot_id"], run["id"]),
            )
        ).fetchone()
        if (
            not snapshot
            or authority["task_id"] != str(run["task_id"])
            or authority["run_id"] != str(run["id"])
            or authority["commit_sha"] != snapshot["commit_sha"]
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Report authority does not match the Run snapshot"
            )
        from agent_service.code_reports import EvidenceLedger, markdown_report

        ledger = EvidenceLedger(
            self,
            {**run, "_snapshot_id": snapshot["id"]},
            {"commit_sha": snapshot["commit_sha"]},
        )
        sources = await (
            await conn.execute(
                "SELECT * FROM agent.code_artifacts WHERE run_id=%s AND kind='source_evidence' AND state='available'",
                (run["id"],),
            )
        ).fetchall()
        for source in sources:
            source_document = json.loads(
                self.artifact_storage.read(
                    source["storage_key"], source["sha256"], source["size_bytes"]
                )
            )
            if (
                source_document["metadata"] != source["metadata"]
                or source["snapshot_id"] != snapshot["id"]
            ):
                raise CodeFailure("REPORT_INVALID", "Stored read evidence differs")
            ledger.sources[str(source["id"])] = source["metadata"]
        checks = await (
            await conn.execute(
                "SELECT * FROM agent.code_check_runs WHERE run_id=%s", (run["id"],)
            )
        ).fetchall()
        expected_checks = {}
        for check in checks:
            if (
                check["status"] == "running"
                or check["snapshot_id"] != snapshot["id"]
                or check["result_digest"] != fingerprint(check["result"])
            ):
                raise CodeFailure(
                    "REPORT_INVALID", "Stored check result differs or is unfinished"
                )
            expected_checks[str(check["id"])] = {
                "id": str(check["id"]),
                "check_id": check["check_id"],
                "snapshot_id": str(check["snapshot_id"]),
                "result": check["result"],
                "result_digest": check["result_digest"],
                "log_artifact_id": str(check["log_artifact_id"])
                if check["log_artifact_id"]
                else None,
            }
        if {c["id"]: c for c in authority["checks"]} != expected_checks:
            raise CodeFailure(
                "REPORT_INVALID", "Report check authority differs from persisted checks"
            )
        ledger.checks = expected_checks
        validated = ledger.validate(document["report"])
        if (
            markdown_report(validated, authority).encode()
            != contents[artifact_ids["markdown"]]
        ):
            raise CodeFailure(
                "REPORT_INVALID", "Rendered report differs from the structured report"
            )
        await conn.execute(
            "UPDATE agent.code_artifacts SET visibility='formal' WHERE id=ANY(%s::uuid[])",
            (list(artifact_ids.values()),),
        )
        await conn.execute(
            "UPDATE agent.code_tasks SET phase='completed',report_artifact_id=%s,markdown_artifact_id=%s WHERE task_id=%s",
            (artifact_ids["report"], artifact_ids["markdown"], run["task_id"]),
        )

    async def gc_code_content(self):
        """Serialize admission/GC and retain active dependency/resource references."""
        import shutil

        enabled = [
            r.settings
            for r in (self.registry.code, self.registry.review, self.registry.repair)
            if r.settings.enabled
        ]
        if not enabled:
            return
        settings = enabled[0].model_copy(
            update={"retention_seconds": max(s.retention_seconds for s in enabled)}
        )
        async with self.locked() as conn:
            tasks = await (
                await conn.execute(
                    """SELECT t.id FROM agent.tasks t JOIN agent.code_tasks c ON c.task_id=t.id
                WHERE t.status IN ('completed','failed','cancelled') AND t.finished_at<clock_timestamp()-%s*interval '1 second'
                AND NOT EXISTS(SELECT 1 FROM agent.task_dependencies d JOIN agent.tasks dep ON dep.id=d.task_id
                    WHERE d.dependency_id=t.id AND dep.status NOT IN ('completed','failed','cancelled'))
                AND NOT EXISTS(SELECT 1 FROM agent.execution_resources e JOIN agent.runs r ON r.id=e.run_id WHERE r.task_id=t.id AND e.state<>'released')
                AND NOT EXISTS(SELECT 1 FROM agent.code_execution_resources e JOIN agent.runs r ON r.id=e.run_id WHERE r.task_id=t.id AND e.state<>'released')
                AND (EXISTS(SELECT 1 FROM agent.code_artifacts a WHERE a.task_id=t.id AND a.state<>'deleted')
                     OR EXISTS(SELECT 1 FROM agent.code_snapshots s WHERE s.task_id=t.id AND s.state<>'deleted'))
                ORDER BY t.finished_at LIMIT 20""",
                    (settings.retention_seconds,),
                )
            ).fetchall()
            for task in tasks:
                artifacts = await (
                    await conn.execute(
                        "SELECT * FROM agent.code_artifacts WHERE task_id=%s AND state<>'deleted' FOR UPDATE",
                        (task["id"],),
                    )
                ).fetchall()
                for artifact in artifacts:
                    await conn.execute(
                        "UPDATE agent.code_artifacts SET state='deleting' WHERE id=%s",
                        (artifact["id"],),
                    )
                    self.artifact_storage.path(artifact["storage_key"]).unlink(
                        missing_ok=True
                    )
                    await conn.execute(
                        "UPDATE agent.code_artifacts SET state='deleted',deleted_at=clock_timestamp() WHERE id=%s",
                        (artifact["id"],),
                    )
                revisions = await (
                    await conn.execute(
                        "SELECT storage_key FROM agent.code_repair_revisions WHERE task_id=%s",
                        (task["id"],),
                    )
                ).fetchall()
                candidates = await (
                    await conn.execute(
                        "SELECT storage_key FROM agent.code_repair_candidates WHERE task_id=%s AND status<>'deleted'",
                        (task["id"],),
                    )
                ).fetchall()
                for content in [*revisions, *candidates]:
                    path = self.artifact_storage.path(content["storage_key"])
                    if path.exists():
                        shutil.rmtree(path)
                await conn.execute(
                    "UPDATE agent.code_repair_candidates SET status='deleted',deleted_at=clock_timestamp() WHERE task_id=%s",
                    (task["id"],),
                )
                snapshots = await (
                    await conn.execute(
                        "SELECT * FROM agent.code_snapshots WHERE task_id=%s AND state<>'deleted' FOR UPDATE",
                        (task["id"],),
                    )
                ).fetchall()
                for snapshot in snapshots:
                    path = self.artifact_storage.path(snapshot["storage_key"])
                    if path.exists():
                        shutil.rmtree(path)
                    await conn.execute(
                        "UPDATE agent.code_snapshots SET state='deleted',deleted_at=clock_timestamp() WHERE id=%s",
                        (snapshot["id"],),
                    )

            preparing = await (
                await conn.execute("""SELECT
                EXISTS(SELECT 1 FROM agent.code_tasks c JOIN agent.tasks t ON t.id=c.task_id
                       WHERE t.status NOT IN ('completed','failed','cancelled')) OR
                EXISTS(SELECT 1 FROM agent.code_execution_resources WHERE state<>'released') OR
                EXISTS(SELECT 1 FROM agent.execution_resources WHERE state<>'released') AS active""")
            ).fetchone()
            if preparing["active"]:
                return
            # Reclaim bounded, aged files left before metadata registration was
            # confirmed. Existing records (including active dependencies) win.
            cutoff = datetime.now(timezone.utc).timestamp() - settings.retention_seconds
            directory = self.artifact_storage.path("artifacts/" + "0" * 32).parent
            if directory.exists():
                cutoff = (
                    datetime.now(timezone.utc).timestamp() - settings.retention_seconds
                )
                observed = 0
                for path in directory.iterdir():
                    candidate = path.name.removeprefix(".pending-")
                    if len(candidate) != 32 or any(
                        c not in "0123456789abcdef" for c in candidate
                    ):
                        continue
                    if path.is_symlink() or path.stat().st_mtime >= cutoff:
                        continue
                    observed += 1
                    registered = await (
                        await conn.execute(
                            "SELECT 1 FROM agent.code_artifacts WHERE storage_key=%s",
                            ("artifacts/" + candidate,),
                        )
                    ).fetchone()
                    if not registered:
                        path.unlink(missing_ok=True)
                    if observed >= 100:
                        break

            # Reclaim abandoned preparing directories without touching registered active content.
            for family, table in (
                ("candidates", "code_repair_candidates"),
                ("revisions", "code_repair_revisions"),
                ("snapshots", "code_snapshots"),
            ):
                directory = self.artifact_storage.path(family + "/" + "0" * 32).parent
                if not directory.exists():
                    continue
                observed = 0
                for path in directory.iterdir():
                    if len(path.name) != 32 or any(
                        c not in "0123456789abcdef" for c in path.name
                    ):
                        continue
                    if path.is_symlink() or path.stat().st_mtime >= cutoff:
                        continue
                    registered = await (
                        await conn.execute(
                            sql.SQL(
                                "SELECT 1 FROM agent.{} WHERE storage_key=%s"
                            ).format(sql.Identifier(table)),
                            (family + "/" + path.name,),
                        )
                    ).fetchone()
                    if not registered:
                        shutil.rmtree(path)
                    observed += 1
                    if observed >= 100:
                        break

    async def reconcile_code_tasks(self, conn):
        await conn.execute("""UPDATE agent.code_repair_candidates c SET status=t.status FROM agent.tasks t
            WHERE c.task_id=t.id AND t.status IN ('failed','cancelled') AND c.status NOT IN ('deleted','failed','cancelled')""")
        await conn.execute("""UPDATE agent.code_repair_attempts a SET status='inconclusive',
            feedback='{"reason":"interrupted_execution","replayed":false}'::jsonb,finished_at=clock_timestamp()
            FROM agent.runs r WHERE a.run_id=r.id AND a.status='running'
            AND (r.status IN ('failed','cancelled','completed') OR r.lease_until<=clock_timestamp())""")
        await conn.execute("""UPDATE agent.code_snapshots s SET state='failed' FROM agent.runs r
            WHERE s.run_id=r.id AND s.state='preparing' AND r.status IN ('failed','cancelled')""")
        await conn.execute("""UPDATE agent.code_tasks c SET phase=t.status FROM agent.tasks t
            WHERE c.task_id=t.id AND t.status IN ('failed','cancelled') AND c.phase<>t.status""")
