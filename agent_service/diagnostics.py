"""Read-only unresolved facts. Counts precede display limits in one SQL snapshot."""

from contextlib import asynccontextmanager

from agent_service.failure import SPECS


class DiagnosticStore:
    async def verification_candidate(self, run_id, delivered=(), *, conn=None):
        """One read-only snapshot; current Run resources are cleaned after verification."""
        if conn is None:
            async with self.snapshot() as snapshot:
                return await self.verification_candidate(
                    run_id, delivered, conn=snapshot
                )
        from agent_service.external_operations import ExternalFailure

        facts = await (
            await conn.execute(
                """
            SELECT r.*, t.status AS task_status, t.error_code AS task_error,
                r.lease_until>clock_timestamp() AND r.deadline_at>clock_timestamp() AS live,
                EXISTS(SELECT 1 FROM agent.external_operations o JOIN agent.runs s ON s.id=o.run_id
                    WHERE (s.id=r.id OR (r.task_id IS NOT NULL AND s.task_id=r.task_id))
                    AND o.state IN ('unknown','dispatched')) AS external_unknown,
                EXISTS(SELECT 1 FROM agent.background_operations o WHERE o.task_id=r.task_id
                    AND o.state='unknown') AS background_unknown,
                EXISTS(SELECT 1 FROM agent.execution_resources e JOIN agent.runs s ON s.id=e.run_id
                    WHERE e.run_id<>r.id AND e.state<>'released' AND
                    (s.task_id=r.task_id OR s.submitted_binding->>'workspace_identity'=r.submitted_binding->>'workspace_identity'
                     OR EXISTS(SELECT 1 FROM agent.execution_resources own WHERE own.run_id=r.id AND
                        (starts_with(e.workspace_root,own.workspace_root||'/')
                         OR starts_with(own.workspace_root,e.workspace_root||'/')
                         OR own.workspace_root='/' OR e.workspace_root='/')))) AS old_resources,
                EXISTS(SELECT 1 FROM agent.background_operations o WHERE o.task_id=r.task_id AND
                    (o.state NOT IN ('succeeded','failed','cancelled') OR o.cleanup_state<>'released'
                     OR (o.consumed_run_id IS NULL AND NOT o.id=ANY(%s::uuid[])))) AS pending
            FROM agent.runs r LEFT JOIN agent.tasks t ON t.id=r.task_id WHERE r.id=%s
        """,
                (list(delivered), run_id),
            )
        ).fetchone()
        if facts["status"] != "running" or not facts["live"]:
            raise ExternalFailure("LEASE_EXPIRED")
        if facts["task_status"] in ("failed", "cancelled", "cancelling", "completed"):
            raise ExternalFailure(facts["task_error"] or "WORKER_STOPPED")
        if facts["external_unknown"] or facts["background_unknown"]:
            raise ExternalFailure(
                "MCP_RESULT_UNKNOWN"
                if facts["external_unknown"]
                else "BACKGROUND_RESULT_UNKNOWN"
            )
        return not (facts["old_resources"] or facts["pending"])

    @asynccontextmanager
    async def snapshot(self):
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute(
                "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
            )
            yield conn

    async def unresolved(self, *, run_id=None, task_id=None, conn=None):
        if conn is None:
            async with self.snapshot() as snapshot:
                return await self.unresolved(
                    run_id=run_id, task_id=task_id, conn=snapshot
                )
        row = await (
            await conn.execute(
                """
            WITH scoped AS (
                SELECT id,task_id FROM agent.runs
                WHERE (%s::uuid IS NULL OR id=%s) AND (%s::uuid IS NULL OR task_id=%s)
            ), facts AS (
                SELECT 'background_operation' AS kind,o.id,o.run_id,r.task_id,
                       o.state AS status,o.cleanup_state,o.created_at,
                       o.state='unknown' AS unknown_operation,o.cleanup_state<>'released' AS unreleased,
                       CASE WHEN o.error_code=ANY(%s::text[]) THEN o.error_code
                            WHEN o.state='unknown' AND o.error_code IS NOT NULL THEN 'LEGACY_FAILURE'
                            WHEN o.state='unknown' THEN 'OPERATION_OUTCOME_UNKNOWN'
                            ELSE 'RESOURCE_RELEASE_UNCONFIRMED' END AS reason
                FROM agent.background_operations o JOIN scoped r ON r.id=o.run_id
                WHERE o.state='unknown' OR o.cleanup_state<>'released'
                UNION ALL
                SELECT 'external_operation',o.dispatch_id,o.run_id,r.task_id,o.state,NULL,o.dispatched_at,
                       true,false,CASE WHEN o.error_code=ANY(%s::text[]) THEN o.error_code
                                       WHEN o.error_code IS NOT NULL THEN 'LEGACY_FAILURE'
                                       ELSE 'OPERATION_OUTCOME_UNKNOWN' END
                FROM agent.external_operations o JOIN scoped r ON r.id=o.run_id WHERE o.state='unknown'
                UNION ALL
                SELECT 'execution_resource',e.run_id,e.run_id,r.task_id,e.state,NULL,e.created_at,
                       false,true,CASE WHEN e.state='cleanup_failed' THEN 'RESOURCE_CLEANUP_FAILED'
                                      ELSE 'RESOURCE_RELEASE_UNCONFIRMED' END
                FROM agent.execution_resources e JOIN scoped r ON r.id=e.run_id WHERE e.state<>'released'
            ), counts AS (
                SELECT count(*) FILTER(WHERE unknown_operation) AS unknown_operations,
                       count(*) FILTER(WHERE unreleased) AS unreleased_resources,
                       CASE WHEN min(created_at) IS NULL THEN NULL
                            ELSE GREATEST(0,EXTRACT(EPOCH FROM transaction_timestamp()-min(created_at)))::float8 END AS oldest_age_seconds,
                       count(*)>50 AS truncated FROM facts
            ), reasons AS (
                SELECT COALESCE(jsonb_object_agg(reason,n),'{}'::jsonb) AS reasons
                FROM (SELECT reason,count(*) AS n FROM facts GROUP BY reason) grouped
            ), records AS (
                SELECT COALESCE(jsonb_agg(to_jsonb(f) ORDER BY created_at,kind,id),'[]'::jsonb) AS records
                FROM (SELECT * FROM facts ORDER BY created_at,kind,id LIMIT 50) f
            ) SELECT counts.*,reasons.reasons,records.records FROM counts,reasons,records
        """,
                (run_id, run_id, task_id, task_id, list(SPECS), list(SPECS)),
            )
        ).fetchone()
        ids = [
            record["id"]
            for record in row["records"]
            if record["kind"] == "background_operation"
        ]
        if ids:
            from agent_service.background_store import operation_diagnostics

            diagnostics = await (
                await conn.execute(
                    "SELECT * FROM agent.background_operations WHERE id=ANY(%s::uuid[])",
                    (ids,),
                )
            ).fetchall()
            by_id = {
                str(item["id"]): operation_diagnostics(item) for item in diagnostics
            }
            # The same read-only repeatable-read transaction owns both projections.
            for record in row["records"]:
                if record["kind"] == "background_operation":
                    record.update(by_id[str(record["id"])])
        return row
