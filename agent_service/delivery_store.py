"""Persistent automatic delivery authority. Models have no entry to these methods."""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from agent_service.code_artifacts import CodeFailure
from agent_service.profiles import fingerprint
from contracts.errors import ApiError


class DeliveryStore:
    async def delivery_for_run(self, run_id, conn=None):
        if conn is None:
            async with self.pool.connection() as connection:
                return await self.delivery_for_run(run_id, connection)
        return await (
            await conn.execute(
                "SELECT * FROM agent.code_deliveries WHERE run_id=%s", (run_id,)
            )
        ).fetchone()

    async def delivery_detail(self, task_id):
        metadata = await self.code_metadata(task_id)
        if not metadata:
            raise ApiError(404, "NOT_FOUND", "Code task not found")
        async with self.pool.connection() as conn:
            delivery = await (
                await conn.execute(
                    "SELECT * FROM agent.code_deliveries WHERE task_id=%s", (task_id,)
                )
            ).fetchone()
            if delivery is None:
                return dict(
                    delivery_mode=metadata["request"].get("delivery_mode"),
                    delivery=None,
                )
            files = await (
                await conn.execute(
                    "SELECT * FROM agent.code_delivery_files WHERE delivery_id=%s ORDER BY plan_id,ordinal",
                    (delivery["id"],),
                )
            ).fetchall()
            observations = await (
                await conn.execute(
                    "SELECT id,document,created_at FROM agent.code_workspace_observations WHERE delivery_id=%s ORDER BY created_at",
                    (delivery["id"],),
                )
            ).fetchall()
            integrations = await (
                await conn.execute(
                    "SELECT * FROM agent.code_integrations WHERE delivery_id=%s ORDER BY created_at",
                    (delivery["id"],),
                )
            ).fetchall()
            occupancy = await (
                await conn.execute(
                    "SELECT * FROM agent.code_workspace_occupancy WHERE delivery_id=%s",
                    (delivery["id"],),
                )
            ).fetchone()
        public = {
            k: v
            for k, v in delivery.items()
            if k not in ("lease_token", "lease_until", "workspace_root")
        }
        return dict(
            delivery_mode="auto_apply",
            delivery=public,
            files=files,
            observations=observations,
            integrations=integrations,
            occupancy=occupancy,
        )

    async def delivery_events(self, task_id, limit=100, cursor=None):
        if not await self.code_metadata(task_id):
            raise ApiError(404, "NOT_FOUND", "Code task not found")
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT e.* FROM agent.code_delivery_events e
                JOIN agent.code_deliveries d ON d.id=e.delivery_id WHERE d.task_id=%s
                AND (%s::uuid IS NULL OR (e.created_at,e.id)>(SELECT c.created_at,c.id FROM agent.code_delivery_events c WHERE c.id=%s AND c.delivery_id=d.id)) ORDER BY e.created_at,e.id LIMIT %s""",
                    (task_id, cursor, cursor, limit + 1),
                )
            ).fetchall()
        return dict(
            items=rows[:limit],
            next_cursor=str(rows[limit - 1]["id"]) if len(rows) > limit else None,
        )

    async def create_delivery(self, row, root_identity):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            root = self.registry.workspaces[row["workspace_id"]]
            if (
                row["workspace_id"]
                not in self.registry.repair.settings.auto_apply_workspaces
            ):
                raise CodeFailure(
                    "PERMISSION_DENIED", "Automatic delivery is not authorized"
                )
            delivery = await (
                await conn.execute(
                    """INSERT INTO agent.code_deliveries
                (id,task_id,run_id,workspace_root,root_identity) VALUES(%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        uuid4(),
                        row["task_id"],
                        row["id"],
                        str(root),
                        Jsonb(root_identity),
                    ),
                )
            ).fetchone()
            await conn.execute(
                """INSERT INTO agent.code_workspace_occupancy(delivery_id,run_id,task_id,workspace_root)
                VALUES(%s,%s,%s,%s)""",
                (delivery["id"], row["id"], row["task_id"], str(root)),
            )
            return delivery

    async def update_delivery(self, row, **changes):
        allowed = {
            "phase",
            "observation_id",
            "integration_id",
            "integrated_observation_id",
            "protected",
            "diagnostic",
            "checkpoint",
            "final_checkpoint",
            "publication",
            "handoffs",
        }
        if not changes or changes.keys() - allowed:
            raise ValueError("Invalid delivery update")
        async with self.locked() as conn:
            await self.code_live(conn, row)
            assignments = sql.SQL(",").join(
                sql.SQL("{}=%s").format(sql.Identifier(k)) for k in changes
            )
            values = [
                Jsonb(v) if isinstance(v, (dict, list)) else v for v in changes.values()
            ]
            return await (
                await conn.execute(
                    sql.SQL(
                        "UPDATE agent.code_deliveries SET {},updated_at=clock_timestamp() WHERE run_id=%s RETURNING *"
                    ).format(assignments),
                    (*values, row["id"]),
                )
            ).fetchone()

    async def reserve_delivery_bytes(self, row, amount):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            limit = self.registry.repair.settings.storage_bytes
            result = await (
                await conn.execute(
                    """UPDATE agent.code_deliveries d SET admitted_bytes=admitted_bytes+%s
                WHERE run_id=%s AND admitted_bytes+%s+(SELECT admitted_bytes FROM agent.code_repair_candidates c WHERE c.run_id=d.run_id)<=%s RETURNING id""",
                    (amount, row["id"], amount, limit),
                )
            ).fetchone()
            if not result:
                raise CodeFailure(
                    "ARTIFACT_LIMIT", "Automatic delivery storage budget exhausted"
                )

    async def settle_delivery_bytes(self, row, reserved, actual):
        if not 0 <= actual <= reserved:
            raise CodeFailure("ARTIFACT_LIMIT", "Managed delivery exceeds admission")
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                "UPDATE agent.code_deliveries SET admitted_bytes=admitted_bytes-%s+%s WHERE run_id=%s",
                (reserved, actual, row["id"]),
            )

    async def save_observation(self, row, identity, key, document, size, refresh):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            delivery = await self.delivery_for_run(row["id"], conn)
            if refresh and delivery["refreshes"] >= 3:
                raise CodeFailure(
                    "DELIVERY_REFRESH_EXHAUSTED",
                    "Original workspace refresh limit exhausted",
                )
            await conn.execute(
                """INSERT INTO agent.code_workspace_observations
                (id,delivery_id,run_id,task_id,storage_key,document,size_bytes) VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                (
                    identity,
                    delivery["id"],
                    row["id"],
                    row["task_id"],
                    key,
                    Jsonb(document),
                    size,
                ),
            )
            await conn.execute(
                """UPDATE agent.code_deliveries SET observation_id=%s,refreshes=refreshes+%s WHERE id=%s""",
                (identity, int(refresh), delivery["id"]),
            )

    async def save_integration(self, row, record):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                """INSERT INTO agent.code_integrations
                (id,delivery_id,run_id,task_id,base_observation_id,observation_id,candidate_revision_id,status,evidence)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    record["id"],
                    record["delivery_id"],
                    row["id"],
                    row["task_id"],
                    record["base_observation_id"],
                    record["observation_id"],
                    record["candidate_revision_id"],
                    record["status"],
                    Jsonb(record["evidence"]),
                ),
            )
            await conn.execute(
                "UPDATE agent.code_deliveries SET integration_id=%s WHERE id=%s",
                (record["id"], record["delivery_id"]),
            )

    async def finish_integration(
        self, row, identity, status, revision_id=None, checks=(), evidence=None
    ):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                """UPDATE agent.code_integrations SET status=%s,final_revision_id=COALESCE(%s,final_revision_id),
                check_ids=%s,evidence=evidence||%s WHERE id=%s AND run_id=%s""",
                (
                    status,
                    revision_id,
                    Jsonb(list(checks)),
                    Jsonb(evidence or {}),
                    identity,
                    row["id"],
                ),
            )

    async def reopen_repair(self, row):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            await conn.execute(
                "UPDATE agent.code_repair_candidates SET status='writable',frozen_revision_id=NULL,report_digest=NULL WHERE run_id=%s AND status='passed'",
                (row["id"],),
            )
            await conn.execute(
                "UPDATE agent.runs SET status='running' WHERE id=%s AND status='delivering'",
                (row["id"],),
            )
            await self.task_transition(conn, row["task_id"], "running")

    async def rebuild_repair_resource(self, row):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            if await self.unreleased_delivery_resources(conn, row["id"]):
                raise CodeFailure(
                    "RESOURCE_CLEANUP_FAILED", "Cannot rebuild unreleased containers"
                )
            old = await (
                await conn.execute(
                    "SELECT * FROM agent.execution_resources WHERE run_id=%s AND state='released'",
                    (row["id"],),
                )
            ).fetchone()
            if old:
                delivery = await self.delivery_for_run(row["id"], conn)
                await conn.execute(
                    "INSERT INTO agent.code_delivery_events(id,delivery_id,phase,detail) VALUES(%s,%s,%s,%s)",
                    (
                        uuid4(),
                        delivery["id"],
                        "container_generation_retired",
                        Jsonb(json.loads(json.dumps(old, default=str))),
                    ),
                )
            record = await (
                await conn.execute(
                    """UPDATE agent.execution_resources SET generation=generation+1,
                container_name=container_name||'-g'||(generation+1)::text,container_id=NULL,
                state='reserved',phase='reserved',released_at=NULL,cleanup_error=NULL WHERE run_id=%s AND state='released' RETURNING *""",
                    (row["id"],),
                )
            ).fetchone()
            if not record:
                raise CodeFailure(
                    "RESOURCE_CLEANUP_FAILED", "Controlled rebuild resource unavailable"
                )
            return record

    async def consume_delivery_usage(self, row, kind):
        limits = {
            "model_calls": self.registry.repair.settings.model_calls,
            "tool_calls": self.registry.repair.settings.tool_calls,
        }
        async with self.locked() as conn:
            await self.code_live(conn, row)
            delivery = await self.delivery_for_run(row["id"], conn)
            if delivery["status"] != "pending" or delivery["phase"] in (
                "applying",
                "applied",
                "publishing",
            ):
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID",
                    "Model/tool entry is closed during delivery recovery",
                )
            usage = delivery["usage"]
            if usage.get(kind, 0) >= limits[kind]:
                raise CodeFailure(
                    "REPAIR_BUDGET_EXHAUSTED", "Persistent repair usage exhausted"
                )
            usage[kind] = usage.get(kind, 0) + 1
            await conn.execute(
                "UPDATE agent.code_deliveries SET usage=%s WHERE id=%s",
                (Jsonb(usage), delivery["id"]),
            )
            return usage

    async def delivery_preparation(self, row, phase, detail, plan_id=None):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            delivery = await self.delivery_for_run(row["id"], conn)
            await conn.execute(
                "INSERT INTO agent.code_delivery_events(id,delivery_id,plan_id,phase,detail) VALUES(%s,%s,%s,%s,%s)",
                (uuid4(), delivery["id"], plan_id, phase, Jsonb(detail)),
            )

    async def stage_delivery_plan(self, row, plan, integration_id):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            delivery = await self.delivery_for_run(row["id"], conn)
            if delivery["status"] != "pending":
                raise CodeFailure(
                    "DELIVERY_CONTROL_INVALID", "Delivery already started"
                )
            # First handoff sets the immutable forward-write deadline. Refreshes
            # and restarts never extend it.
            deadline = delivery["deadline_at"] or min(
                row["deadline_at"],
                datetime.now(UTC)
                + timedelta(seconds=self.registry.repair.settings.delivery_seconds),
            )
            await conn.execute(
                """UPDATE agent.code_deliveries SET plan=%s,plan_digest=%s,integration_id=%s,
                phase='handoff',checkpoint=NULL,deadline_at=%s WHERE id=%s""",
                (
                    Jsonb(plan),
                    fingerprint(plan),
                    integration_id,
                    deadline,
                    delivery["id"],
                ),
            )
            for number, operation in enumerate(plan["files"]):
                await conn.execute(
                    """INSERT INTO agent.code_delivery_files(delivery_id,plan_id,ordinal,path,operation)
                    VALUES(%s,%s,%s,%s,%s)""",
                    (
                        delivery["id"],
                        plan["id"],
                        number,
                        operation["path"],
                        Jsonb(operation),
                    ),
                )
            return await self.delivery_for_run(row["id"], conn)

    async def bind_delivery_handoff(self, row, checkpoint):
        async with self.locked() as conn:
            await self.code_live(conn, row)
            if await self.unreleased_delivery_resources(conn, row["id"]):
                raise CodeFailure(
                    "RESOURCE_CLEANUP_FAILED",
                    "All execution containers must be released",
                )
            await conn.execute(
                """UPDATE agent.code_deliveries SET checkpoint=%s,phase='preflight'
                WHERE run_id=%s AND status='pending' """,
                (Jsonb(checkpoint), row["id"]),
            )
            await conn.execute(
                "UPDATE agent.runs SET status='delivering' WHERE id=%s", (row["id"],)
            )
            await self.task_transition(conn, row["task_id"], "delivering")
            await conn.execute(
                "UPDATE agent.code_tasks SET phase='delivery_preflight' WHERE task_id=%s",
                (row["task_id"],),
            )

    async def unreleased_delivery_resources(self, conn, run_id):
        return await (
            await conn.execute(
                """SELECT 1 FROM agent.execution_resources WHERE run_id=%s AND state<>'released'
            UNION ALL SELECT 1 FROM agent.code_execution_resources WHERE run_id=%s AND state<>'released' LIMIT 1""",
                (run_id, run_id),
            )
        ).fetchone()

    async def acquire_delivery_lease(self, run_id, token, recovery=False):
        async with self.locked() as conn:
            record = await self.delivery_for_run(run_id, conn)
            if not record:
                raise CodeFailure("DELIVERY_CONTROL_INVALID", "Delivery is unavailable")
            # The caller must already hold flock. Expiry alone never overrules it.
            result = await (
                await conn.execute(
                    """UPDATE agent.code_deliveries SET lease_token=%s,
                lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND (lease_token=%s OR lease_until IS NULL OR lease_until<=clock_timestamp()) RETURNING *""",
                    (token, self.settings.lease_seconds, record["id"], token),
                )
            ).fetchone()
            if not result:
                raise CodeFailure(
                    "LEASE_EXPIRED", "Another delivery controller holds the lease"
                )
            return result

    async def delivery_control(self, run_id, token, forward=False):
        async with self.locked() as conn:
            delivery = await (
                await conn.execute(
                    """UPDATE agent.code_deliveries SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE run_id=%s AND lease_token=%s AND lease_until>clock_timestamp() RETURNING *""",
                    (self.settings.lease_seconds, run_id, token),
                )
            ).fetchone()
            if not delivery:
                raise CodeFailure("LEASE_EXPIRED", "Delivery lease expired")
            run = await (
                await conn.execute(
                    "SELECT r.*,t.status AS task_status,t.deadline_at AS task_deadline FROM agent.runs r JOIN agent.tasks t ON t.id=r.task_id WHERE r.id=%s",
                    (run_id,),
                )
            ).fetchone()
            if forward:
                now = datetime.now(UTC)
                if (
                    run["lease_token"] != token
                    or run["lease_until"] <= now
                    or delivery["cancel_requested"]
                    or delivery["rollback_requested"]
                    or run["status"] not in ("running", "delivering")
                    or run["task_status"] not in ("running", "delivering")
                    or min(
                        delivery["deadline_at"],
                        run["deadline_at"],
                        run["task_deadline"],
                    )
                    <= now
                ):
                    raise CodeFailure(
                        "DELIVERY_STOPPED",
                        "Cancellation, timeout or rollback prevents forward delivery",
                    )
                if await self.unreleased_delivery_resources(conn, run_id):
                    raise CodeFailure(
                        "RESOURCE_CLEANUP_FAILED",
                        "Execution resources are not released",
                    )
                if not delivery["checkpoint"] or delivery["protocol"] != 1:
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID", "Native handoff unavailable"
                    )
            await conn.execute(
                "UPDATE agent.runs SET lease_until=clock_timestamp()+%s*interval '1 second' WHERE id=%s AND lease_token=%s AND lease_until>clock_timestamp()",
                (self.settings.lease_seconds, run_id, token),
            )
            return delivery

    async def delivery_journal(
        self, run_id, token, phase, detail, ordinal=None, status=None, impact=None
    ):
        async with self.locked() as conn:
            delivery = await (
                await conn.execute(
                    """SELECT * FROM agent.code_deliveries WHERE run_id=%s
                AND lease_token=%s AND lease_until>clock_timestamp() FOR UPDATE""",
                    (run_id, token),
                )
            ).fetchone()
            if not delivery:
                raise CodeFailure(
                    "LEASE_EXPIRED", "Cannot journal after delivery lease expired"
                )
            await conn.execute(
                """INSERT INTO agent.code_delivery_events(id,delivery_id,plan_id,ordinal,phase,detail)
                VALUES(%s,%s,%s,%s,%s,%s)""",
                (
                    uuid4(),
                    delivery["id"],
                    delivery["plan"]["id"] if delivery["plan"] else None,
                    ordinal,
                    phase,
                    Jsonb(detail),
                ),
            )
            if ordinal is not None:
                await conn.execute(
                    """UPDATE agent.code_delivery_files SET phase=%s,actual=%s WHERE delivery_id=%s AND plan_id=%s AND ordinal=%s""",
                    (
                        phase,
                        Jsonb(detail),
                        delivery["id"],
                        delivery["plan"]["id"],
                        ordinal,
                    ),
                )
            await conn.execute(
                """UPDATE agent.code_deliveries SET phase=%s,status=COALESCE(%s,status),impact=COALESCE(%s,impact),
                diagnostic=%s,updated_at=clock_timestamp(),applied_at=CASE WHEN %s='applied' THEN clock_timestamp() ELSE applied_at END WHERE id=%s""",
                (phase, status, impact, Jsonb(detail), status, delivery["id"]),
            )
            await conn.execute(
                "UPDATE agent.code_tasks SET phase=%s WHERE task_id=%s",
                ("delivery_" + phase, delivery["task_id"]),
            )
            if status in ("rolled_back",):
                await conn.execute(
                    "UPDATE agent.code_workspace_occupancy SET released_at=clock_timestamp() WHERE delivery_id=%s",
                    (delivery["id"],),
                )
            return await self.delivery_for_run(run_id, conn)

    async def recoverable_deliveries(self):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute("""SELECT d.*,r.workspace_id FROM agent.code_deliveries d JOIN agent.runs r ON r.id=d.run_id
                WHERE (d.status IN ('pending','applying') OR (d.status='applied' AND d.phase<>'published'))
                AND (r.status IN ('failed','cancelled','completed') OR r.lease_until<=clock_timestamp())
                AND (d.lease_until IS NULL OR d.lease_until<=clock_timestamp()) ORDER BY d.created_at LIMIT 20""")
            ).fetchall()

    async def claim_delivery_recovery(self, run_id, worker_id):
        """Called only under the workspace flock. Never creates a Run or deadline."""
        async with self.locked() as conn:
            delivery = await self.delivery_for_run(run_id, conn)
            run = await (
                await conn.execute(
                    "SELECT * FROM agent.runs WHERE id=%s FOR UPDATE", (run_id,)
                )
            ).fetchone()
            live = await (
                await conn.execute(
                    "SELECT %s::timestamptz>clock_timestamp() AS live",
                    (run["lease_until"],),
                )
            ).fetchone()
            if live["live"] and run["status"] in (
                "running",
                "delivering",
                "cancelling",
            ):
                return None
            token = uuid4()
            leased = await (
                await conn.execute(
                    """UPDATE agent.code_deliveries SET lease_token=%s,lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND (lease_until IS NULL OR lease_until<=clock_timestamp()) RETURNING *""",
                    (token, self.settings.lease_seconds, delivery["id"]),
                )
            ).fetchone()
            if not leased:
                return None
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET lease_token=%s,worker_id=%s,lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s RETURNING *""",
                    (token, worker_id, self.settings.lease_seconds, run_id),
                )
            ).fetchone()
            await conn.execute(
                "INSERT INTO agent.code_delivery_events(id,delivery_id,plan_id,phase,detail) VALUES(%s,%s,%s,%s,%s)",
                (
                    uuid4(),
                    delivery["id"],
                    delivery["plan"]["id"] if delivery["plan"] else None,
                    "recovery_claimed",
                    Jsonb(
                        {
                            "previous_status": delivery["status"],
                            "protocol": delivery["protocol"],
                        }
                    ),
                ),
            )
            row["workspace_root"] = delivery["workspace_root"]
            return row

    async def finish_delivery_recovery(self, row, outcome, publication=None):
        """Only original-delivery reconciliation and publication; no agent authority."""
        async with self.locked() as conn:
            delivery = await self.delivery_for_run(row["id"], conn)
            if delivery["lease_token"] != row["lease_token"]:
                raise CodeFailure("LEASE_EXPIRED", "Recovery ownership lost")
            task = await (
                await conn.execute(
                    "SELECT * FROM agent.tasks WHERE id=%s", (row["task_id"],)
                )
            ).fetchone()
            cancelled = delivery["cancel_requested"] or task["status"] in (
                "cancelling",
                "cancelled",
            )
            if outcome["status"] == "applied":
                if not publication:
                    raise CodeFailure(
                        "DELIVERY_EVIDENCE_INVALID",
                        "Applied recovery requires final report and checkpoint",
                    )
                await conn.execute(
                    """UPDATE agent.code_repair_candidates SET status='passed',final_checkpoint=%s WHERE run_id=%s""",
                    (Jsonb(publication["checkpoint"]), row["id"]),
                )
                await conn.execute(
                    "UPDATE agent.code_deliveries SET final_checkpoint=%s,publication=%s WHERE id=%s",
                    (
                        Jsonb(publication["checkpoint"]),
                        Jsonb(publication),
                        delivery["id"],
                    ),
                )
                # Publish validates A, M, cleanup and every file confirmation within
                # this same transaction. Applied bytes are never replayed here.
                await conn.execute(
                    "UPDATE agent.runs SET final_checkpoint=%s,answer=%s,status='completed' WHERE id=%s",
                    (
                        Jsonb(publication["checkpoint"]),
                        publication["answer"],
                        row["id"],
                    ),
                )
                await self.publish_repair_report(
                    conn, row, publication["code_artifacts"]
                )
                state = "cancelled" if cancelled else "completed"
                await conn.execute(
                    """UPDATE agent.runs SET status=%s,answer=%s,final_checkpoint=%s,error_code=NULL,finished_at=clock_timestamp() WHERE id=%s""",
                    (
                        state,
                        None if cancelled else publication["answer"],
                        None if cancelled else Jsonb(publication["checkpoint"]),
                        row["id"],
                    ),
                )
                if task["status"] not in ("completed", "cancelled"):
                    await conn.execute(
                        """UPDATE agent.tasks SET status=%s,answer=%s,error_code=NULL,finished_at=clock_timestamp() WHERE id=%s""",
                        (
                            state,
                            None if cancelled else publication["answer"],
                            row["task_id"],
                        ),
                    )
                    await self.task_event(conn, row["task_id"], state)
                await self.event(conn, row["id"], state)
            elif outcome["status"] == "rolled_back":
                state = "cancelled" if cancelled else "failed"
                await conn.execute(
                    """UPDATE agent.runs SET status=%s,answer=NULL,final_checkpoint=NULL,error_code=%s,finished_at=clock_timestamp() WHERE id=%s""",
                    (state, None if cancelled else "DELIVERY_STOPPED", row["id"]),
                )
                if task["status"] not in ("completed", "failed", "cancelled"):
                    await self.task_transition(
                        conn,
                        row["task_id"],
                        state,
                        None if cancelled else "DELIVERY_STOPPED",
                    )
                await self.event(
                    conn, row["id"], state, None if cancelled else "DELIVERY_STOPPED"
                )
            await conn.execute(
                "UPDATE agent.code_deliveries SET lease_until=clock_timestamp() WHERE id=%s",
                (delivery["id"],),
            )

    async def gc_delivery_content(self, conn, task_id):
        """Called by the existing retention transaction, only after resolution."""
        import shutil
        from pathlib import Path

        delivery = await (
            await conn.execute(
                "SELECT * FROM agent.code_deliveries WHERE task_id=%s FOR UPDATE",
                (task_id,),
            )
        ).fetchone()
        if not delivery:
            return
        if delivery["status"] not in ("applied", "rolled_back") or (
            delivery["status"] == "applied" and delivery["phase"] != "published"
        ):
            raise CodeFailure(
                "DELIVERY_CONTROL_INVALID", "Unresolved delivery content cannot expire"
            )
        observations = await (
            await conn.execute(
                "SELECT storage_key FROM agent.code_workspace_observations WHERE delivery_id=%s",
                (delivery["id"],),
            )
        ).fetchall()
        integrations = await (
            await conn.execute(
                "SELECT id FROM agent.code_integrations WHERE delivery_id=%s",
                (delivery["id"],),
            )
        ).fetchall()
        for key in [o["storage_key"] for o in observations] + [
            "integrations/" + i["id"].hex for i in integrations
        ]:
            path = self.artifact_storage.path(key)
            if path.exists():
                shutil.rmtree(path)
        run = await (
            await conn.execute(
                "SELECT workspace_id FROM agent.runs WHERE id=%s", (delivery["run_id"],)
            )
        ).fetchone()
        private = self.registry.repair.settings.auto_apply_workspaces.get(
            run["workspace_id"]
        )
        if private:
            private = Path(private)
            plans = await (
                await conn.execute(
                    "SELECT DISTINCT plan_id FROM agent.code_delivery_files WHERE delivery_id=%s",
                    (delivery["id"],),
                )
            ).fetchall()
            intents = await (
                await conn.execute(
                    "SELECT DISTINCT plan_id FROM agent.code_delivery_events WHERE delivery_id=%s AND plan_id IS NOT NULL",
                    (delivery["id"],),
                )
            ).fetchall()
            names = {p["plan_id"].hex for p in [*plans, *intents]}
            names.update(h["payload"]["plan_id"] for h in delivery["handoffs"])
            if delivery["plan"]:
                names.add(delivery["plan"]["id"])
            for name in names:
                if len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID", "Invalid private retention path"
                    )
                path = private / name
                if path.is_symlink() or path.resolve().parent != private:
                    raise CodeFailure(
                        "DELIVERY_CONTROL_INVALID",
                        "Private retention directory identity changed",
                    )
                if path.exists():
                    shutil.rmtree(path)

    async def delivery_diagnostics(self):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT d.id,d.task_id,d.run_id,d.status,d.impact,d.phase,d.deadline_at,d.diagnostic,d.plan_digest,d.updated_at FROM agent.code_deliveries d JOIN agent.code_workspace_occupancy o ON o.delivery_id=d.id WHERE o.released_at IS NULL ORDER BY d.created_at"
                )
            ).fetchall()
