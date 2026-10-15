"""Durable dispatch journal. No action is replayed after an uncertain submission."""

from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from contracts.errors import ApiError
from contracts.v1 import ErrorCode
from langchain_core.tools import ToolException
from agent_service.external_operations import ExternalFailure
from agent_service.tasks import OP_TERMINAL


def public_operation(row, *, result=True):
    keys = [
        "id",
        "task_id",
        "run_id",
        "backend",
        "state",
        "deadline_at",
        "created_at",
        "finished_at",
        "error_code",
        "cleanup_state",
        "cleanup_error",
        "consumed_run_id",
    ]
    if result:
        keys.append("result")
    return {k: row[k] for k in keys}


class BackgroundStore:
    async def operations(self, task_id=None, *, raw=False):
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT * FROM agent.background_operations
                WHERE (%s::uuid IS NULL OR task_id=%s) ORDER BY created_at,id""",
                    (task_id, task_id),
                )
            ).fetchall()
        return rows if raw else [public_operation(r) for r in rows]

    async def operation(self, task_id, operation_id, *, raw=False):
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.background_operations WHERE id=%s AND task_id=%s",
                    (operation_id, task_id),
                )
            ).fetchone()
        if not row:
            raise ApiError(404, ErrorCode.NOT_FOUND, "Operation not found")
        return row if raw else public_operation(row)

    async def cancel_operation(self, task_id, operation_id):
        async with self.locked() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.background_operations WHERE id=%s AND task_id=%s",
                    (operation_id, task_id),
                )
            ).fetchone()
            if not row:
                raise ApiError(404, ErrorCode.NOT_FOUND, "Operation not found")
            if row["state"] not in OP_TERMINAL:
                await conn.execute(
                    """UPDATE agent.background_operations SET state='cancelling',
                    cancel_requested_at=COALESCE(cancel_requested_at,clock_timestamp()) WHERE id=%s""",
                    (operation_id,),
                )
                await self.task_event(
                    conn, task_id, "operation_cancelling", operation_id=operation_id
                )
        return await self.operation(task_id, operation_id)

    async def reserve_operation(
        self, row, call_id, backend, arguments_hash, binding, *, workspace_root=None
    ):
        oid, dispatch, token = uuid4(), uuid4(), uuid4()
        for attempt in range(3):
            try:
                async with self.locked() as conn:
                    live = await (
                        await conn.execute(
                            """SELECT t.deadline_at FROM agent.tasks t JOIN agent.runs r ON r.task_id=t.id
                        WHERE r.id=%s AND r.lease_token=%s AND r.status='running' AND t.status='running'
                        AND r.lease_until>clock_timestamp() AND r.deadline_at>clock_timestamp() AND t.deadline_at>clock_timestamp()
                        FOR UPDATE OF r,t""",
                            (row["id"], row["lease_token"]),
                        )
                    ).fetchone()
                    if not live:
                        raise ExternalFailure("BACKGROUND_LEASE_LOST")
                    prior = await (
                        await conn.execute(
                            "SELECT * FROM agent.background_operations WHERE run_id=%s AND call_id=%s",
                            (row["id"], call_id),
                        )
                    ).fetchone()
                    if prior:
                        if prior["dispatch_id"] != dispatch:
                            raise ExternalFailure("BACKGROUND_REPLAY_DENIED")
                        return prior
                    counts = await (
                        await conn.execute(
                            """SELECT count(*) FILTER(WHERE cleanup_state<>'released' OR state IN ('starting','running','cancelling')) AS active,
                        count(*) FILTER(WHERE task_id=%s AND (cleanup_state<>'released' OR state IN ('starting','running','cancelling'))) AS task_active,
                        count(*) FILTER(WHERE task_id=%s) AS total FROM agent.background_operations""",
                            (row["task_id"], row["task_id"]),
                        )
                    ).fetchone()
                    if (
                        counts["active"] >= self.settings.max_active_operations
                        or counts["task_active"]
                        >= self.settings.max_task_active_operations
                        or counts["total"] >= self.settings.max_task_operations
                    ):
                        raise ToolException("BACKGROUND_CAPACITY_EXCEEDED")
                    identity = (
                        await (
                            await conn.execute("SELECT current_database() AS name")
                        ).fetchone()
                    )["name"]
                    from agent_service.profiles import fingerprint

                    name = (
                        "agent-bg-" + fingerprint(identity)[:12] + "-" + oid.hex
                        if backend == "docker"
                        else None
                    )
                    record = await (
                        await conn.execute(
                            """INSERT INTO agent.background_operations
                        (id,task_id,run_id,call_id,dispatch_id,backend,arguments_hash,binding,state,workspace_root,container_name,deadline_at,monitor_token,monitor_until)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'starting',%s,%s,
                        LEAST(%s,clock_timestamp()+%s*interval '1 second'),%s,clock_timestamp()+%s*interval '1 second') RETURNING *""",
                            (
                                oid,
                                row["task_id"],
                                row["id"],
                                call_id,
                                dispatch,
                                backend,
                                arguments_hash,
                                Jsonb(binding),
                                workspace_root,
                                name,
                                live["deadline_at"],
                                self.settings.background_timeout,
                                token,
                                self.settings.reconciliation_seconds,
                            ),
                        )
                    ).fetchone()
                    await self.task_event(
                        conn, row["task_id"], "operation_starting", operation_id=oid
                    )
                return record
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise ExternalFailure("BACKGROUND_DISPATCH_UNCONFIRMED") from None

    async def operation_update(self, operation, **changes):
        allowed = {
            "state",
            "phase",
            "container_id",
            "daemon_id",
            "remote_handle",
            "receipt",
            "result",
            "error_code",
            "cleanup_state",
            "cleanup_error",
            "cancel_sent_at",
            "cancel_requested_at",
        }
        if not changes or changes.keys() - allowed:
            raise ValueError("Invalid operation update")
        assignments = sql.SQL(",").join(
            sql.SQL(
                "state=CASE WHEN cancel_requested_at IS NOT NULL AND %s IN ('starting','running') THEN 'cancelling' ELSE %s END"
            )
            if k == "state"
            else sql.SQL("{}=%s").format(sql.Identifier(k))
            for k in changes
        )
        values = []
        for key, value in changes.items():
            values.append(
                Jsonb(value)
                if key in ("receipt", "result") and value is not None
                else value
            )
            if key == "state":
                values.append(value)
        for attempt in range(3):
            try:
                async with self.locked() as conn:
                    prior = await (
                        await conn.execute(
                            "SELECT state,cleanup_state FROM agent.background_operations WHERE id=%s",
                            (operation["id"],),
                        )
                    ).fetchone()
                    record = await (
                        await conn.execute(
                            sql.SQL("""UPDATE agent.background_operations SET {},
                        finished_at=CASE WHEN %s IN ('succeeded','failed','cancelled','unknown') THEN COALESCE(finished_at,clock_timestamp()) ELSE finished_at END
                        WHERE id=%s AND monitor_token=%s AND monitor_until>clock_timestamp() RETURNING *""").format(
                                assignments
                            ),
                            (
                                *values,
                                changes.get("state"),
                                operation["id"],
                                operation["monitor_token"],
                            ),
                        )
                    ).fetchone()
                    if not record:
                        raise ExternalFailure("BACKGROUND_MONITOR_LEASE_LOST")
                    if "state" in changes and prior["state"] != record["state"]:
                        await self.task_event(
                            conn,
                            record["task_id"],
                            "operation_" + record["state"],
                            record["error_code"],
                            record["id"],
                        )
                    if prior["cleanup_state"] != record["cleanup_state"]:
                        await self.task_event(
                            conn,
                            record["task_id"],
                            "operation_cleanup_" + record["cleanup_state"],
                            record["error_code"],
                            record["id"],
                        )
                operation.update(record)
                return record
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise

    async def release_monitor(self, op, delay=None):
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE agent.background_operations SET monitor_until=NULL,monitor_token=NULL,
                next_poll_at=clock_timestamp()+%s*interval '1 second' WHERE id=%s AND monitor_token=%s""",
                (
                    self.settings.background_poll_seconds if delay is None else delay,
                    op["id"],
                    op["monitor_token"],
                ),
            )

    async def claim_operations(self):
        async with self.locked() as conn:
            rows = await (
                await conn.execute("""SELECT * FROM agent.background_operations
                WHERE (cleanup_state<>'released' OR state IN ('starting','running','cancelling'))
                AND (monitor_until IS NULL OR monitor_until<=clock_timestamp()) AND next_poll_at<=clock_timestamp()
                ORDER BY next_poll_at,id""")
            ).fetchall()
            result = []
            for row in rows:
                token = uuid4()
                result.append(
                    await (
                        await conn.execute(
                            """UPDATE agent.background_operations SET monitor_token=%s,
                    monitor_until=clock_timestamp()+%s*interval '1 second' WHERE id=%s RETURNING *""",
                            (token, self.settings.reconciliation_seconds, row["id"]),
                        )
                    ).fetchone()
                )
        return result

    async def workspace_background(self, task_id, root=None):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT id,state,cleanup_state FROM agent.background_operations
                WHERE task_id=%s AND backend='docker' AND cleanup_state<>'released' """,
                    (task_id,),
                )
            ).fetchall()
