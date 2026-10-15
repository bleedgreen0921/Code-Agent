"""Persist dispatch once; retry only database confirmation, never remote writes."""

from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from agent_service.execution_errors import ExecutionFailure
from agent_service.output import bounded_artifact


class ExternalFailure(ExecutionFailure):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ExternalOperations:
    def __init__(self, db, row):
        self.db, self.row = db, row

    async def dispatch(self, call_id, tool, arguments_hash):
        dispatch_id = uuid4()
        for attempt in range(3):
            try:
                async with self.db.locked() as conn:
                    live = await (
                        await conn.execute(
                            """SELECT id FROM agent.runs
                        WHERE id=%s AND lease_token=%s AND status='running'
                        AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp() FOR UPDATE""",
                            (self.row["id"], self.row["lease_token"]),
                        )
                    ).fetchone()
                    if not live:
                        raise ExternalFailure("MCP_LEASE_LOST")
                    await conn.execute(
                        """INSERT INTO agent.external_operations
                        (run_id,call_id,lease_token,dispatch_id,server_id,account_id,tool_name,arguments_hash,state)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'dispatched') ON CONFLICT(run_id,call_id) DO NOTHING""",
                        (
                            self.row["id"],
                            call_id,
                            self.row["lease_token"],
                            dispatch_id,
                            tool.server.id,
                            tool.server.account_id,
                            tool.selection.name,
                            arguments_hash,
                        ),
                    )
                    record = await (
                        await conn.execute(
                            "SELECT dispatch_id,state FROM agent.external_operations WHERE run_id=%s AND call_id=%s",
                            (self.row["id"], call_id),
                        )
                    ).fetchone()
                    if record != {"dispatch_id": dispatch_id, "state": "dispatched"}:
                        raise ExternalFailure("MCP_REPLAY_DENIED")
                    # Only the first insert creates a timeline event; retries after
                    # commit acknowledgement loss must not duplicate observation.
                    await conn.execute(
                        """INSERT INTO agent.events(run_id,status,error_code)
                        SELECT %s,'external_operation_dispatched',NULL WHERE NOT EXISTS
                        (SELECT 1 FROM agent.events WHERE run_id=%s AND status='external_operation_dispatched'
                        AND created_at >= (SELECT dispatched_at FROM agent.external_operations WHERE run_id=%s AND call_id=%s))""",
                        (self.row["id"], self.row["id"], self.row["id"], call_id),
                    )
                return
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise ExternalFailure("MCP_DISPATCH_UNCONFIRMED") from None

    async def outcome(self, call_id, state, receipt=None, code=None):
        receipt = bounded_artifact(receipt, 8192)
        for attempt in range(3):
            try:
                async with self.db.pool.connection() as conn, conn.transaction():
                    record = await (
                        await conn.execute(
                            """UPDATE agent.external_operations
                        SET state=%s,receipt=%s,error_code=%s,finished_at=clock_timestamp()
                        WHERE run_id=%s AND call_id=%s AND lease_token=%s
                        AND (state='dispatched' OR (state='unknown' AND %s IN ('succeeded','error')))
                        RETURNING state""",
                            (
                                state,
                                Jsonb(receipt),
                                code,
                                self.row["id"],
                                call_id,
                                self.row["lease_token"],
                                state,
                            ),
                        )
                    ).fetchone()
                    if record:
                        await self.db.event(
                            conn, self.row["id"], "external_operation_" + state, code
                        )
                    else:
                        record = await (
                            await conn.execute(
                                "SELECT state FROM agent.external_operations WHERE run_id=%s AND call_id=%s",
                                (self.row["id"], call_id),
                            )
                        ).fetchone()
                        if not record or record["state"] != state:
                            raise ExternalFailure("MCP_RESULT_UNKNOWN")
                return
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise ExternalFailure("MCP_RESULT_UNCONFIRMED") from None
