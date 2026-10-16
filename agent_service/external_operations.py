"""Persist dispatch once; retry only database confirmation, never remote writes."""

from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from agent_service.execution_errors import ExecutionFailure
from agent_service.output import bounded_artifact
from agent_service.failure import failure_info, validated_info


class ExternalFailure(ExecutionFailure):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ExternalOperations:
    def __init__(self, db, row):
        self.db, self.row = db, row
        self.dispatches = {}

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
                self.dispatches[call_id] = dispatch_id
                return
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise ExternalFailure("MCP_DISPATCH_UNCONFIRMED") from None

    async def outcome(self, call_id, state, receipt=None, code=None, *, info=None):
        return await self._outcome(call_id, state, receipt, code, info, reconcile=False)

    async def reconcile_original(
        self, call_id, state, receipt=None, code=None, *, info=None
    ):
        """Record late facts only; never authorize another dispatch or resume a Task."""
        return await self._outcome(call_id, state, receipt, code, info, reconcile=True)

    async def _outcome(self, call_id, state, receipt, code, info, *, reconcile):
        if state not in ("succeeded", "error", "unknown"):
            raise ValueError("Invalid external outcome")
        receipt = bounded_artifact(receipt, 8192)
        info = (
            validated_info(info)
            if info is not None
            else failure_info(code)
            if code
            else None
        )
        expected = {
            "state": state,
            "receipt": receipt,
            "error_code": code,
            "failure_info": info,
        }
        for attempt in range(3):
            try:
                async with self.db.locked() as conn:
                    prior = await (
                        await conn.execute(
                            """SELECT op.*,r.task_id, r.lease_token AS current_token,
                            (r.status IN ('running','cancelling') AND r.lease_until>clock_timestamp()
                            AND r.deadline_at>clock_timestamp()) AS live
                            FROM agent.external_operations op JOIN agent.runs r ON r.id=op.run_id
                            WHERE op.run_id=%s AND op.call_id=%s FOR UPDATE OF op,r""",
                            (self.row["id"], call_id),
                        )
                    ).fetchone()
                    if (
                        not prior
                        or prior["lease_token"] != self.row["lease_token"]
                        or prior["dispatch_id"]
                        != self.dispatches.get(call_id, prior["dispatch_id"])
                    ):
                        raise ExternalFailure("MCP_RESULT_UNKNOWN")
                    if all(prior[k] == v for k, v in expected.items()):
                        return
                    if reconcile:
                        if prior["state"] != "unknown" or state not in (
                            "succeeded",
                            "error",
                        ):
                            raise ExternalFailure("MCP_RESULT_UNKNOWN")
                        if prior["task_id"]:
                            await self.db.stop_task(
                                conn,
                                prior["task_id"],
                                code=prior["error_code"] or "MCP_RESULT_UNKNOWN",
                            )
                    elif (
                        prior["state"] != "dispatched"
                        or not prior["live"]
                        or prior["current_token"] != self.row["lease_token"]
                    ):
                        raise ExternalFailure("MCP_RESULT_UNKNOWN")
                    await conn.execute(
                        """UPDATE agent.external_operations SET state=%s,receipt=%s,error_code=%s,
                        failure_info=%s,finished_at=COALESCE(finished_at,clock_timestamp())
                        WHERE run_id=%s AND call_id=%s""",
                        (
                            state,
                            Jsonb(receipt) if receipt is not None else None,
                            code,
                            Jsonb(info) if info is not None else None,
                            self.row["id"],
                            call_id,
                        ),
                    )
                    if state == "unknown" and prior["task_id"]:
                        await self.db.stop_task(
                            conn, prior["task_id"], code=code or "MCP_RESULT_UNKNOWN"
                        )
                    await self.db.event(
                        conn, self.row["id"], "external_operation_" + state, code
                    )
                return
            except (psycopg.Error, OSError, TimeoutError):
                if attempt == 2:
                    raise ExternalFailure("MCP_RESULT_UNCONFIRMED") from None
