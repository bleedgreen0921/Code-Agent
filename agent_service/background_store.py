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
from agent_service.failure import failure_info, validated_info


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
        "failure_info",
        "cleanup_state",
        "cleanup_error",
        "consumed_run_id",
    ]
    if result:
        keys.append("result")
    return {k: row[k] for k in keys} | operation_diagnostics(row)


def operation_diagnostics(row):
    return {
        "observation": {
            k: row.get(k)
            for k in (
                "observation_failures",
                "observation_first_failed_at",
                "last_observed_at",
                "last_observation_error",
                "next_poll_at",
                "remote_poll_interval_ms",
                "remote_poll_not_before",
            )
        },
        "cancellation": {
            k: row.get(k)
            for k in (
                "cancel_requested_at",
                "cancel_sent_at",
                "cancel_attempts",
                "cancel_attempted_at",
                "cancel_acknowledged_at",
                "cancel_last_error",
            )
        },
    }


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
        from agent_service.reliability import background_policy

        binding = {
            **binding,
            "reliability": background_policy(self.registry, self.settings),
        }
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
                        from agent_service.policy import active_call

                        call = active_call.get()
                        if call:
                            call.diagnostic = failure_info(
                                "BACKGROUND_CAPACITY_EXCEEDED", effect="not_dispatched"
                            )
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
        return await self._operation_update(operation, changes, mode="ordinary")

    async def operation_reconcile(self, operation, **changes):
        """Internal authoritative observation of the ORIGINAL resource only."""
        return await self._operation_update(operation, changes, mode="reconcile")

    async def operation_cleanup(self, operation, **changes):
        if changes.keys() - {"cleanup_state", "cleanup_error"}:
            raise ValueError("Invalid cleanup update")
        return await self._operation_update(operation, changes, mode="cleanup")

    async def _operation_update(self, operation, changes, *, mode):
        allowed = {
            "state",
            "phase",
            "container_id",
            "daemon_id",
            "remote_handle",
            "receipt",
            "result",
            "error_code",
            "failure_info",
            "cleanup_state",
            "cleanup_error",
            "cancel_sent_at",
            "cancel_requested_at",
        }
        if not changes or changes.keys() - allowed:
            raise ValueError("Invalid operation update")
        changes = dict(changes)
        if changes.get("state") in OP_TERMINAL:
            # A terminal submission describes the complete execution outcome, including
            # absent details. Confirmation cannot silently accept a different result.
            changes.setdefault("result", None)
            changes.setdefault("error_code", None)
        if "failure_info" in changes:
            changes["failure_info"] = validated_info(changes["failure_info"])
        elif "error_code" in changes:
            changes["failure_info"] = (
                failure_info(
                    changes["error_code"],
                    effect="unknown" if changes.get("state") == "unknown" else None,
                )
                if changes["error_code"]
                else None
            )
        for attempt in range(3):
            try:
                async with self.locked() as conn:
                    prior = await (
                        await conn.execute(
                            """SELECT * FROM agent.background_operations WHERE id=%s
                            AND task_id=%s AND run_id=%s AND monitor_token=%s
                            AND monitor_until>clock_timestamp() FOR UPDATE""",
                            (
                                operation["id"],
                                operation["task_id"],
                                operation["run_id"],
                                operation["monitor_token"],
                            ),
                        )
                    ).fetchone()
                    if not prior:
                        raise ExternalFailure("BACKGROUND_MONITOR_LEASE_LOST")
                    updates = guarded_changes(prior, changes, mode=mode)
                    if changes.get("receipt") is not None and prior["receipt"] is None:
                        from datetime import timedelta
                        from agent_service.reliability import poll_interval

                        interval = poll_interval(
                            changes["receipt"].get("pollIntervalMs")
                        )
                        now = (
                            await (
                                await conn.execute("SELECT clock_timestamp() AS now")
                            ).fetchone()
                        )["now"]
                        updates.update(
                            remote_poll_interval_ms=interval,
                            remote_poll_not_before=now
                            + timedelta(milliseconds=interval or 0),
                        )
                    if prior["state"] == "unknown" and mode == "reconcile":
                        # No scheduler or result consumption: keep the Task terminal.
                        await self.stop_task(
                            conn,
                            prior["task_id"],
                            code="BACKGROUND_RESULT_UNKNOWN",
                        )
                    different = {k: v for k, v in updates.items() if prior[k] != v}
                    record = prior
                    if different:
                        assignments = sql.SQL(",").join(
                            sql.SQL("{}=%s").format(sql.Identifier(k))
                            for k in different
                        )
                        values = [
                            Jsonb(v)
                            if k in ("receipt", "result", "failure_info")
                            and v is not None
                            else v
                            for k, v in different.items()
                        ]
                        record = await (
                            await conn.execute(
                                sql.SQL("""UPDATE agent.background_operations SET {},
                                finished_at=CASE WHEN %s IN ('succeeded','failed','cancelled','unknown')
                                THEN COALESCE(finished_at,clock_timestamp()) ELSE finished_at END
                                WHERE id=%s RETURNING *""").format(assignments),
                                (*values, updates.get("state"), operation["id"]),
                            )
                        ).fetchone()
                    if prior["state"] != record["state"]:
                        await self.task_event(
                            conn,
                            record["task_id"],
                            "operation_" + record["state"],
                            record["error_code"],
                            record["id"],
                        )
                    if record["state"] == "unknown":
                        await self.stop_task(
                            conn,
                            record["task_id"],
                            code="BACKGROUND_RESULT_UNKNOWN",
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

    async def monitor_record(self, operation):
        async with self.pool.connection() as conn:
            current = await (
                await conn.execute(
                    """SELECT * FROM agent.background_operations WHERE id=%s AND task_id=%s
                AND run_id=%s AND monitor_token=%s AND monitor_until>clock_timestamp()""",
                    (
                        operation["id"],
                        operation["task_id"],
                        operation["run_id"],
                        operation["monitor_token"],
                    ),
                )
            ).fetchone()
        if not current:
            raise ExternalFailure("BACKGROUND_MONITOR_LEASE_LOST")
        for key in (
            "backend",
            "binding",
            "container_name",
            "container_id",
            "daemon_id",
            "remote_handle",
            "receipt",
        ):
            if operation[key] != current[key]:
                raise ExternalFailure("BACKGROUND_RESOURCE_IDENTITY_CHANGED")
        return current

    async def release_monitor(self, op, delay=None):
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE agent.background_operations SET monitor_until=NULL,monitor_token=NULL,
                next_poll_at=CASE WHEN %s::double precision IS NULL THEN next_poll_at
                ELSE LEAST(GREATEST(COALESCE(remote_poll_not_before,clock_timestamp()),
                    clock_timestamp()+%s*interval '1 second'),
                    CASE WHEN state IN ('starting','running','cancelling') THEN deadline_at ELSE 'infinity'::timestamptz END) END
                WHERE id=%s AND monitor_token=%s""",
                (delay, delay, op["id"], op["monitor_token"]),
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
                if row["observation_inflight"]:
                    from agent_service.reliability import backoff
                    from datetime import timedelta

                    from agent_service.reliability import ReliabilitySettings

                    frozen = row["binding"].get("reliability")
                    policy = (
                        ReliabilitySettings.model_validate(
                            {k: frozen[k] for k in ReliabilitySettings.model_fields}
                        )
                        if frozen
                        else self.registry.reliability
                    )
                    now = (
                        await (
                            await conn.execute("SELECT clock_timestamp() AS now")
                        ).fetchone()
                    )["now"]
                    failures = row["observation_failures"] + 1
                    first = (
                        row["observation_first_failed_at"]
                        or row["observation_started_at"]
                    )
                    delay = backoff(
                        failures,
                        policy.observation_backoff_base,
                        policy.observation_backoff_cap,
                    )
                    wakeup = max(
                        now
                        + timedelta(
                            seconds=max(delay, self.settings.background_poll_seconds)
                        ),
                        row["remote_poll_not_before"] or now,
                    )
                    if row["state"] not in OP_TERMINAL:
                        wakeup = min(
                            wakeup, first + timedelta(seconds=policy.observation_window)
                        )
                    await conn.execute(
                        """UPDATE agent.background_operations SET observation_inflight=false,
                        observation_failures=%s,observation_first_failed_at=%s,last_observation_error=%s,
                        next_poll_at=%s WHERE id=%s""",
                        (
                            failures,
                            first,
                            Jsonb(failure_info("BACKGROUND_OBSERVATION_INTERRUPTED")),
                            wakeup,
                            row["id"],
                        ),
                    )
                    if row["state"] != "unknown" and (
                        failures >= policy.observation_max_failures
                        or now >= first + timedelta(seconds=policy.observation_window)
                    ):
                        changes = self._observation_exhausted(row, now)
                        assignments = sql.SQL(",").join(
                            sql.SQL("{}=%s").format(sql.Identifier(k)) for k in changes
                        )
                        values = [
                            Jsonb(v) if k == "failure_info" and v is not None else v
                            for k, v in changes.items()
                        ]
                        await conn.execute(
                            sql.SQL(
                                "UPDATE agent.background_operations SET {} WHERE id=%s"
                            ).format(assignments),
                            (*values, row["id"]),
                        )
                        if changes.get("state") == "unknown":
                            await self.task_event(
                                conn,
                                row["task_id"],
                                "operation_unknown",
                                "BACKGROUND_OBSERVATION_EXHAUSTED",
                                row["id"],
                            )
                        await self.stop_task(
                            conn, row["task_id"], code="BACKGROUND_RESULT_UNKNOWN"
                        )
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

    async def _monitor_write(self, operation, changes_for):
        """Retry only the transaction; callbacks are pure and never perform external I/O."""
        from agent_service.reliability import transient_database

        for attempt in range(3):
            try:
                async with self.locked() as conn:
                    prior = await (
                        await conn.execute(
                            """SELECT *,clock_timestamp() AS db_now FROM agent.background_operations
                        WHERE id=%s AND task_id=%s AND run_id=%s AND monitor_token=%s
                        AND monitor_until>clock_timestamp() FOR UPDATE""",
                            (
                                operation["id"],
                                operation["task_id"],
                                operation["run_id"],
                                operation["monitor_token"],
                            ),
                        )
                    ).fetchone()
                    if not prior:
                        raise ExternalFailure("BACKGROUND_MONITOR_LEASE_LOST")
                    changes = guarded_changes(prior, changes_for(prior))
                    if changes:
                        assignments = sql.SQL(",").join(
                            sql.SQL("{}=%s").format(sql.Identifier(k)) for k in changes
                        )
                        values = [
                            Jsonb(v)
                            if k
                            in {
                                "last_observation_error",
                                "cancel_last_error",
                                "failure_info",
                            }
                            and v is not None
                            else v
                            for k, v in changes.items()
                        ]
                        record = await (
                            await conn.execute(
                                sql.SQL(
                                    "UPDATE agent.background_operations SET {} WHERE id=%s RETURNING *"
                                ).format(assignments),
                                (*values, operation["id"]),
                            )
                        ).fetchone()
                        if prior["state"] != record["state"]:
                            await self.task_event(
                                conn,
                                prior["task_id"],
                                "operation_" + record["state"],
                                record["error_code"],
                                record["id"],
                            )
                        if (
                            record["state"] == "unknown"
                            or changes.get("cleanup_state") == "failed"
                        ):
                            await self.stop_task(
                                conn, prior["task_id"], code="BACKGROUND_RESULT_UNKNOWN"
                            )
                    else:
                        record = prior
                operation.update(record)
                return record
            except Exception as exc:
                if not transient_database(exc) or attempt == 2:
                    raise

    def _observation_exhausted(self, prior, now):
        from datetime import timedelta

        policy = self.registry.reliability
        if prior["cleanup_state"] == "released":
            return {
                "next_poll_at": now + timedelta(seconds=policy.cleanup_backoff_base)
            }
        if prior["state"] in {"succeeded", "failed", "cancelled", "unknown"}:
            return {
                "cleanup_state": "failed",
                "cleanup_error": "BACKGROUND_OBSERVATION_EXHAUSTED",
                "next_poll_at": max(
                    now + timedelta(seconds=policy.cleanup_backoff_base),
                    prior["remote_poll_not_before"] or now,
                ),
            }
        return {
            "state": "unknown",
            "error_code": "BACKGROUND_OBSERVATION_EXHAUSTED",
            "failure_info": failure_info("BACKGROUND_OBSERVATION_EXHAUSTED"),
            "finished_at": prior["finished_at"] or now,
            "cleanup_state": "failed",
            "cleanup_error": "BACKGROUND_OBSERVATION_EXHAUSTED",
            "next_poll_at": max(
                now + timedelta(seconds=policy.cleanup_backoff_base),
                prior["remote_poll_not_before"] or now,
            ),
        }

    async def begin_observation(self, operation):
        """Start one round, or handle a due budget event without making a remote call."""
        from datetime import timedelta

        policy = self.registry.reliability

        def changes(prior):
            now = prior["db_now"]
            if prior["observation_inflight"]:
                return {}  # Identical begin commit acknowledgement was lost.
            if (
                prior["state"] not in OP_TERMINAL
                and prior["cancel_sent_at"] is not None
                and now
                >= prior["cancel_sent_at"]
                + timedelta(seconds=self.settings.reconciliation_seconds)
            ):
                return {
                    "state": "unknown",
                    "error_code": "MCP_CANCEL_UNCONFIRMED",
                    "failure_info": failure_info("MCP_CANCEL_UNCONFIRMED"),
                    "finished_at": prior["finished_at"] or now,
                    "next_poll_at": max(
                        now + timedelta(seconds=policy.cleanup_backoff_base),
                        prior["remote_poll_not_before"] or now,
                    ),
                }
            exhausted = prior[
                "observation_failures"
            ] >= policy.observation_max_failures or (
                prior["observation_first_failed_at"] is not None
                and now
                >= prior["observation_first_failed_at"]
                + timedelta(seconds=policy.observation_window)
            )
            if exhausted and (
                prior["state"] not in OP_TERMINAL
                or (
                    prior["state"] != "unknown"
                    and prior["cleanup_error"] != "BACKGROUND_OBSERVATION_EXHAUSTED"
                )
            ):
                return self._observation_exhausted(prior, now)
            if prior["next_poll_at"] > now:
                return {}
            if (
                prior["remote_poll_not_before"]
                and prior["remote_poll_not_before"] > now
            ):
                if prior["state"] not in OP_TERMINAL and prior["deadline_at"] <= now:
                    return {
                        "state": "unknown",
                        "error_code": "BACKGROUND_TIMEOUT",
                        "failure_info": failure_info("BACKGROUND_TIMEOUT"),
                        "finished_at": prior["finished_at"] or now,
                        "next_poll_at": prior["remote_poll_not_before"],
                    }
                wakeup = prior["remote_poll_not_before"]
                if prior["state"] not in OP_TERMINAL:
                    deadlines = [prior["deadline_at"]]
                    if prior["observation_first_failed_at"]:
                        deadlines.append(
                            prior["observation_first_failed_at"]
                            + timedelta(seconds=policy.observation_window)
                        )
                    if prior["cancel_sent_at"]:
                        deadlines.append(
                            prior["cancel_sent_at"]
                            + timedelta(seconds=self.settings.reconciliation_seconds)
                        )
                    wakeup = min(wakeup, *deadlines)
                return {"next_poll_at": wakeup}
            return {"observation_started_at": now, "observation_inflight": True}

        record = await self._monitor_write(operation, changes)
        return bool(record["observation_inflight"])

    async def finish_observation(self, operation, *, error=None, delay=None):
        """Confirm the entire round once, including its budget and durable wakeup."""
        from datetime import timedelta
        from agent_service.reliability import backoff

        policy = self.registry.reliability
        started = operation["observation_started_at"]
        jitter = backoff(
            operation["observation_failures"] + 1,
            policy.cleanup_backoff_base
            if operation["state"] in OP_TERMINAL
            else policy.observation_backoff_base,
            policy.cleanup_backoff_cap
            if operation["state"] in OP_TERMINAL
            else policy.observation_backoff_cap,
        )
        error = validated_info(error)

        def changes(prior):
            if prior["observation_started_at"] != started:
                raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
            if not prior["observation_inflight"]:
                return {}  # Full transaction already committed; never debit twice.
            now = prior["db_now"]
            updates = {"observation_inflight": False}
            remote = prior["remote_poll_not_before"] or now
            if error is None:
                updates.update(
                    observation_failures=0,
                    observation_first_failed_at=None,
                    last_observation_error=None,
                    last_observed_at=now,
                )
                local = (
                    policy.cleanup_backoff_base
                    if prior["state"] == "unknown"
                    else (delay or self.settings.background_poll_seconds)
                )
                wakeup = max(now + timedelta(seconds=local), remote)
            else:
                failures = prior["observation_failures"] + 1
                first = prior["observation_first_failed_at"] or now
                updates.update(
                    observation_failures=failures,
                    observation_first_failed_at=first,
                    last_observation_error=error,
                )
                end = first + timedelta(seconds=policy.observation_window)
                wakeup = max(
                    now
                    + timedelta(
                        seconds=max(jitter, self.settings.background_poll_seconds)
                    ),
                    remote,
                )
                if prior["state"] not in OP_TERMINAL and (
                    failures >= policy.observation_max_failures or now >= end
                ):
                    updates.update(self._observation_exhausted(prior, now))
                    wakeup = max(updates["next_poll_at"], remote)
                elif prior["state"] not in OP_TERMINAL:
                    wakeup = min(
                        wakeup, end
                    )  # Expiry wakeup never authorizes an early remote query.
                elif prior["state"] != "unknown" and (
                    failures >= policy.observation_max_failures or now >= end
                ):
                    updates.update(self._observation_exhausted(prior, now))
            if prior["state"] not in OP_TERMINAL and prior["deadline_at"] > now:
                wakeup = min(wakeup, prior["deadline_at"])
            if prior["state"] not in OP_TERMINAL and prior["cancel_sent_at"]:
                wakeup = min(
                    wakeup,
                    prior["cancel_sent_at"]
                    + timedelta(seconds=self.settings.reconciliation_seconds),
                )
            updates["next_poll_at"] = wakeup
            return updates

        return await self._monitor_write(operation, changes)

    async def remote_poll(self, operation, interval_ms):
        from datetime import timedelta
        from agent_service.reliability import poll_interval

        interval_ms = poll_interval(interval_ms)
        previous = operation["remote_poll_not_before"]

        def changes(prior):
            if prior["remote_poll_not_before"] != previous:
                return {}  # This status response's interval already committed.
            return {
                "remote_poll_interval_ms": interval_ms,
                "remote_poll_not_before": prior["db_now"]
                + timedelta(milliseconds=interval_ms or 0),
            }

        return await self._monitor_write(operation, changes)

    async def reserve_cancel_attempt(self, operation, attempt_id):
        """The occupied attempt survives a crash even if handle.cancel was never sent."""
        from datetime import timedelta

        policy = self.registry.reliability
        safe = (
            operation["binding"]
            .get("manifest", {})
            .get("tool", {})
            .get("cancel_retry_safe")
            is True
        )

        def changes(prior):
            if prior["cancel_attempt_id"] == attempt_id:
                return {}
            now = prior["db_now"]
            first = prior["cancel_sent_at"]
            can_retry = (
                safe
                and prior["cancel_last_error"]
                and prior["cancel_last_error"]["code"] == "MCP_CANCEL_TRANSPORT_FAILED"
            )
            if (
                prior["cancel_acknowledged_at"]
                or prior["state"] in {"succeeded", "failed", "cancelled"}
                or prior["cancel_attempts"]
                >= (policy.cancel_max_attempts if safe else 1)
                or (prior["cancel_attempts"] and not can_retry)
                or (
                    first
                    and now
                    >= first + timedelta(seconds=self.settings.reconciliation_seconds)
                )
            ):
                return {}
            return {
                "cancel_requested_at": prior["cancel_requested_at"] or now,
                "cancel_sent_at": first or now,
                "cancel_attempts": prior["cancel_attempts"] + 1,
                "cancel_attempt_id": attempt_id,
                "cancel_attempted_at": now,
                "cancel_last_error": None,
            }

        record = await self._monitor_write(operation, changes)
        return record["cancel_attempt_id"] == attempt_id

    async def confirm_cancel_attempt(self, operation, attempt_id, *, error=None):
        error = validated_info(error)

        def changes(prior):
            if prior["cancel_attempt_id"] != attempt_id:
                raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
            if prior["cancel_acknowledged_at"]:
                return {}
            return {
                "cancel_last_error": error,
                **(
                    {"cancel_acknowledged_at": prior["db_now"]} if error is None else {}
                ),
            }

        return await self._monitor_write(operation, changes)

    async def workspace_background(self, task_id, root=None):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT id,state,cleanup_state FROM agent.background_operations
                WHERE task_id=%s AND backend='docker' AND cleanup_state<>'released' """,
                    (task_id,),
                )
            ).fetchall()


def guarded_changes(prior, changes, *, mode="ordinary"):
    """Apply one state contract to observations, confirmations and cleanup writes."""
    changes = dict(changes)
    terminal = {"succeeded", "failed", "cancelled"}
    protected = {"state", "result", "error_code", "failure_info"}
    for key in ("container_id", "daemon_id", "remote_handle", "receipt"):
        if prior[key] is not None and key in changes and changes[key] != prior[key]:
            raise ExternalFailure("BACKGROUND_RESOURCE_IDENTITY_CHANGED")
    if (
        prior["cleanup_state"] == "released"
        and changes.get("cleanup_state", "released") != "released"
    ):
        raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
    state = prior["state"]
    target = changes.get("state", state)
    if state in terminal or (state == "unknown" and mode != "reconcile"):
        if any(k in changes and changes[k] != prior[k] for k in protected):
            raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
    elif state == "unknown":
        if target == "unknown" and any(
            k in changes and changes[k] != prior[k] for k in protected
        ):
            raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
        if target not in terminal | {"unknown"}:
            raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
    else:
        transitions = {
            "starting": {"starting", "running", "cancelling", *terminal, "unknown"},
            "running": {"running", "cancelling", *terminal, "unknown"},
            "cancelling": {"cancelling", *terminal, "unknown"},
        }
        if target in {"starting", "running"} and (
            prior["cancel_requested_at"] is not None or state == "cancelling"
        ):
            target = changes["state"] = "cancelling"
        if target not in transitions[state]:
            raise ExternalFailure("BACKGROUND_STATE_CONFLICT")
    return changes
