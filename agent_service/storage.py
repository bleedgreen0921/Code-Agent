"""PostgreSQL task state; all admission/final-publication races serialize in the DB."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4
from pathlib import Path

from agent_service.execution_config import WORKSPACE_TOOLS

from contracts.errors import ApiError
from contracts.v1 import ErrorCode
from psycopg.types.json import Jsonb

from agent_service.failure import failure_info, validated_info
from agent_service.profiles import Registry
from agent_service.tasks import TaskStore
from agent_service.background_store import BackgroundStore
from agent_service.diagnostics import DiagnosticStore
from agent_service.code_store import CodeStore

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running", "cancelling")
TERMINAL = ("completed", "failed", "cancelled")
ADMISSION_LOCK = 81431002


class Store(TaskStore, BackgroundStore, DiagnosticStore, CodeStore):
    def __init__(self, pool, settings, registry=None):
        self.pool, self.settings = pool, settings
        self.registry = registry or Registry.from_env()

    @asynccontextmanager
    async def locked(self):
        async with self.pool.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADMISSION_LOCK,))
            yield conn

    async def ready(self):
        async with self.pool.connection() as conn:
            revisions = await (
                await conn.execute("SELECT version_num FROM agent.alembic_version")
            ).fetchall()
            if revisions != [{"version_num": "core_0010"}]:
                raise RuntimeError("Core database baseline required")
            await conn.execute("SELECT 1 FROM agent.checkpoints LIMIT 1")
            await conn.execute("SELECT 1 FROM agent.store LIMIT 1")
            await conn.execute("SELECT 1 FROM agent.external_operations LIMIT 1")
            import json
            from db.migrate import CORE_QUERIES, ROOT

            signature = {}
            async with conn.transaction():
                await conn.execute("SET LOCAL search_path TO public")
                for key, query in CORE_QUERIES.items():
                    rows = await (await conn.execute(query)).fetchall()
                    signature[key] = [list(r.values()) for r in rows]
            if (
                signature
                != json.loads((ROOT / "structure.json").read_text())["core_0010"]
            ):
                raise RuntimeError("Core structure mismatch")
            from langgraph.store.postgres import PostgresStore

            versions = await (
                await conn.execute("SELECT v FROM agent.store_migrations ORDER BY v")
            ).fetchall()
            if versions != [{"v": v} for v in range(len(PostgresStore.MIGRATIONS))]:
                raise RuntimeError("Native Store migrations required")
            from db.native_store import QUERIES, check_signature

            actual = {}
            for key, query in QUERIES.items():
                rows = await (await conn.execute(query)).fetchall()
                actual[key] = [list(r.values()) for r in rows]
            check_signature(actual, [r["v"] for r in versions], complete=True)
            invalid = await (
                await conn.execute(
                    """SELECT 1 FROM pg_index WHERE indrelid IN ('agent.store'::regclass,'agent.store_migrations'::regclass) AND NOT indisvalid"""
                )
            ).fetchone()
            if invalid:
                raise RuntimeError("Native Store has invalid indexes")
            await conn.execute("SELECT run_id FROM agent.execution_resources LIMIT 1")
            privileges = await (
                await conn.execute("""SELECT has_schema_privilege(current_user,'agent','CREATE') AS ddl,
                has_database_privilege(current_user,current_database(),'CREATE') AS db_ddl,
                has_table_privilege(current_user,'agent.alembic_version','INSERT,UPDATE,DELETE,TRUNCATE') OR
                has_table_privilege(current_user,'agent.schema_version','INSERT,UPDATE,DELETE,TRUNCATE') OR
                has_table_privilege(current_user,'agent.checkpoint_migrations','INSERT,UPDATE,DELETE,TRUNCATE') OR
                has_table_privilege(current_user,'agent.store_migrations','INSERT,UPDATE,DELETE,TRUNCATE') AS revision_write""")
            ).fetchone()
            if any(privileges.values()):
                raise RuntimeError(
                    "A separate runtime role without migration permissions is required"
                )
        if self.registry.code.settings.enabled or self.registry.review.settings.enabled:
            root = Path(
                (
                    self.registry.code
                    if self.registry.code.settings.enabled
                    else self.registry.review
                ).settings.storage_root
            )
            import os

            if (
                not root.is_dir()
                or str(root.resolve()) != str(root)
                or not os.access(root, os.W_OK | os.X_OK)
            ):
                raise RuntimeError("Managed code storage is unavailable")
        return True

    async def event(self, conn, run_id, status, error_code=None):
        await conn.execute(
            "INSERT INTO agent.events(run_id,status,error_code) VALUES(%s,%s,%s)",
            (run_id, status, error_code),
        )
        task = await (
            await conn.execute("SELECT task_id FROM agent.runs WHERE id=%s", (run_id,))
        ).fetchone()
        try:
            log.info(
                "run_transition",
                extra={
                    "run_id": str(run_id),
                    "task_id": task["task_id"] if task else None,
                    "status": status,
                    "error_code": error_code,
                },
            )
        except Exception:
            pass

    async def sweep(self, conn):
        rows = await (
            await conn.execute("""UPDATE agent.runs SET status='failed',finished_at=clock_timestamp(),
            error_code=CASE WHEN status='queued' THEN 'QUEUE_TIMEOUT'
                WHEN lease_until<=clock_timestamp() THEN 'LEASE_EXPIRED' ELSE 'RUN_TIMEOUT' END
            WHERE (status='queued' AND queue_deadline<=clock_timestamp()) OR
                (status IN ('running','cancelling') AND (lease_until<=clock_timestamp() OR deadline_at<=clock_timestamp()))
            RETURNING id,status,error_code""")
        ).fetchall()
        for row in rows:
            await conn.execute(
                "UPDATE agent.runs SET failure_info=%s WHERE id=%s",
                (Jsonb(failure_info(row["error_code"])), row["id"]),
            )
            await self.event(conn, row["id"], row["status"], row["error_code"])
        unknown = await (
            await conn.execute(
                """UPDATE agent.external_operations AS op
            SET state='unknown',error_code='MCP_RESULT_UNKNOWN',failure_info=%s,finished_at=clock_timestamp()
            FROM agent.runs AS r WHERE op.run_id=r.id AND op.state='dispatched'
            AND (r.status IN ('completed','failed','cancelled') OR r.lease_until<=clock_timestamp()
                 OR r.lease_token<>op.lease_token) RETURNING op.run_id,r.task_id""",
                (Jsonb(failure_info("MCP_RESULT_UNKNOWN")),),
            )
        ).fetchall()
        for item in unknown:
            if item["task_id"]:
                await self.stop_task(conn, item["task_id"], code="MCP_RESULT_UNKNOWN")
            await self.event(
                conn, item["run_id"], "external_operation_unknown", "MCP_RESULT_UNKNOWN"
            )

        await self.reconcile_tasks(conn)
        await self.reconcile_code_tasks(conn)

    async def maintain(self):
        async with self.locked() as conn:
            await self.sweep(conn)
            await self.schedule_tasks(conn)

    def require_code_entry(self, profile, *, code_task=False):
        from agent_service.code_registry import CODE_PROFILES

        if profile in CODE_PROFILES and not code_task:
            raise ApiError(
                422, "CODE_TASK_REQUIRED", "Use POST /v1/code/tasks for this profile"
            )

    def binding(self, profile="chat", workspace_id=None):
        try:
            return self.registry.bind(profile, workspace_id)
        except ValueError as exc:
            raise ApiError(422, ErrorCode.INVALID_REQUEST, str(exc)) from exc

    async def create_conversation(self, title, profile="chat", workspace_id=None):
        self.require_code_entry(profile)
        binding = self.binding(profile, workspace_id)
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "INSERT INTO agent.conversations(id,title,profile,workspace_id,workspace_identity) VALUES(%s,%s,%s,%s,%s) RETURNING *",
                    (
                        uuid4(),
                        title,
                        profile,
                        workspace_id,
                        binding["workspace_identity"],
                    ),
                )
            ).fetchone()

    async def conversations(self, limit=30, cursor=None):
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT * FROM agent.conversations WHERE (%s::uuid IS NULL OR id<%s::uuid) ORDER BY id DESC LIMIT %s",
                    (cursor, cursor, limit + 1),
                )
            ).fetchall()
        return page(rows, limit)

    async def conversation(self, conversation_id):
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.conversations WHERE id=%s", (conversation_id,)
                )
            ).fetchone()
            if not row:
                raise ApiError(404, ErrorCode.NOT_FOUND, "Conversation not found")
            turns = await (
                await conn.execute(
                    "SELECT * FROM agent.runs WHERE conversation_id=%s ORDER BY turn_no",
                    (conversation_id,),
                )
            ).fetchall()
        return {**row, "turns": [public_run(r) for r in turns]}

    async def submit(
        self, task, key=None, conversation_id=None, *, profile="chat", workspace_id=None
    ):
        if key is not None and (not key.strip() or len(key) > 200):
            raise ApiError(422, ErrorCode.INVALID_REQUEST, "Invalid Idempotency-Key")
        self.require_code_entry(profile)
        scope = str(conversation_id) if conversation_id else "standalone"
        async with self.locked() as conn:
            await self.sweep(conn)
            if key is not None:
                prior = await (
                    await conn.execute(
                        "SELECT * FROM agent.runs WHERE idempotency_scope=%s AND idempotency_key=%s",
                        (scope, key),
                    )
                ).fetchone()
                if prior:
                    if prior["task"] != task or (
                        not conversation_id
                        and (
                            prior["profile"] != profile
                            or prior["workspace_id"] != workspace_id
                        )
                    ):
                        raise ApiError(
                            409,
                            ErrorCode.IDEMPOTENCY_CONFLICT,
                            "Key already used for another task",
                        )
                    return prior, True
            previous, turn_no = None, None
            if conversation_id:
                conversation = await (
                    await conn.execute(
                        "SELECT * FROM agent.conversations WHERE id=%s FOR UPDATE",
                        (conversation_id,),
                    )
                ).fetchone()
                if not conversation:
                    raise ApiError(404, ErrorCode.NOT_FOUND, "Conversation not found")
                profile, workspace_id = (
                    conversation["profile"],
                    conversation["workspace_id"],
                )
                self.require_code_entry(profile)
                try:
                    binding = self.registry.bind(profile, workspace_id)
                except ValueError as exc:
                    raise ApiError(
                        409,
                        ErrorCode.CONFIGURATION_CHANGED,
                        "Conversation configuration unavailable",
                    ) from exc
                if binding["workspace_identity"] != conversation["workspace_identity"]:
                    raise ApiError(
                        409,
                        ErrorCode.CONFIGURATION_CHANGED,
                        "Workspace root changed; create a new conversation",
                    )
                busy = await (
                    await conn.execute(
                        "SELECT 1 FROM agent.runs WHERE conversation_id=%s AND status IN ('queued','running','cancelling')",
                        (conversation_id,),
                    )
                ).fetchone()
                if busy:
                    raise ApiError(
                        409,
                        ErrorCode.CONVERSATION_BUSY,
                        "Conversation already has an active turn",
                    )
                previous = conversation["last_success_run_id"]
                turn_no = (
                    await (
                        await conn.execute(
                            "SELECT COALESCE(max(turn_no),0)+1 AS n FROM agent.runs WHERE conversation_id=%s",
                            (conversation_id,),
                        )
                    ).fetchone()
                )["n"]
            binding = self.binding(profile, workspace_id)
            selected = self.registry.profiles[profile]
            from agent_service.background import BACKGROUND_TOOLS

            if set(selected.tools) & BACKGROUND_TOOLS or any(
                self.registry.mcp_tools[n].selection.execution == "background"
                for n in selected.tools
                if n in self.registry.mcp_tools
            ):
                raise ApiError(
                    422,
                    ErrorCode.INVALID_REQUEST,
                    "Background profiles require POST /v1/tasks",
                )
            counts = await (
                await conn.execute(
                    "SELECT count(*) FILTER(WHERE status='queued') AS queued,count(*) AS unfinished FROM agent.runs WHERE status IN ('queued','running','cancelling')"
                )
            ).fetchone()
            if (
                counts["queued"] >= self.settings.max_queued_runs
                or counts["unfinished"] >= self.settings.max_unfinished_runs
            ):
                raise ApiError(
                    429, ErrorCode.QUEUE_CAPACITY_EXCEEDED, "Task queue is full"
                )
            row = await (
                await conn.execute(
                    """INSERT INTO agent.runs(id,task,conversation_id,turn_no,previous_run_id,
                idempotency_scope,idempotency_key,profile,workspace_id,submitted_binding,status,queue_deadline) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued',
                clock_timestamp()+%s*interval '1 second') RETURNING *""",
                    (
                        uuid4(),
                        task,
                        conversation_id,
                        turn_no,
                        previous,
                        scope,
                        key,
                        profile,
                        workspace_id,
                        Jsonb(binding),
                        self.settings.queue_timeout,
                    ),
                )
            ).fetchone()
            await self.event(conn, row["id"], "queued")
        return row, False

    async def get(self, run_id, conversation_id=None):
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute("SELECT * FROM agent.runs WHERE id=%s", (run_id,))
            ).fetchone()
        if not row or (
            conversation_id is not None and row["conversation_id"] != conversation_id
        ):
            raise ApiError(404, ErrorCode.NOT_FOUND, "Run not found")
        return row

    async def run_detail(self, run_id, conversation_id=None):
        async with self.snapshot() as conn:
            row = await (
                await conn.execute("SELECT * FROM agent.runs WHERE id=%s", (run_id,))
            ).fetchone()
            if not row or (
                conversation_id is not None
                and row["conversation_id"] != conversation_id
            ):
                raise ApiError(404, ErrorCode.NOT_FOUND, "Run not found")
            return public_run(row) | {
                "unresolved": await self.unresolved(run_id=run_id, conn=conn)
            }

    async def runs(
        self,
        *,
        status=None,
        task_id=None,
        created_after=None,
        created_before=None,
        limit=30,
        cursor=None,
    ):
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT * FROM agent.runs WHERE
                (%s::uuid IS NULL OR task_id=%s) AND (%s::text IS NULL OR status=%s) AND (%s::timestamptz IS NULL OR created_at>%s)
                AND (%s::timestamptz IS NULL OR created_at<%s) AND (%s::uuid IS NULL OR id<%s::uuid)
                ORDER BY id DESC LIMIT %s""",
                    (
                        task_id,
                        task_id,
                        status,
                        status,
                        created_after,
                        created_after,
                        created_before,
                        created_before,
                        cursor,
                        cursor,
                        limit + 1,
                    ),
                )
            ).fetchall()
        return page([public_run(r) for r in rows], limit)

    async def summary(self):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute("""SELECT
                count(*) FILTER(WHERE status='queued' AND queue_deadline>clock_timestamp()) AS queued_within_deadline,
                count(*) FILTER(WHERE status='queued' AND queue_deadline<=clock_timestamp()) AS queued_past_deadline,
                count(*) FILTER(WHERE status='running' AND lease_until>clock_timestamp()) AS running_lease_valid,
                count(*) FILTER(WHERE status='running' AND lease_until<=clock_timestamp()) AS running_lease_expired,
                count(*) FILTER(WHERE status='cancelling') AS cancelling,
                min(created_at) FILTER(WHERE status='queued' AND queue_deadline>clock_timestamp()) AS oldest_claimable_created_at
                FROM agent.runs""")
            ).fetchone()

    async def cancel(self, run_id, conversation_id=None):
        async with self.locked() as conn:
            await self.sweep(conn)
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.runs WHERE id=%s FOR UPDATE", (run_id,)
                )
            ).fetchone()
            if not row or (
                conversation_id is not None
                and row["conversation_id"] != conversation_id
            ):
                raise ApiError(404, ErrorCode.NOT_FOUND, "Run not found")
            if row.get("task_id"):
                await self.stop_task(conn, row["task_id"])
                await self.reconcile_tasks(conn)
                return await (
                    await conn.execute(
                        "SELECT * FROM agent.runs WHERE id=%s", (run_id,)
                    )
                ).fetchone()
            if row["status"] in TERMINAL or row["status"] == "cancelling":
                return row
            status = "cancelled" if row["status"] == "queued" else "cancelling"
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET status=%s,
                finished_at=CASE WHEN %s='cancelled' THEN clock_timestamp() ELSE NULL END
                WHERE id=%s RETURNING *""",
                    (status, status, run_id),
                )
            ).fetchone()
            await self.event(conn, run_id, status)
        return row

    async def claim(self, worker_id):
        async with self.locked() as conn:
            await self.sweep(conn)
            await self.schedule_tasks(conn)
            occupied = await (
                await conn.execute(
                    """SELECT e.run_id,r.task_id,e.workspace_root FROM agent.execution_resources e JOIN agent.runs r ON r.id=e.run_id WHERE e.state<>'released'
                    UNION ALL SELECT e.run_id,r.task_id,e.workspace_root FROM agent.code_execution_resources e JOIN agent.runs r ON r.id=e.run_id WHERE e.state<>'released'
                    UNION ALL SELECT run_id,task_id,workspace_root FROM agent.background_operations WHERE backend='docker' AND cleanup_state<>'released'"""
                )
            ).fetchall()
            n = (
                await (
                    await conn.execute("""SELECT count(*) AS n FROM (
                SELECT id FROM agent.runs WHERE status IN ('running','cancelling')
                UNION SELECT run_id FROM agent.execution_resources WHERE state <> 'released'
                UNION SELECT run_id FROM agent.code_execution_resources WHERE state <> 'released'
            ) capacity""")
                ).fetchone()
            )["n"]
            if n >= self.settings.max_active_runs:
                return None
            queued = await (
                await conn.execute(
                    "SELECT * FROM agent.runs WHERE status='queued' ORDER BY created_at,id"
                )
            ).fetchall()
            selected, root = None, None
            for candidate in queued:
                candidate_root = None
                try:
                    profile = self.registry.resolve(candidate["submitted_binding"])
                    if set(profile.tools) & (
                        WORKSPACE_TOOLS | {"start_background_shell"}
                    ):
                        candidate_root = self.registry.workspaces[
                            candidate["workspace_id"]
                        ]
                except (RuntimeError, ValueError, KeyError):
                    # Claim changed bindings so the Worker publishes CONFIGURATION_CHANGED.
                    pass
                if candidate_root and any(
                    Path(candidate_root).is_relative_to(item["workspace_root"])
                    or Path(item["workspace_root"]).is_relative_to(candidate_root)
                    for item in occupied
                ):
                    continue
                selected, root = candidate, candidate_root
                break
            if selected is None:
                return None
            token = uuid4()
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET status='running',worker_id=%s,lease_token=%s,
                started_at=clock_timestamp(),deadline_at=LEAST(clock_timestamp()+%s*interval '1 second',(SELECT deadline_at FROM agent.tasks WHERE id=agent.runs.task_id)),
                lease_until=clock_timestamp()+%s*interval '1 second' WHERE id=%s
                RETURNING *,extract(epoch FROM (deadline_at-clock_timestamp())) AS remaining""",
                    (
                        worker_id,
                        token,
                        min(
                            self.settings.run_timeout,
                            self.registry.code_for(selected).settings.run_seconds,
                        )
                        if selected["profile"]
                        in (
                            self.registry.code.settings.profile,
                            self.registry.review.settings.profile,
                        )
                        else self.settings.run_timeout,
                        self.settings.lease_seconds,
                        selected["id"],
                    ),
                )
            ).fetchone()
            if root:
                # Database identity is included in names/labels to isolate projects sharing a daemon.
                identity = (
                    await (
                        await conn.execute("SELECT current_database() AS name")
                    ).fetchone()
                )["name"]
                from agent_service.profiles import fingerprint

                name = "agent-exec-" + fingerprint(identity)[:12] + "-" + row["id"].hex
                await conn.execute(
                    """INSERT INTO agent.execution_resources
                    (run_id,lease_token,worker_id,workspace_root,container_name) VALUES(%s,%s,%s,%s,%s)""",
                    (row["id"], token, worker_id, root, name),
                )
            if row.get("task_id"):
                await self.task_transition(conn, row["task_id"], "running")
            await self.event(conn, row["id"], "running")
        row["_execution_reserved"] = bool(root)
        return row

    async def execution_resource(self, run_id):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM agent.execution_resources WHERE run_id=%s", (run_id,)
                )
            ).fetchone()

    async def resource_update(
        self, run_id, token, *, require_live=False, recovered=False, **changes
    ):
        allowed = {"daemon_id", "container_id", "state", "phase", "cleanup_error"}
        if not changes or changes.keys() - allowed:
            raise ValueError("Invalid execution resource update")
        from psycopg import sql

        async with self.locked() as conn:
            suffix = sql.SQL("")
            if require_live:
                suffix = sql.SQL(""" AND EXISTS(SELECT 1 FROM agent.runs r WHERE r.id=e.run_id
                    AND r.lease_token=e.lease_token AND r.status='running'
                    AND r.lease_until>clock_timestamp() AND r.deadline_at>clock_timestamp())
                    AND e.state IN ('reserved','active')""")
            identity_values = []
            for key in ("daemon_id", "container_id"):
                if key in changes:
                    suffix += sql.SQL(
                        " AND (e.{} IS NULL OR e.{} IS NOT DISTINCT FROM %s)"
                    ).format(sql.Identifier(key), sql.Identifier(key))
                    identity_values.append(changes[key])
            assignments = sql.SQL(",").join(
                sql.SQL("{}=%s").format(sql.Identifier(k)) for k in changes
            )
            statement = sql.SQL("""UPDATE agent.execution_resources e SET {},updated_at=clock_timestamp(),
                released_at=CASE WHEN %s='released' THEN clock_timestamp() ELSE released_at END
                WHERE e.run_id=%s AND e.lease_token=%s AND e.state<>'released' {} RETURNING e.*""").format(
                assignments, suffix
            )
            result = await (
                await conn.execute(
                    statement,
                    (
                        *changes.values(),
                        changes.get("state"),
                        run_id,
                        token,
                        *identity_values,
                    ),
                )
            ).fetchone()
            if result and "state" in changes:
                await self.event(
                    conn,
                    run_id,
                    "environment_" + changes["state"],
                    "RESOURCE_CLEANUP_FAILED"
                    if changes["state"] == "cleanup_failed"
                    else None,
                )
            if result and changes.get("phase") == "creating":
                await self.event(conn, run_id, "environment_preparing")
            if result and recovered and changes.get("state") == "released":
                await self.event(conn, run_id, "environment_recovered")
            return result

    async def orphan_resources(self):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute("""SELECT e.* FROM agent.execution_resources e
                JOIN agent.runs r ON r.id=e.run_id WHERE e.state<>'released'
                AND (r.status IN ('failed','cancelled','completed') OR r.lease_until<=clock_timestamp())
                ORDER BY e.created_at""")
            ).fetchall()

    async def resource_diagnostics(self):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute("""SELECT run_id,worker_id,workspace_root,daemon_id,
                container_name,container_id,state,phase,cleanup_error,updated_at
                FROM agent.execution_resources WHERE state<>'released' ORDER BY created_at""")
            ).fetchall()

    async def renew(self, run_id, token, *, with_lease=False):
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND lease_token=%s AND status IN ('running','cancelling')
                AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp() RETURNING status,lease_until""",
                    (self.settings.lease_seconds, run_id, token),
                )
            ).fetchone()
        return row if with_lease else row["status"] if row else None

    async def record_runtime_config(self, run_id, token, snapshot):
        """Persist once under a valid lease. Repeats confirm the identical snapshot."""
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET runtime_config=%s
                WHERE id=%s AND lease_token=%s AND status='running'
                AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp()
                AND (runtime_config IS NULL OR runtime_config=%s) RETURNING id""",
                    (Jsonb(snapshot), run_id, token, Jsonb(snapshot)),
                )
            ).fetchone()
        return row is not None

    async def finish(
        self,
        run_id,
        token,
        *,
        answer=None,
        checkpoint=None,
        error_code=None,
        failure_info=None,
        cancelled=False,
        delivered=(),
        verification=None,
        candidate_answer=None,
        candidate_checkpoint=None,
        code_artifacts=None,
    ):
        async with self.locked() as conn:
            await self.sweep(conn)
            row = await (
                await conn.execute(
                    "SELECT * FROM agent.runs WHERE id=%s FOR UPDATE", (run_id,)
                )
            ).fetchone()
            if row["lease_token"] != token or row["status"] not in (
                "running",
                "cancelling",
            ):
                return False
            from agent_service.verification import (
                validate_publication,
                message_source_eligible,
            )

            if verification is not None:
                verification = validate_publication(
                    verification,
                    row,
                    candidate_answer if candidate_answer is not None else answer,
                    candidate_checkpoint
                    if candidate_checkpoint is not None
                    else checkpoint,
                )
            task_failure = None
            if row.get("task_id"):
                task_failure = await (
                    await conn.execute(
                        "SELECT error_code FROM agent.tasks WHERE id=%s AND status='failed'",
                        (row["task_id"],),
                    )
                ).fetchone()
            if task_failure:
                status, answer, checkpoint, error_code = (
                    "failed",
                    None,
                    None,
                    task_failure["error_code"],
                )
            elif row["status"] == "cancelling" or cancelled:
                status, answer, checkpoint, error_code = "cancelled", None, None, None
            elif error_code:
                status, answer, checkpoint = "failed", None, None
            else:
                if answer is None or checkpoint is None:
                    raise ValueError(
                        "A completed Run requires an answer and explicit final checkpoint"
                    )
                occupied = await (
                    await conn.execute(
                        """SELECT 1 FROM agent.execution_resources WHERE run_id=%s AND state<>'released'
                        UNION ALL SELECT 1 FROM agent.code_execution_resources WHERE run_id=%s AND state<>'released'""",
                        (run_id, run_id),
                    )
                ).fetchone()
                if occupied:
                    return False
                config = row["submitted_binding"].get("verification")
                if config:
                    verification = validate_publication(
                        verification, row, answer, checkpoint
                    )
                    from agent_service.execution_errors import ExecutionFailure

                    try:
                        final_now = await self.verification_candidate(
                            run_id, delivered, conn=conn
                        )
                    except ExecutionFailure as exc:
                        error_code, final_now = exc.code, False
                    if (
                        not error_code
                        and config["mode"] == "required"
                        and verification["final_candidate"]
                    ):
                        if verification["status"] != "passed" or not final_now:
                            error_code = (
                                "VERIFICATION_FAILED"
                                if verification["status"] == "failed"
                                else "VERIFICATION_INCONCLUSIVE"
                            )
                status = "completed"
                if error_code:
                    status, answer, checkpoint = "failed", None, None
            from agent_service.failure import failure_info as describe_failure

            info = (
                validated_info(failure_info)
                if failure_info is not None and failure_info["code"] == error_code
                else describe_failure(error_code)
                if error_code
                else None
            )
            published = await (
                await conn.execute(
                    """UPDATE agent.runs SET status=%s,answer=%s,final_checkpoint=%s,error_code=%s,failure_info=%s,verification=%s,
                finished_at=clock_timestamp() WHERE id=%s AND lease_token=%s
                AND status IN ('running','cancelling') AND lease_until>clock_timestamp()
                AND deadline_at>clock_timestamp() RETURNING id""",
                    (
                        status,
                        answer,
                        Jsonb(checkpoint) if checkpoint is not None else None,
                        error_code,
                        Jsonb(info) if info else None,
                        Jsonb(verification) if verification is not None else None,
                        run_id,
                        token,
                    ),
                )
            ).fetchone()
            if (
                published
                and status == "completed"
                and row["profile"]
                in (
                    self.registry.code.settings.profile,
                    self.registry.review.settings.profile,
                )
            ):
                if not code_artifacts:
                    raise ValueError("A code task requires staged report artifacts")
                await self.publish_code_report(conn, row, code_artifacts)
            if not published:
                await self.sweep(conn)
                return False
            eligible = message_source_eligible(
                {
                    **row,
                    "status": status,
                    "verification": verification,
                    "answer": answer,
                    "final_checkpoint": checkpoint,
                }
            )
            if eligible and row["conversation_id"]:
                await conn.execute(
                    "UPDATE agent.conversations SET last_success_run_id=%s WHERE id=%s",
                    (run_id, row["conversation_id"]),
                )
            if row.get("task_id") and eligible:
                await conn.execute(
                    "UPDATE agent.tasks SET last_success_run_id=%s WHERE id=%s",
                    (run_id, row["task_id"]),
                )
                await conn.execute(
                    """UPDATE agent.background_operations SET consumed_run_id=%s
                    WHERE task_id=%s AND id=ANY(%s::uuid[]) AND state IN ('succeeded','failed','cancelled')
                    AND cleanup_state='released' AND consumed_run_id IS NULL""",
                    (run_id, row["task_id"], list(delivered)),
                )
            await self.event(conn, run_id, status, error_code)
            await self.reconcile_tasks(conn)
        return True

    async def external_operations(self, run_id=None):
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    "SELECT * FROM agent.external_operations WHERE (%s::uuid IS NULL OR run_id=%s) ORDER BY dispatched_at,call_id",
                    (run_id, run_id),
                )
            ).fetchall()

    async def timeline(self, run_id):
        await self.get(run_id)
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id,status,error_code,created_at FROM agent.events WHERE run_id=%s ORDER BY id",
                    (run_id,),
                )
            ).fetchall()
        return {
            "run_id": str(run_id),
            "events": rows,
            "external_operations": await self.external_operations(run_id),
        }

    async def heartbeat(self, worker_id, *, stopped=False):
        async with self.pool.connection() as conn:
            await conn.execute(
                """INSERT INTO agent.workers(id) VALUES(%s) ON CONFLICT(id) DO UPDATE
                SET heartbeat_at=clock_timestamp(),stopped_at=CASE WHEN %s THEN clock_timestamp() ELSE NULL END""",
                (worker_id, stopped),
            )

    async def workers(self):
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT *,stopped_at IS NULL AND heartbeat_at>clock_timestamp()-%s*interval '1 second' AS online
                FROM agent.workers ORDER BY started_at DESC""",
                    (self.settings.lease_seconds,),
                )
            ).fetchall()
        return {
            "service": "agent",
            "online_count": sum(r["online"] for r in rows),
            "instances": rows,
        }


def public_run(row):
    return {
        key: row[key]
        for key in (
            "id",
            "status",
            "task",
            "profile",
            "workspace_id",
            "conversation_id",
            "task_id",
            "execution_no",
            "turn_no",
            "created_at",
            "started_at",
            "finished_at",
            "error_code",
            "failure_info",
            "verification",
        )
    } | {
        "run_id": str(row["id"]),
        "result": {"answer": row["answer"]} if row["status"] == "completed" else None,
    }


def page(rows, limit):
    return {
        "items": rows[:limit],
        "next_cursor": str(rows[limit - 1]["id"]) if len(rows) > limit else None,
    }
