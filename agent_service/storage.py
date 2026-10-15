"""PostgreSQL task state; all admission/final-publication races serialize in the DB."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from contracts.errors import ApiError
from contracts.v1 import ErrorCode
from psycopg.types.json import Jsonb

from agent_service.profiles import Registry

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running", "cancelling")
TERMINAL = ("completed", "failed", "cancelled")
ADMISSION_LOCK = 81431002


class Store:
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
            if revisions != [{"version_num": "core_0002"}]:
                raise RuntimeError("Core database baseline required")
            await conn.execute("SELECT 1 FROM agent.checkpoints LIMIT 1")
            privileges = await (
                await conn.execute("""SELECT has_schema_privilege(current_user,'agent','CREATE') AS ddl,
                has_database_privilege(current_user,current_database(),'CREATE') AS db_ddl,
                has_table_privilege(current_user,'agent.alembic_version','INSERT,UPDATE,DELETE,TRUNCATE') OR
                has_table_privilege(current_user,'agent.schema_version','INSERT,UPDATE,DELETE,TRUNCATE') OR
                has_table_privilege(current_user,'agent.checkpoint_migrations','INSERT,UPDATE,DELETE,TRUNCATE') AS revision_write""")
            ).fetchone()
            if any(privileges.values()):
                raise RuntimeError(
                    "A separate runtime role without migration permissions is required"
                )
        return True

    async def event(self, conn, run_id, status, error_code=None):
        await conn.execute(
            "INSERT INTO agent.events(run_id,status,error_code) VALUES(%s,%s,%s)",
            (run_id, status, error_code),
        )
        log.info(
            "run_transition",
            extra={"run_id": str(run_id), "status": status, "error_code": error_code},
        )

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
            await self.event(conn, row["id"], row["status"], row["error_code"])

    async def maintain(self):
        async with self.locked() as conn:
            await self.sweep(conn)

    def binding(self, profile="chat", workspace_id=None):
        try:
            return self.registry.bind(profile, workspace_id)
        except ValueError as exc:
            raise ApiError(422, ErrorCode.INVALID_REQUEST, str(exc)) from exc

    async def create_conversation(self, title, profile="chat", workspace_id=None):
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

    async def runs(
        self,
        *,
        status=None,
        created_after=None,
        created_before=None,
        limit=30,
        cursor=None,
    ):
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT * FROM agent.runs WHERE
                (%s::text IS NULL OR status=%s) AND (%s::timestamptz IS NULL OR created_at>%s)
                AND (%s::timestamptz IS NULL OR created_at<%s) AND (%s::uuid IS NULL OR id<%s::uuid)
                ORDER BY id DESC LIMIT %s""",
                    (
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
            n = (
                await (
                    await conn.execute(
                        "SELECT count(*) AS n FROM agent.runs WHERE status IN ('running','cancelling')"
                    )
                ).fetchone()
            )["n"]
            if n >= self.settings.max_active_runs:
                return None
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET status='running',worker_id=%s,lease_token=%s,
                started_at=clock_timestamp(),deadline_at=clock_timestamp()+%s*interval '1 second',
                lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=(SELECT id FROM agent.runs WHERE status='queued' ORDER BY created_at,id LIMIT 1)
                RETURNING *,extract(epoch FROM (deadline_at-clock_timestamp())) AS remaining""",
                    (
                        worker_id,
                        uuid4(),
                        self.settings.run_timeout,
                        self.settings.lease_seconds,
                    ),
                )
            ).fetchone()
            if row:
                await self.event(conn, row["id"], "running")
        return row

    async def renew(self, run_id, token):
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """UPDATE agent.runs SET lease_until=clock_timestamp()+%s*interval '1 second'
                WHERE id=%s AND lease_token=%s AND status IN ('running','cancelling')
                AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp() RETURNING status""",
                    (self.settings.lease_seconds, run_id, token),
                )
            ).fetchone()
        return row["status"] if row else None

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
        cancelled=False,
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
            if row["status"] == "cancelling" or cancelled:
                status, answer, checkpoint, error_code = "cancelled", None, None, None
            elif error_code:
                status, answer, checkpoint = "failed", None, None
            else:
                if answer is None or checkpoint is None:
                    raise ValueError(
                        "A completed Run requires an answer and explicit final checkpoint"
                    )
                status = "completed"
            published = await (
                await conn.execute(
                    """UPDATE agent.runs SET status=%s,answer=%s,final_checkpoint=%s,error_code=%s,
                finished_at=clock_timestamp() WHERE id=%s AND lease_token=%s
                AND status IN ('running','cancelling') AND lease_until>clock_timestamp()
                AND deadline_at>clock_timestamp() RETURNING id""",
                    (
                        status,
                        answer,
                        Jsonb(checkpoint) if checkpoint is not None else None,
                        error_code,
                        run_id,
                        token,
                    ),
                )
            ).fetchone()
            if not published:
                await self.sweep(conn)
                return False
            if status == "completed" and row["conversation_id"]:
                await conn.execute(
                    "UPDATE agent.conversations SET last_success_run_id=%s WHERE id=%s",
                    (run_id, row["conversation_id"]),
                )
            await self.event(conn, run_id, status, error_code)
        return True

    async def timeline(self, run_id):
        await self.get(run_id)
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    "SELECT id,status,error_code,created_at FROM agent.events WHERE run_id=%s ORDER BY id",
                    (run_id,),
                )
            ).fetchall()
        return {"run_id": str(run_id), "events": rows}

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
            "turn_no",
            "created_at",
            "started_at",
            "finished_at",
            "error_code",
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
