"""Persistent tasks and independently leased background operations."""

from alembic import op

revision = "core_0005"
down_revision = "core_0004"


def upgrade():
    op.get_bind().exec_driver_sql("""
    CREATE TABLE agent.tasks (
        id uuid PRIMARY KEY,
        task text NOT NULL,
        profile text NOT NULL,
        workspace_id text,
        submitted_binding jsonb NOT NULL,
        idempotency_key text UNIQUE,
        input_hash text NOT NULL,
        status text NOT NULL CHECK(status IN ('waiting_dependencies','queued','running','waiting_background','cancelling','completed','failed','cancelled')),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        ready_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        deadline_at timestamptz NOT NULL,
        finished_at timestamptz,
        current_run_id uuid REFERENCES agent.runs(id),
        last_success_run_id uuid REFERENCES agent.runs(id),
        answer text,
        error_code text
    );
    CREATE TABLE agent.task_dependencies (
        task_id uuid NOT NULL REFERENCES agent.tasks(id),
        dependency_id uuid NOT NULL REFERENCES agent.tasks(id),
        PRIMARY KEY(task_id,dependency_id), CHECK(task_id<>dependency_id)
    );
    ALTER TABLE agent.runs ADD COLUMN task_id uuid REFERENCES agent.tasks(id);
    ALTER TABLE agent.runs ADD COLUMN execution_no integer;
    ALTER TABLE agent.runs ADD CONSTRAINT runs_task_execution CHECK
        ((task_id IS NULL AND execution_no IS NULL) OR (task_id IS NOT NULL AND execution_no BETWEEN 1 AND 5 AND conversation_id IS NULL));
    ALTER TABLE agent.runs ADD CONSTRAINT runs_task_sequence UNIQUE(task_id,execution_no);
    CREATE UNIQUE INDEX runs_task_active ON agent.runs(task_id)
        WHERE task_id IS NOT NULL AND status IN ('queued','running','cancelling');
    ALTER TABLE agent.runs ADD CONSTRAINT runs_task_owner UNIQUE(id,task_id);
    CREATE TABLE agent.background_operations (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.tasks(id),
        run_id uuid NOT NULL,
        call_id text NOT NULL,
        dispatch_id uuid NOT NULL UNIQUE,
        backend text NOT NULL CHECK(backend IN ('docker','mcp')),
        arguments_hash text NOT NULL,
        binding jsonb NOT NULL,
        state text NOT NULL CHECK(state IN ('starting','running','cancelling','succeeded','failed','cancelled','unknown')),
        phase text NOT NULL DEFAULT 'reserved',
        workspace_root text,
        container_name text UNIQUE,
        container_id text,
        daemon_id text,
        remote_handle text,
        receipt jsonb,
        deadline_at timestamptz NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        finished_at timestamptz,
        cancel_requested_at timestamptz,
        cancel_sent_at timestamptz,
        next_poll_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        monitor_token uuid,
        monitor_until timestamptz,
        result jsonb,
        error_code text,
        cleanup_state text NOT NULL DEFAULT 'pending' CHECK(cleanup_state IN ('pending','released','failed')),
        cleanup_error text,
        consumed_run_id uuid,
        FOREIGN KEY(run_id,task_id) REFERENCES agent.runs(id,task_id),
        FOREIGN KEY(consumed_run_id,task_id) REFERENCES agent.runs(id,task_id),
        UNIQUE(run_id,call_id),
        CHECK(result IS NULL OR octet_length(result::text)<=65536),
        CHECK(receipt IS NULL OR octet_length(receipt::text)<=16384),
        CHECK((backend='docker' AND workspace_root IS NOT NULL AND container_name IS NOT NULL) OR
              (backend='mcp' AND workspace_root IS NULL AND container_name IS NULL))
    );
    CREATE INDEX background_pending ON agent.background_operations(next_poll_at)
        WHERE cleanup_state<>'released' OR state IN ('starting','running','cancelling');
    CREATE TABLE agent.task_events (
        id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.tasks(id),
        operation_id uuid REFERENCES agent.background_operations(id),
        status text NOT NULL,
        error_code text,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE INDEX tasks_ready ON agent.tasks(ready_at,id) WHERE status IN ('queued','waiting_dependencies','waiting_background');
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
