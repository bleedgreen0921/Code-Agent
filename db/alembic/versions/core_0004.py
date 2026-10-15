"""A bounded audit record for non-replayable external writes."""

from alembic import op

revision = "core_0004"
down_revision = "core_0003"


def upgrade():
    op.get_bind().exec_driver_sql("""
    CREATE TABLE agent.external_operations (
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        call_id text NOT NULL,
        lease_token uuid NOT NULL,
        dispatch_id uuid NOT NULL,
        server_id text NOT NULL,
        account_id text NOT NULL,
        tool_name text NOT NULL,
        arguments_hash text NOT NULL,
        state text NOT NULL CHECK (state IN ('dispatched','succeeded','error','unknown')),
        dispatched_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        finished_at timestamptz,
        error_code text,
        receipt jsonb,
        PRIMARY KEY(run_id,call_id),
        CHECK (receipt IS NULL OR octet_length(receipt::text)<=16384)
    );
    CREATE INDEX external_operations_pending ON agent.external_operations(run_id) WHERE state='dispatched';
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
