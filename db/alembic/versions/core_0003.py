"""Lease-owned execution environments. Released resources remain diagnostic records."""

from alembic import op

revision = "core_0003"
down_revision = "core_0002"


def upgrade():
    op.get_bind().exec_driver_sql("""
    CREATE TABLE agent.execution_resources (
        run_id uuid PRIMARY KEY REFERENCES agent.runs(id),
        lease_token uuid NOT NULL,
        worker_id uuid NOT NULL,
        workspace_root text NOT NULL,
        daemon_id text,
        container_name text NOT NULL UNIQUE,
        container_id text,
        state text NOT NULL DEFAULT 'reserved' CHECK (state IN ('reserved','active','closing','cleanup_failed','released')),
        phase text NOT NULL DEFAULT 'reserved' CHECK (phase IN ('reserved','creating','created','starting','started')),
        cleanup_error text,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        released_at timestamptz
    );
    CREATE INDEX execution_resource_occupied ON agent.execution_resources(workspace_root) WHERE state <> 'released';
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
