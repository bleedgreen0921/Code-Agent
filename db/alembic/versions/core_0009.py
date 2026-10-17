"""Isolated analysis tasks, immutable snapshots, checks and managed artifacts."""

from alembic import op

revision = "core_0009"
down_revision = "core_0008"


def upgrade():
    op.get_bind().exec_driver_sql("""
    CREATE TABLE agent.code_tasks (
        task_id uuid PRIMARY KEY REFERENCES agent.tasks(id),
        request jsonb NOT NULL,
        policy_fingerprint text NOT NULL,
        check_suite_id text,
        check_ids jsonb NOT NULL,
        stage text NOT NULL DEFAULT 'analysis' CHECK(stage='analysis'),
        phase text NOT NULL DEFAULT 'queued',
        snapshot_id uuid,
        report_artifact_id uuid,
        markdown_artifact_id uuid,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE TABLE agent.code_snapshots (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL UNIQUE REFERENCES agent.runs(id),
        state text NOT NULL CHECK(state IN ('preparing','sealed','failed','deleted')),
        target jsonb NOT NULL,
        commit_sha text,
        policy_fingerprint text NOT NULL,
        storage_key text NOT NULL UNIQUE,
        content_digest text,
        git_digest text,
        manifest_digest text,
        resolved_at timestamptz,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        deleted_at timestamptz
    );
    ALTER TABLE agent.execution_resources ADD COLUMN execution_root text,
        ADD COLUMN snapshot_id uuid REFERENCES agent.code_snapshots(id);
    CREATE TABLE agent.code_execution_resources (
        id uuid PRIMARY KEY,
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        lease_token uuid NOT NULL,
        worker_id uuid NOT NULL,
        purpose text NOT NULL CHECK(purpose IN ('snapshot','check')),
        workspace_root text NOT NULL,
        execution_root text,
        snapshot_id uuid REFERENCES agent.code_snapshots(id),
        daemon_id text,
        container_name text NOT NULL UNIQUE,
        container_id text,
        state text NOT NULL DEFAULT 'reserved' CHECK(state IN ('reserved','active','closing','cleanup_failed','released')),
        phase text NOT NULL DEFAULT 'reserved' CHECK(phase IN ('reserved','creating','created','starting','started')),
        cleanup_error text,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        released_at timestamptz
    );
    CREATE INDEX code_resource_occupied ON agent.code_execution_resources(run_id) WHERE state<>'released';
    CREATE TABLE agent.code_artifacts (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        snapshot_id uuid REFERENCES agent.code_snapshots(id),
        kind text NOT NULL,
        storage_key text NOT NULL UNIQUE,
        sha256 text NOT NULL,
        size_bytes bigint NOT NULL CHECK(size_bytes>=0),
        metadata jsonb NOT NULL DEFAULT '{}',
        visibility text NOT NULL DEFAULT 'diagnostic' CHECK(visibility IN ('diagnostic','formal')),
        state text NOT NULL DEFAULT 'available' CHECK(state IN ('available','deleting','deleted')),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        deleted_at timestamptz
    );
    CREATE INDEX code_artifacts_task ON agent.code_artifacts(task_id,id);
    CREATE TABLE agent.code_check_runs (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        snapshot_id uuid NOT NULL REFERENCES agent.code_snapshots(id),
        resource_id uuid NOT NULL UNIQUE REFERENCES agent.code_execution_resources(id),
        tool_call_id text NOT NULL,
        check_id text NOT NULL,
        definition jsonb NOT NULL,
        status text NOT NULL DEFAULT 'running' CHECK(status IN ('running','passed','failed','inconclusive')),
        result jsonb,
        result_digest text,
        log_artifact_id uuid REFERENCES agent.code_artifacts(id),
        result_artifact_id uuid REFERENCES agent.code_artifacts(id),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        finished_at timestamptz,
        UNIQUE(run_id,tool_call_id)
    );
    CREATE INDEX code_checks_task ON agent.code_check_runs(task_id,id);
    ALTER TABLE agent.code_tasks ADD FOREIGN KEY(snapshot_id) REFERENCES agent.code_snapshots(id),
        ADD FOREIGN KEY(report_artifact_id) REFERENCES agent.code_artifacts(id),
        ADD FOREIGN KEY(markdown_artifact_id) REFERENCES agent.code_artifacts(id);
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
