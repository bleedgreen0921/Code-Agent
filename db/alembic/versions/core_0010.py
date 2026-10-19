"""Fixed Review ranges and journalled head-only reproductions."""

from alembic import op

revision = "core_0010"
down_revision = "core_0009"


def upgrade():
    op.get_bind().exec_driver_sql("""
    ALTER TABLE agent.code_tasks DROP CONSTRAINT code_tasks_stage_check;
    ALTER TABLE agent.code_tasks ADD CONSTRAINT code_tasks_stage_check CHECK(stage IN ('analysis','review'));
    CREATE TABLE agent.code_review_ranges (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL UNIQUE REFERENCES agent.runs(id),
        snapshot_id uuid NOT NULL UNIQUE REFERENCES agent.code_snapshots(id),
        metadata jsonb NOT NULL,
        changes_artifact_id uuid NOT NULL REFERENCES agent.code_artifacts(id),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE TABLE agent.code_review_reproductions (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        snapshot_id uuid NOT NULL REFERENCES agent.code_snapshots(id),
        resource_id uuid NOT NULL UNIQUE REFERENCES agent.code_execution_resources(id),
        tool_call_id text NOT NULL,
        hypothesis text NOT NULL,
        script_artifact_id uuid NOT NULL REFERENCES agent.code_artifacts(id),
        script_digest text NOT NULL,
        runner_version text NOT NULL,
        content_digest text NOT NULL,
        status text NOT NULL DEFAULT 'running' CHECK(status IN ('running','observed','not_observed','inconclusive')),
        result jsonb,
        result_digest text,
        log_artifact_id uuid REFERENCES agent.code_artifacts(id),
        result_artifact_id uuid REFERENCES agent.code_artifacts(id),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        finished_at timestamptz,
        UNIQUE(run_id,tool_call_id)
    );
    CREATE INDEX code_reproductions_task ON agent.code_review_reproductions(task_id,id);
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
