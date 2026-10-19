"""Isolated candidate repairs, immutable revisions and in-graph gate attempts."""

from alembic import op

revision = "core_0011"
down_revision = "core_0010"


def upgrade():
    op.get_bind().exec_driver_sql("""
    ALTER TABLE agent.code_tasks DROP CONSTRAINT code_tasks_stage_check;
    ALTER TABLE agent.code_tasks ADD CONSTRAINT code_tasks_stage_check CHECK(stage IN ('analysis','review','repair'));
    ALTER TABLE agent.code_tasks ADD COLUMN diagnostic_artifact_id uuid REFERENCES agent.code_artifacts(id);
    ALTER TABLE agent.runs ADD UNIQUE(id,task_id);
    ALTER TABLE agent.code_snapshots ADD UNIQUE(id,run_id), ADD UNIQUE(id,run_id,task_id),
        ADD FOREIGN KEY(run_id,task_id) REFERENCES agent.runs(id,task_id);
    CREATE TABLE agent.code_repair_candidates (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL UNIQUE REFERENCES agent.runs(id),
        baseline_id uuid NOT NULL,
        storage_key text NOT NULL UNIQUE,
        acceptance jsonb NOT NULL,
        policy_digest text NOT NULL,
        current_revision_id uuid,
        frozen_revision_id uuid,
        status text NOT NULL DEFAULT 'preparing' CHECK(status IN ('preparing','writable','passed','published','failed','cancelled','deleted')),
        admitted_bytes bigint NOT NULL DEFAULT 0 CHECK(admitted_bytes>=0),
        usage jsonb NOT NULL DEFAULT '{}',
        final_checkpoint jsonb,
        report_digest text,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        deleted_at timestamptz,
        UNIQUE(id,run_id), UNIQUE(id,run_id,task_id),
        CHECK(status NOT IN ('passed','published') OR (frozen_revision_id IS NOT NULL AND report_digest IS NOT NULL)),
        FOREIGN KEY(run_id,task_id) REFERENCES agent.runs(id,task_id),
        FOREIGN KEY(baseline_id,run_id,task_id) REFERENCES agent.code_snapshots(id,run_id,task_id)
    );
    CREATE TABLE agent.code_repair_revisions (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        candidate_id uuid NOT NULL,
        tool_call_id text NOT NULL,
        storage_key text NOT NULL UNIQUE,
        content_digest text NOT NULL,
        manifest_artifact_id uuid NOT NULL REFERENCES agent.code_artifacts(id),
        patch_artifact_id uuid NOT NULL REFERENCES agent.code_artifacts(id),
        package_artifact_id uuid NOT NULL REFERENCES agent.code_artifacts(id),
        size_bytes bigint NOT NULL CHECK(size_bytes>=0),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        UNIQUE(run_id,tool_call_id),
        UNIQUE(id,candidate_id,run_id),
        FOREIGN KEY(candidate_id,run_id,task_id) REFERENCES agent.code_repair_candidates(id,run_id,task_id)
    );
    ALTER TABLE agent.code_repair_candidates
        ADD FOREIGN KEY(current_revision_id,id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id),
        ADD FOREIGN KEY(frozen_revision_id,id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id);
    CREATE TABLE agent.code_repair_attempts (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        candidate_id uuid NOT NULL,
        message_id text NOT NULL,
        draft_digest text NOT NULL,
        draft jsonb NOT NULL,
        revision_id uuid,
        status text NOT NULL DEFAULT 'running' CHECK(status IN ('running','passed','failed','inconclusive')),
        check_ids jsonb NOT NULL DEFAULT '[]',
        feedback jsonb,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        finished_at timestamptz,
        UNIQUE(run_id,message_id),
        FOREIGN KEY(candidate_id,run_id,task_id) REFERENCES agent.code_repair_candidates(id,run_id,task_id),
        FOREIGN KEY(revision_id,candidate_id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id)
    );
    ALTER TABLE agent.code_check_runs ADD COLUMN candidate_id uuid,
        ADD COLUMN revision_id uuid,
        ADD COLUMN purpose text NOT NULL DEFAULT 'development' CHECK(purpose IN ('development','acceptance','baseline')),
        ADD COLUMN acceptance_digest text,
        ADD FOREIGN KEY(snapshot_id,run_id) REFERENCES agent.code_snapshots(id,run_id),
        ADD FOREIGN KEY(candidate_id,run_id,task_id) REFERENCES agent.code_repair_candidates(id,run_id,task_id),
        ADD FOREIGN KEY(candidate_id,run_id) REFERENCES agent.code_repair_candidates(id,run_id),
        ADD FOREIGN KEY(revision_id,candidate_id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id),
        ADD CHECK((candidate_id IS NULL)=(revision_id IS NULL)),
        ADD CHECK((purpose='development' AND acceptance_digest IS NULL) OR (purpose IN ('acceptance','baseline') AND acceptance_digest IS NOT NULL));
    ALTER TABLE agent.execution_resources ADD COLUMN candidate_id uuid,
        ADD COLUMN revision_id uuid,
        ADD FOREIGN KEY(snapshot_id,run_id) REFERENCES agent.code_snapshots(id,run_id),
        ADD FOREIGN KEY(candidate_id,run_id) REFERENCES agent.code_repair_candidates(id,run_id),
        ADD FOREIGN KEY(revision_id,candidate_id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id);
    ALTER TABLE agent.code_execution_resources ADD COLUMN candidate_id uuid,
        ADD COLUMN revision_id uuid,
        ADD FOREIGN KEY(snapshot_id,run_id) REFERENCES agent.code_snapshots(id,run_id),
        ADD FOREIGN KEY(candidate_id,run_id) REFERENCES agent.code_repair_candidates(id,run_id),
        ADD FOREIGN KEY(revision_id,candidate_id,run_id) REFERENCES agent.code_repair_revisions(id,candidate_id,run_id);
    CREATE INDEX code_repair_revisions_task ON agent.code_repair_revisions(task_id,id);
    CREATE INDEX code_repair_attempts_task ON agent.code_repair_attempts(task_id,id);
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
