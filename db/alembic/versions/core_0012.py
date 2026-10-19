"""Automatic delivery, persistent handoff, observations and file operation journal."""

from alembic import op

revision = "core_0012"
down_revision = "core_0011"


def upgrade():
    op.get_bind().exec_driver_sql("""
    ALTER TABLE agent.tasks DROP CONSTRAINT tasks_status_check;
    ALTER TABLE agent.tasks ADD CONSTRAINT tasks_status_check CHECK(status IN
        ('waiting_dependencies','queued','running','delivering','waiting_background','cancelling','completed','failed','cancelled'));
    ALTER TABLE agent.runs DROP CONSTRAINT runs_status_check;
    ALTER TABLE agent.runs ADD CONSTRAINT runs_status_check CHECK(status IN
        ('queued','running','delivering','cancelling','completed','failed','cancelled'));
    DROP INDEX agent.one_active_conversation_run;
    CREATE UNIQUE INDEX one_active_conversation_run ON agent.runs(conversation_id)
        WHERE conversation_id IS NOT NULL AND status IN ('queued','running','delivering','cancelling');
    DROP INDEX agent.runs_task_active;
    CREATE UNIQUE INDEX runs_task_active ON agent.runs(task_id)
        WHERE task_id IS NOT NULL AND status IN ('queued','running','delivering','cancelling');
    CREATE TABLE agent.code_deliveries (
        id uuid PRIMARY KEY,
        task_id uuid NOT NULL UNIQUE REFERENCES agent.code_tasks(task_id),
        run_id uuid NOT NULL UNIQUE REFERENCES agent.runs(id),
        workspace_root text NOT NULL,
        root_identity jsonb NOT NULL,
        mode text NOT NULL DEFAULT 'auto_apply' CHECK(mode='auto_apply'),
        status text NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','applying','applied','rolled_back','needs_recovery')),
        impact text NOT NULL DEFAULT 'none' CHECK(impact IN ('none','partial','all','unknown')),
        phase text NOT NULL DEFAULT 'preparing',
        protocol integer NOT NULL DEFAULT 1 CHECK(protocol=1),
        plan jsonb,
        plan_digest text,
        checkpoint jsonb,
        final_checkpoint jsonb,
        publication jsonb,
        usage jsonb NOT NULL DEFAULT '{}',
        refreshes integer NOT NULL DEFAULT 0 CHECK(refreshes BETWEEN 0 AND 3),
        observation_id uuid,
        integration_id uuid,
        integrated_observation_id uuid,
        protected jsonb NOT NULL DEFAULT '{}',
        handoffs jsonb NOT NULL DEFAULT '[]',
        deadline_at timestamptz,
        lease_token uuid,
        lease_until timestamptz,
        cancel_requested boolean NOT NULL DEFAULT false,
        rollback_requested boolean NOT NULL DEFAULT false,
        diagnostic jsonb NOT NULL DEFAULT '{}',
        admitted_bytes bigint NOT NULL DEFAULT 0 CHECK(admitted_bytes>=0),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        applied_at timestamptz,
        UNIQUE(id,run_id,task_id),
        FOREIGN KEY(run_id,task_id) REFERENCES agent.runs(id,task_id),
        CHECK(status<>'applied' OR (impact IN ('none','all') AND (impact='all' OR (plan->'files'='[]'::jsonb AND plan->'directories'='[]'::jsonb)) AND plan IS NOT NULL AND checkpoint IS NOT NULL))
    );
    CREATE TABLE agent.code_workspace_observations (
        id uuid PRIMARY KEY,
        delivery_id uuid NOT NULL REFERENCES agent.code_deliveries(id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        storage_key text NOT NULL UNIQUE,
        document jsonb NOT NULL,
        size_bytes bigint NOT NULL CHECK(size_bytes>=0),
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        FOREIGN KEY(delivery_id,run_id,task_id) REFERENCES agent.code_deliveries(id,run_id,task_id)
    );
    CREATE TABLE agent.code_integrations (
        id uuid PRIMARY KEY,
        delivery_id uuid NOT NULL REFERENCES agent.code_deliveries(id),
        run_id uuid NOT NULL REFERENCES agent.runs(id),
        task_id uuid NOT NULL REFERENCES agent.code_tasks(task_id),
        base_observation_id uuid REFERENCES agent.code_workspace_observations(id),
        observation_id uuid NOT NULL REFERENCES agent.code_workspace_observations(id),
        candidate_revision_id uuid NOT NULL REFERENCES agent.code_repair_revisions(id),
        final_revision_id uuid REFERENCES agent.code_repair_revisions(id),
        status text NOT NULL CHECK(status IN ('preparing','conflict','checking','passed','failed','installing','installed')),
        evidence jsonb NOT NULL,
        check_ids jsonb NOT NULL DEFAULT '[]',
        created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
        FOREIGN KEY(delivery_id,run_id,task_id) REFERENCES agent.code_deliveries(id,run_id,task_id)
    );
    ALTER TABLE agent.code_deliveries
        ADD FOREIGN KEY(observation_id) REFERENCES agent.code_workspace_observations(id),
        ADD FOREIGN KEY(integrated_observation_id) REFERENCES agent.code_workspace_observations(id),
        ADD FOREIGN KEY(integration_id) REFERENCES agent.code_integrations(id);
    CREATE TABLE agent.code_delivery_files (
        delivery_id uuid NOT NULL REFERENCES agent.code_deliveries(id),
        plan_id uuid NOT NULL,
        ordinal integer NOT NULL CHECK(ordinal>=0),
        path text NOT NULL,
        operation jsonb NOT NULL,
        phase text NOT NULL DEFAULT 'staged',
        actual jsonb NOT NULL DEFAULT '{}',
        PRIMARY KEY(delivery_id,plan_id,ordinal),
        UNIQUE(delivery_id,plan_id,path)
    );
    CREATE TABLE agent.code_delivery_events (
        id uuid PRIMARY KEY,
        delivery_id uuid NOT NULL REFERENCES agent.code_deliveries(id),
        plan_id uuid,
        ordinal integer,
        phase text NOT NULL,
        detail jsonb NOT NULL,
        created_at timestamptz NOT NULL DEFAULT clock_timestamp()
    );
    CREATE INDEX code_delivery_events_order ON agent.code_delivery_events(delivery_id,created_at,id);
    CREATE TABLE agent.code_workspace_occupancy (
        delivery_id uuid PRIMARY KEY REFERENCES agent.code_deliveries(id),
        run_id uuid NOT NULL UNIQUE REFERENCES agent.runs(id),
        task_id uuid NOT NULL UNIQUE REFERENCES agent.tasks(id),
        workspace_root text NOT NULL,
        released_at timestamptz
    );
    ALTER TABLE agent.code_check_runs ADD COLUMN integration_id uuid REFERENCES agent.code_integrations(id);
    ALTER TABLE agent.execution_resources ADD COLUMN generation integer NOT NULL DEFAULT 0 CHECK(generation>=0);
    ALTER TABLE agent.code_execution_resources ADD COLUMN generation integer NOT NULL DEFAULT 0 CHECK(generation>=0);
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
