"""Persist bounded observation recovery and distinguish cancellation from acknowledgement."""

from alembic import op

revision = "core_0007"
down_revision = "core_0006"


def upgrade():
    op.get_bind().exec_driver_sql("""
    ALTER TABLE agent.background_operations
        ADD COLUMN observation_failures integer NOT NULL DEFAULT 0 CHECK(observation_failures>=0),
        ADD COLUMN observation_first_failed_at timestamptz,
        ADD COLUMN last_observed_at timestamptz,
        ADD COLUMN last_observation_error jsonb,
        ADD COLUMN observation_started_at timestamptz,
        ADD COLUMN observation_inflight boolean NOT NULL DEFAULT false,
        ADD COLUMN remote_poll_interval_ms bigint CHECK(remote_poll_interval_ms>=0),
        ADD COLUMN remote_poll_not_before timestamptz,
        ADD COLUMN cancel_attempts integer NOT NULL DEFAULT 0 CHECK(cancel_attempts BETWEEN 0 AND 3),
        ADD COLUMN cancel_attempt_id uuid,
        ADD COLUMN cancel_attempted_at timestamptz,
        ADD COLUMN cancel_acknowledged_at timestamptz,
        ADD COLUMN cancel_last_error jsonb,
        ADD CONSTRAINT background_observation_error_bounded CHECK(last_observation_error IS NULL OR
            (jsonb_typeof(last_observation_error)='object' AND octet_length(last_observation_error::text)<=2048)),
        ADD CONSTRAINT background_cancel_error_bounded CHECK(cancel_last_error IS NULL OR
            (jsonb_typeof(cancel_last_error)='object' AND octet_length(cancel_last_error::text)<=2048));
    UPDATE agent.background_operations SET cancel_attempts=1,cancel_attempted_at=cancel_sent_at
        WHERE cancel_sent_at IS NOT NULL;
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
