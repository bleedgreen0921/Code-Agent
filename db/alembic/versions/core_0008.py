"""Bounded controller-owned verification evidence; historical Runs stay NULL."""

from alembic import op

revision = "core_0008"
down_revision = "core_0007"


def upgrade():
    op.get_bind().exec_driver_sql("""
        ALTER TABLE agent.runs ADD COLUMN verification jsonb,
        ADD CONSTRAINT runs_verification_bounded CHECK(verification IS NULL OR
            (jsonb_typeof(verification)='object' AND octet_length(verification::text)<=32768))
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
