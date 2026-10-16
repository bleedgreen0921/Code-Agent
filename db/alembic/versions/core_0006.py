"""Bounded failure facts; historical records deliberately retain SQL NULL."""

from alembic import op

revision = "core_0006"
down_revision = "core_0005"


def upgrade():
    for table in ("runs", "background_operations", "external_operations"):
        op.get_bind().exec_driver_sql(f"""
            ALTER TABLE agent.{table} ADD COLUMN failure_info jsonb;
            ALTER TABLE agent.{table} ADD CONSTRAINT {table}_failure_info_bounded
            CHECK (failure_info IS NULL OR
                (jsonb_typeof(failure_info)='object' AND octet_length(failure_info::text)<=2048));
        """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
