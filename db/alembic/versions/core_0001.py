"""Existing core baseline."""

from pathlib import Path
from alembic import op

revision = "core_0001"
down_revision = None


def upgrade():
    op.get_bind().exec_driver_sql(Path(__file__).with_suffix(".sql").read_text())


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
