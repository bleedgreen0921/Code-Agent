"""Runtime bindings and effective execution snapshot."""

import json
from alembic import op

DEFAULT_BINDING = {
    "version": 1,
    "profile": "chat",
    "profile_fingerprint": "8e25c56b8c9813cd4762b40af69185d95886716d2d315d2c11825794fcdcad88",
    "workspace_id": None,
    "workspace_identity": None,
    "fingerprint": "ae78dea4d1e462d7d4ec3bb906ab595c819dc084b5201829457e87f93be0260f",
}

revision = "core_0002"
down_revision = "core_0001"


def upgrade():
    binding = json.dumps(DEFAULT_BINDING, sort_keys=True).replace("'", "''")
    op.get_bind().exec_driver_sql(f"""
        ALTER TABLE agent.conversations
          ADD COLUMN profile text NOT NULL DEFAULT 'chat',
          ADD COLUMN workspace_id text,
          ADD COLUMN workspace_identity text;
        ALTER TABLE agent.runs
          ADD COLUMN profile text NOT NULL DEFAULT 'chat',
          ADD COLUMN workspace_id text,
          ADD COLUMN submitted_binding jsonb NOT NULL DEFAULT '{binding}'::jsonb,
          ADD COLUMN runtime_config jsonb;
        ALTER TABLE agent.runs ALTER COLUMN submitted_binding DROP DEFAULT;
    """)


def downgrade():
    raise RuntimeError("Core downgrades are not supported")
