"""Validate before DDL, then run the single Alembic core migration chain."""

import json
import os
from pathlib import Path

import psycopg
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg.rows import dict_row
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

VERSION = "core_0012"
STORE_TABLES = {"store", "store_migrations"}
CORE_TABLES = {"schema_version", "conversations", "runs", "events", "workers"}
CHECKPOINT_TABLES = {
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}
ROOT = Path(__file__).parent


CORE_QUERIES = {
    "columns": "SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default,is_identity\n            FROM information_schema.columns WHERE table_schema='agent'\n            AND table_name IN ('schema_version','conversations','runs','events','workers','alembic_version','execution_resources','external_operations','tasks','task_dependencies','background_operations','task_events','code_tasks','code_snapshots','code_check_runs','code_execution_resources','code_artifacts','code_review_ranges','code_review_reproductions','code_repair_candidates','code_repair_revisions','code_repair_attempts','code_deliveries','code_workspace_observations','code_integrations','code_delivery_files','code_delivery_events','code_workspace_occupancy')\n            ORDER BY table_name,ordinal_position",
    "constraints": "SELECT c.relname,con.conname,pg_get_constraintdef(con.oid)\n            FROM pg_constraint con JOIN pg_class c ON c.oid=con.conrelid\n            JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='agent'\n            AND c.relname IN ('schema_version','conversations','runs','events','workers','alembic_version','execution_resources','external_operations','tasks','task_dependencies','background_operations','task_events','code_tasks','code_snapshots','code_check_runs','code_execution_resources','code_artifacts','code_review_ranges','code_review_reproductions','code_repair_candidates','code_repair_revisions','code_repair_attempts','code_deliveries','code_workspace_observations','code_integrations','code_delivery_files','code_delivery_events','code_workspace_occupancy')\n            ORDER BY 1,2",
    "indexes": "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='agent'\n            AND tablename IN ('schema_version','conversations','runs','events','workers','alembic_version','execution_resources','external_operations','tasks','task_dependencies','background_operations','task_events','code_tasks','code_snapshots','code_check_runs','code_execution_resources','code_artifacts','code_review_ranges','code_review_reproductions','code_repair_candidates','code_repair_revisions','code_repair_attempts','code_deliveries','code_workspace_observations','code_integrations','code_delivery_files','code_delivery_events','code_workspace_occupancy')\n            ORDER BY 1,2",
}


def structure(conn):
    """Stable structural signature independent of rows, owners and checkpoints."""
    return {key: conn.execute(query).fetchall() for key, query in CORE_QUERIES.items()}


def validate_database(conn):
    schemas = {
        r[0]
        for r in conn.execute(
            "SELECT nspname FROM pg_namespace WHERE nspname NOT LIKE 'pg_%' AND nspname NOT IN ('public','information_schema')"
        )
    }
    rows = conn.execute("""SELECT n.nspname,c.relname,c.relkind FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE c.relkind IN ('r','p','v','m','S','f') AND n.nspname NOT LIKE 'pg_%'
        AND n.nspname <> 'information_schema'""").fetchall()
    unknown_objects = conn.execute("""SELECT
        EXISTS(SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname<>'information_schema') OR
        EXISTS(SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace
            LEFT JOIN pg_class c ON c.oid=t.typrelid
            WHERE n.nspname NOT LIKE 'pg_%' AND n.nspname<>'information_schema'
            AND (t.typtype IN ('d','e','r','m') OR c.relkind='c')) OR
        EXISTS(SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE NOT t.tgisinternal AND n.nspname NOT LIKE 'pg_%'
            AND n.nspname<>'information_schema') OR
        EXISTS(SELECT 1 FROM pg_policies WHERE schemaname NOT LIKE 'pg_%'
            AND schemaname<>'information_schema')
    """).fetchone()[0]
    if unknown_objects:
        raise RuntimeError("Unknown database objects rejected without modification")
    if not schemas and not rows:
        return None
    if schemas != {"agent"} or any(
        s != "agent"
        or (
            t
            not in CORE_TABLES
            | CHECKPOINT_TABLES
            | STORE_TABLES
            | {
                "alembic_version",
                "execution_resources",
                "external_operations",
                "tasks",
                "task_dependencies",
                "background_operations",
                "task_events",
                "code_tasks",
                "code_snapshots",
                "code_check_runs",
                "code_execution_resources",
                "code_artifacts",
                "code_review_ranges",
                "code_review_reproductions",
                "code_repair_candidates",
                "code_repair_revisions",
                "code_repair_attempts",
                "code_deliveries",
                "code_workspace_observations",
                "code_integrations",
                "code_delivery_files",
                "code_delivery_events",
                "code_workspace_occupancy",

            }
            and (t, k) not in (("events_id_seq", "S"), ("task_events_id_seq", "S"))
        )
        or (t not in ("events_id_seq", "task_events_id_seq") and k != "r")
        for s, t, k in rows
    ):
        raise RuntimeError(
            "Historical or unknown database rejected without modification"
        )
    tables = {t for _, t, _ in rows}
    if not CORE_TABLES <= tables:
        raise RuntimeError("Incomplete baseline rejected without modification")
    if conn.execute("SELECT version FROM agent.schema_version").fetchall() != [
        ("agent_core_0001",)
    ]:
        raise RuntimeError("Unknown baseline rejected without modification")
    revision = "legacy"
    if "alembic_version" in tables:
        revisions = conn.execute(
            "SELECT version_num FROM agent.alembic_version"
        ).fetchall()
        if revisions not in [
            [("core_0001",)],
            [("core_0002",)],
            [("core_0003",)],
            [("core_0004",)],
            [("core_0005",)],
            [("core_0006",)],
            [("core_0007",)],
            [("core_0008",)],
            [("core_0009",)],
            [("core_0010",)],
            [("core_0011",)],
            [("core_0012",)],
        ]:
            raise RuntimeError("Unknown revision rejected without modification")
        revision = revisions[0][0]
    expected = json.loads((ROOT / "structure.json").read_text())[revision]
    if json.loads(json.dumps(structure(conn))) != expected:
        raise RuntimeError("Core structure mismatch rejected without modification")
    from db.native_store import validate

    validate(conn)
    return revision


def upgrade(conn, revision):
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    cfg.attributes["connection"] = conn
    if revision is None:
        conn.exec_driver_sql("CREATE SCHEMA agent")
    if revision == "legacy":
        command.stamp(cfg, "core_0001")
    command.upgrade(cfg, "head")


def main():
    # Keep one session (and advisory lock) through validation, upgrade and setup.
    engine = create_engine(
        "postgresql+psycopg://",
        creator=lambda: psycopg.connect(
            os.environ["MIGRATION_DATABASE_URL"], prepare_threshold=None
        ),
        poolclass=NullPool,
    )
    try:
        with engine.connect() as connection:
            conn = connection.connection.driver_connection
            conn.execute("SELECT pg_advisory_lock(81431001)")
            conn.commit()
            try:
                revision = validate_database(conn)
                conn.commit()
                with connection.begin():
                    upgrade(connection, revision)
                conn.autocommit = True
                conn.execute("SET search_path TO agent")
                conn.row_factory = dict_row
                PostgresSaver(conn).setup()
                invalid = conn.execute(
                    """SELECT 1 FROM pg_index WHERE indexrelid=to_regclass('agent.store_prefix_idx') AND NOT indisvalid"""
                ).fetchone()
                if invalid:
                    # Only the exact known interrupted concurrent-index step was
                    # admitted by preflight; repair outside a transaction.
                    conn.execute("DROP INDEX CONCURRENTLY agent.store_prefix_idx")
                PostgresStore(conn, index=None, ttl=None).setup()
            finally:
                conn.rollback()
                conn.execute("SELECT pg_advisory_unlock(81431001)")
                conn.commit()
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
