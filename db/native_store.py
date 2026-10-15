"""Structural verification for the pinned, non-vector native Store migrations."""

import json
from pathlib import Path
from langgraph.store.postgres import PostgresStore

QUERIES = {
    "columns": """SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default,is_identity
        FROM information_schema.columns WHERE table_schema='agent' AND table_name IN ('store','store_migrations') ORDER BY table_name,ordinal_position""",
    "constraints": """SELECT c.relname,con.conname,pg_get_constraintdef(con.oid)
        FROM pg_constraint con JOIN pg_class c ON c.oid=con.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='agent' AND c.relname IN ('store','store_migrations') ORDER BY 1,2""",
    "indexes": """SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='agent'
        AND tablename IN ('store','store_migrations') ORDER BY 1,2""",
}


def signature(conn):
    return {key: conn.execute(query).fetchall() for key, query in QUERIES.items()}


def check_signature(actual, versions, *, complete=False):
    fixtures = json.loads(Path(__file__).with_name("store_structure.json").read_text())
    if versions != list(range(len(versions))) or len(versions) > len(
        PostgresStore.MIGRATIONS
    ):
        raise RuntimeError(
            "Unknown native Store revision rejected without modification"
        )
    current = len(versions) - 1
    allowed = [str(current)]
    if not complete and current < len(PostgresStore.MIGRATIONS) - 1:
        # setup may have executed idempotent DDL before its version insert committed.
        allowed.append(str(current + 1))
    if complete and len(versions) != len(PostgresStore.MIGRATIONS):
        raise RuntimeError("Native Store initialization required")
    normalized = json.loads(json.dumps(actual))
    if not any(normalized == fixtures[key] for key in allowed):
        raise RuntimeError(
            "Native Store structure mismatch rejected without modification"
        )


def validate(conn, *, complete=False):
    existing = conn.execute(
        "SELECT to_regclass('agent.store_migrations'),to_regclass('agent.store')"
    ).fetchone()
    if existing == (None, None) and not complete:
        return
    if existing[0] is None:
        raise RuntimeError(
            "Unknown native Store initialization rejected without modification"
        )
    versions = [
        r[0] for r in conn.execute("SELECT v FROM agent.store_migrations ORDER BY v")
    ]
    check_signature(signature(conn), versions, complete=complete)
    invalid = conn.execute("""SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
        JOIN pg_class t ON t.oid=i.indrelid JOIN pg_namespace n ON n.oid=t.relnamespace
        WHERE n.nspname='agent' AND t.relname IN ('store','store_migrations') AND NOT i.indisvalid""").fetchall()
    if invalid and (complete or invalid != [("store_prefix_idx",)] or 1 in versions):
        raise RuntimeError("Invalid native Store index rejected without modification")
