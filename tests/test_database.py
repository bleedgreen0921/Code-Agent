import os
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from db import bootstrap, migrate
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo


def test_new_baseline_is_idempotent_and_runtime_has_no_migration_rights(database):
    migrate.main()
    migrate.main()
    bootstrap.main()
    with psycopg.connect(os.environ["AGENT_DATABASE_URL"], autocommit=True) as conn:
        assert not conn.execute(
            "SELECT has_schema_privilege(current_user,'agent','CREATE')"
        ).fetchone()[0]
        for statement in (
            "CREATE TABLE agent.denied(id int)",
            "CREATE SCHEMA denied",
            "UPDATE agent.schema_version SET version='bad'",
            "UPDATE agent.alembic_version SET version_num='bad'",
            "DELETE FROM agent.alembic_version",
            "INSERT INTO agent.alembic_version VALUES('bad')",
            "TRUNCATE agent.alembic_version",
            "DELETE FROM agent.checkpoint_migrations",
            "ALTER TABLE agent.runs ADD COLUMN denied int",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)


def test_historical_and_unknown_databases_are_rejected_without_modification(database):
    dsn = os.environ["MIGRATION_DATABASE_URL"]

    def snapshot(conn):
        return conn.execute("""SELECT n.nspname,c.relname,c.relkind,c.relfilenode FROM pg_class c
            JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname NOT LIKE 'pg_%'
            AND n.nspname<>'information_schema' ORDER BY 1,2""").fetchall()

    # Dedicated, generated test DBs only. Never inspect or modify a real historical DB.
    for fixture in (
        "CREATE SCHEMA execution; CREATE SCHEMA rag; CREATE SCHEMA agent; CREATE TABLE agent.alembic_version(version_num text); INSERT INTO agent.alembic_version VALUES ('agent_personal_0006')",
        "CREATE TABLE public.unrelated(id int); INSERT INTO public.unrelated VALUES(42)",
        "CREATE SCHEMA identity",
        "CREATE FUNCTION public.unknown() RETURNS int LANGUAGE sql AS 'SELECT 1'",
        "CREATE TYPE public.unknown AS ENUM ('one','two')",
    ):
        name = "agent_test_legacy_" + uuid4().hex
        with psycopg.connect(dsn, autocommit=True) as owner:
            owner.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        values = conninfo_to_dict(dsn)
        values["dbname"] = name
        legacy = make_conninfo(**values)
        try:
            with psycopg.connect(legacy, autocommit=True) as conn:
                conn.execute(fixture)
                before = snapshot(conn)
                with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": legacy}):
                    with pytest.raises(RuntimeError, match="rejected"):
                        migrate.main()
                    with pytest.raises(RuntimeError, match="rejected"):
                        bootstrap.main()
                assert before == snapshot(conn)
                assert (
                    conn.execute(
                        "SELECT to_regclass('agent.schema_version')"
                    ).fetchone()[0]
                    is None
                )
        finally:
            with psycopg.connect(dsn, autocommit=True) as owner:
                owner.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def test_current_baseline_upgrade_preserves_history_queue_and_checkpoints(database):
    import json
    from pathlib import Path
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row, tuple_row
    from importlib import import_module

    DEFAULT_BINDING = import_module("db.alembic.versions.core_0002").DEFAULT_BINDING

    dsn = os.environ["MIGRATION_DATABASE_URL"]
    name = "agent_test_upgrade_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as owner:
        owner.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        expected = json.loads(json.dumps(migrate.structure(owner)))
    legacy = make_conninfo(**{**conninfo_to_dict(dsn), "dbname": name})
    try:
        with psycopg.connect(legacy, autocommit=True) as conn:
            conn.execute(
                Path("db/alembic/versions/core_0001.sql").read_text(), prepare=False
            )
            conversation, completed, queued = uuid4(), uuid4(), uuid4()
            conn.execute(
                "INSERT INTO agent.conversations(id,title) VALUES(%s,%s)",
                (conversation, "kept"),
            )
            conn.execute(
                """INSERT INTO agent.runs(id,task,conversation_id,turn_no,idempotency_scope,status,queue_deadline,answer,final_checkpoint)
                VALUES(%s,'history',%s,1,'history','completed',clock_timestamp(),'kept answer','{"configurable":{"thread_id":"old"}}')""",
                (completed, conversation),
            )
            conn.execute(
                "UPDATE agent.conversations SET last_success_run_id=%s WHERE id=%s",
                (completed, conversation),
            )
            conn.execute(
                "INSERT INTO agent.runs(id,task,idempotency_scope,status,queue_deadline) VALUES(%s,'queued','standalone','queued',clock_timestamp()+interval '1 hour')",
                (queued,),
            )
            # A malformed legacy baseline must not even create alembic_version.
            conn.execute("ALTER TABLE agent.runs ALTER COLUMN task DROP NOT NULL")
            before = json.dumps(migrate.structure(conn))
            with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": legacy}):
                with pytest.raises(RuntimeError, match="rejected"):
                    migrate.main()
            assert json.dumps(migrate.structure(conn)) == before
            assert conn.execute(
                "SELECT to_regclass('agent.alembic_version')"
            ).fetchone() == (None,)
            conn.execute("ALTER TABLE agent.runs ALTER COLUMN task SET NOT NULL")
            conn.execute("SET search_path TO agent")
            conn.row_factory = dict_row
            saver = PostgresSaver(conn)
            saver.setup()
            from langgraph.checkpoint.base import empty_checkpoint

            cp = empty_checkpoint()
            cp["channel_values"] = {"messages": []}
            saved = saver.put(
                {"configurable": {"thread_id": "old", "checkpoint_ns": ""}},
                cp,
                {"source": "input", "step": 0, "parents": {}},
                {},
            )
            conn.row_factory = tuple_row
        with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": legacy}):
            migrate.main()
            migrate.main()
        with psycopg.connect(legacy, autocommit=True) as conn:
            assert json.loads(json.dumps(migrate.structure(conn))) == expected
            assert conn.execute(
                "SELECT answer,profile,workspace_id,submitted_binding,runtime_config FROM agent.runs WHERE id=%s",
                (completed,),
            ).fetchone() == ("kept answer", "chat", None, DEFAULT_BINDING, None)
            assert conn.execute(
                "SELECT status,submitted_binding FROM agent.runs WHERE id=%s", (queued,)
            ).fetchone() == ("queued", DEFAULT_BINDING)
            assert conn.execute(
                "SELECT last_success_run_id,profile FROM agent.conversations"
            ).fetchone() == (completed, "chat")
            assert conn.execute(
                "SELECT version_num FROM agent.alembic_version"
            ).fetchone() == ("core_0011",)
            conn.execute("SET search_path TO agent")
            conn.row_factory = dict_row
            assert PostgresSaver(conn).get_tuple(saved).checkpoint["id"] == cp["id"]
            conn.row_factory = tuple_row
            # Known revision with a subtly altered structure must be rejected before DDL.
            conn.execute("ALTER TABLE agent.runs ALTER COLUMN task DROP NOT NULL")
            before = json.dumps(migrate.structure(conn))
            with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": legacy}):
                with pytest.raises(RuntimeError, match="rejected"):
                    migrate.main()
            assert json.dumps(migrate.structure(conn)) == before
    finally:
        with psycopg.connect(dsn, autocommit=True) as owner:
            owner.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


@pytest.mark.parametrize(
    "prior_revision",
    ["core_0002", "core_0003", "core_0004", "core_0005", "core_0006", "core_0007"],
)
def test_core_0002_upgrade_keeps_queued_binding_and_grants_execution_table(
    database, prior_revision
):
    from alembic.config import Config
    from alembic import command
    from pathlib import Path
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    from agent_service.profiles import DEFAULT_BINDING

    dsn = os.environ["MIGRATION_DATABASE_URL"]
    name = "agent_test_v2_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    prior = make_conninfo(**{**conninfo_to_dict(dsn), "dbname": name})
    try:
        engine = create_engine(
            "postgresql+psycopg://",
            creator=lambda: psycopg.connect(prior),
            poolclass=NullPool,
        )
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql("CREATE SCHEMA agent")
                cfg = Config()
                cfg.set_main_option(
                    "script_location", str(Path("db/alembic").resolve())
                )
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, prior_revision)
        finally:
            engine.dispose()
        run_id = uuid4()
        with psycopg.connect(prior, autocommit=True) as conn:
            from psycopg.types.json import Jsonb

            conn.execute(
                """INSERT INTO agent.runs(id,task,idempotency_scope,status,queue_deadline,submitted_binding)
                VALUES(%s,'kept','standalone','queued',clock_timestamp()+interval '1 hour',%s)""",
                (run_id, Jsonb(DEFAULT_BINDING)),
            )
        with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": prior}):
            migrate.main()
            migrate.main()
        with psycopg.connect(prior) as conn:
            assert migrate.validate_database(conn) == "core_0011"
            assert conn.execute(
                "SELECT status,submitted_binding FROM agent.runs WHERE id=%s", (run_id,)
            ).fetchone() == ("queued", DEFAULT_BINDING)
            assert conn.execute(
                "SELECT verification FROM agent.runs WHERE id=%s", (run_id,)
            ).fetchone() == (None,)
        with psycopg.connect(os.environ["AGENT_DATABASE_URL"]) as conn:
            assert conn.execute(
                "SELECT has_table_privilege(current_user,'agent.execution_resources','SELECT,INSERT,UPDATE,DELETE')"
            ).fetchone()[0]
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def test_native_store_runtime_migration_privileges_and_schema_drift(database):
    from db.native_store import validate

    with psycopg.connect(os.environ["AGENT_DATABASE_URL"], autocommit=True) as conn:
        for statement in (
            "DELETE FROM agent.store_migrations",
            "INSERT INTO agent.store_migrations VALUES(99)",
            "ALTER TABLE agent.store ADD COLUMN denied int",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)
    with psycopg.connect(os.environ["MIGRATION_DATABASE_URL"], autocommit=True) as conn:
        validate(conn, complete=True)
        try:
            conn.execute("ALTER TABLE agent.store ALTER COLUMN value DROP NOT NULL")
            with pytest.raises(RuntimeError, match="rejected"):
                migrate.main()
        finally:
            conn.execute("ALTER TABLE agent.store ALTER COLUMN value SET NOT NULL")
        try:
            conn.execute("INSERT INTO agent.store_migrations VALUES(99)")
            with pytest.raises(RuntimeError, match="revision"):
                migrate.main()
        finally:
            conn.execute("DELETE FROM agent.store_migrations WHERE v=99")


def test_native_store_known_partial_setup_is_idempotent(database):
    import json
    from pathlib import Path
    from langgraph.store.postgres import PostgresStore

    dsn = os.environ["MIGRATION_DATABASE_URL"]
    name = "agent_test_store_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as owner:
        owner.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    target = make_conninfo(**{**conninfo_to_dict(dsn), "dbname": name})
    try:
        # Upgrade core schema first and mimic a Store setup interruption after
        # each idempotent DDL but before its version record was committed.
        with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": target}):
            migrate.main()
        with psycopg.connect(target, autocommit=True) as conn:
            conn.execute("SET search_path TO agent")
            for stop in range(len(PostgresStore.MIGRATIONS)):
                conn.execute("SET search_path TO agent")
                conn.execute("DROP TABLE agent.store,agent.store_migrations")
                conn.execute(
                    "CREATE TABLE agent.store_migrations(v integer PRIMARY KEY)"
                )
                for version, statement in enumerate(
                    PostgresStore.MIGRATIONS[: stop + 1]
                ):
                    conn.execute(statement)
                    if version < stop:
                        conn.execute(
                            "INSERT INTO agent.store_migrations VALUES(%s)", (version,)
                        )
                if stop == 1:
                    conn.execute(
                        "UPDATE pg_index SET indisvalid=false WHERE indexrelid='agent.store_prefix_idx'::regclass"
                    )
                with patch.dict(os.environ, {"MIGRATION_DATABASE_URL": target}):
                    migrate.main()
                    migrate.main()
                assert conn.execute(
                    "SELECT v FROM agent.store_migrations ORDER BY v"
                ).fetchall() == [(v,) for v in range(len(PostgresStore.MIGRATIONS))]
                conn.execute("SET search_path TO public")
                assert (
                    json.loads(json.dumps(migrate.structure(conn)))
                    == json.loads(Path("db/structure.json").read_text())["core_0011"]
                )
    finally:
        with psycopg.connect(dsn, autocommit=True) as owner:
            owner.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
