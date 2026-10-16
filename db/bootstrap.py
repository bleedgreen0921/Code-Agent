"""Grant DML to a dedicated runtime login, without schema or migration rights."""

import os
from urllib.parse import unquote, urlsplit

import psycopg
from psycopg import sql

from db.migrate import validate_database


def main():
    runtime = urlsplit(os.environ["AGENT_DATABASE_URL"])
    role, password = unquote(runtime.username or ""), unquote(runtime.password or "")
    if not role or not password:
        raise ValueError(
            "AGENT_DATABASE_URL requires a dedicated username and password"
        )
    with psycopg.connect(os.environ["MIGRATION_DATABASE_URL"]) as conn:
        if validate_database(conn) != "core_0007":
            raise RuntimeError("Run db.migrate before bootstrap")
        owner = conn.execute("SELECT current_user").fetchone()[0]
        if role == owner:
            raise ValueError("Runtime login must differ from the migration owner")
        existing = conn.execute(
            "SELECT rolsuper,rolcreatedb,rolcreaterole,rolbypassrls FROM pg_roles WHERE rolname=%s",
            (role,),
        ).fetchone()
        if existing:
            if (
                any(existing)
                or conn.execute(
                    "SELECT 1 FROM pg_auth_members WHERE member=(SELECT oid FROM pg_roles WHERE rolname=%s)",
                    (role,),
                ).fetchone()
            ):
                raise ValueError(
                    "Existing runtime login has elevated rights or memberships"
                )
            conn.execute(
                sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
        else:
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOREPLICATION PASSWORD {}"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        conn.execute(
            sql.SQL("REVOKE CREATE,TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                sql.Identifier(conn.info.dbname)
            )
        )
        conn.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(conn.info.dbname), sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA agent TO {}").format(sql.Identifier(role))
        )
        tables = (
            "conversations",
            "runs",
            "events",
            "workers",
            "execution_resources",
            "external_operations",
            "tasks",
            "task_dependencies",
            "background_operations",
            "task_events",
            "store",
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
        )
        for table in tables:
            conn.execute(
                sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON agent.{} TO {}").format(
                    sql.Identifier(table), sql.Identifier(role)
                )
            )
        conn.execute(
            sql.SQL("GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA agent TO {}").format(
                sql.Identifier(role)
            )
        )
        conn.execute(
            sql.SQL(
                "REVOKE ALL ON agent.alembic_version,agent.schema_version,agent.checkpoint_migrations,agent.store_migrations FROM PUBLIC, {}"
            ).format(sql.Identifier(role))
        )
        # Startup checks may read these tables; runtime cannot update either revision.
        conn.execute(
            sql.SQL(
                "GRANT SELECT ON agent.schema_version,agent.alembic_version,agent.checkpoint_migrations,agent.store_migrations TO {}"
            ).format(sql.Identifier(role))
        )


if __name__ == "__main__":
    main()
