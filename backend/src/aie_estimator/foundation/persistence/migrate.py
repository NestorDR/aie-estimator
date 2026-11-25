# src/aie_estimator/foundation/persistence/migrate.py

# --- Python Standard Library ---
from importlib import resources
from typing import Final

# --- Third Party Libraries ---
import psycopg
import structlog

# --- App modules ---
from aie_estimator.logging import configure_logging
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

_MIGRATIONS_PACKAGE: Final[str] = "aie_estimator.foundation.persistence.migrations"

_CREATE_SCHEMA_MIGRATIONS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS public.schema_migrations
(
    version    text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""
_SELECT_APPLIED_MIGRATIONS_SQL: Final[str] = "SELECT version FROM public.schema_migrations;"
_INSERT_MIGRATION_SQL: Final[str] = "INSERT INTO public.schema_migrations (version) VALUES (%s);"


def list_migration_scripts() -> list[tuple[str, str]]:
    """
    Load ordered SQL migration scripts packaged under persistence/migrations.

    :return: List of (version_filename, sql_content) pairs sorted lexicographically by filename.
    """
    package_root = resources.files(_MIGRATIONS_PACKAGE)
    sql_entries = sorted(
        (entry for entry in package_root.iterdir() if entry.is_file() and entry.name.endswith(".sql")),
        key=lambda entry: entry.name,
    )
    return [(entry.name, entry.read_text(encoding="utf-8")) for entry in sql_entries]


def apply_migrations(conn_kwargs: dict[str, str | int]) -> list[str]:
    """
    Apply all pending versioned SQL delta scripts sequentially against PostgreSQL.

    :param conn_kwargs: PostgreSQL connection keyword arguments.

    :return: Ordered list of migration versions applied during this invocation.
    :raises psycopg.Error: If connecting or executing any migration script fails.
    """
    scripts = list_migration_scripts()
    applied_now: list[str] = []

    try:
        with psycopg.connect(**conn_kwargs) as conn:
            with conn.transaction():
                conn.execute(_CREATE_SCHEMA_MIGRATIONS_SQL)
                rows = conn.execute(_SELECT_APPLIED_MIGRATIONS_SQL).fetchall()
                already_applied = {row[0] for row in rows}

            pending = [(version, sql) for version, sql in scripts if version not in already_applied]
            log.info("migration_started", total_scripts=len(scripts), pending_count=len(pending))

            for version, sql in pending:
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(_INSERT_MIGRATION_SQL, (version,))
                applied_now.append(version)
                log.info("migration_applied", version=version)

        log.info("migration_completed", applied_count=len(applied_now))
        return applied_now
    except Exception as e:
        log.exception("migration_failed", error=repr(e), exc_info=e)
        raise


def main() -> None:
    """
    CLI entry point applying pending SQL migrations using application settings.

    :raises ConnectionError: If PostgreSQL credentials are not configured.
    """
    settings = get_settings()
    configure_logging(settings)
    conn_kwargs = settings.get_database_kwargs()
    if conn_kwargs is None:
        raise ConnectionError("PostgreSQL is not configured.")
    apply_migrations(conn_kwargs)


if __name__ == "__main__":
    main()
