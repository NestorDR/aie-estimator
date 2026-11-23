# tests/persistence/test_migrate.py

# --- Python Standard Library ---
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.persistence.migrate import apply_migrations, list_migration_scripts, main
from aie_estimator.settings import Environment, Settings


def test_list_migration_scripts_includes_initial_schema_without_connect_command() -> None:
    """
    GIVEN packaged SQL scripts under persistence/migrations
    WHEN list_migration_scripts is called
    THEN 0001_initial_schema.sql is discovered with pure SQL free of psql \\connect commands.
    """
    scripts = list_migration_scripts()
    versions = [version for version, _ in scripts]
    assert "0001_initial_schema.sql" in versions

    initial_sql = dict(scripts)["0001_initial_schema.sql"]
    assert "ingestion_jobs" in initial_sql
    assert "pseudonym_mappings" in initial_sql
    assert "\\connect" not in initial_sql


def test_apply_migrations_executes_only_pending_versions() -> None:
    """
    GIVEN two migration scripts where the first is already recorded in schema_migrations
    WHEN apply_migrations is invoked
    THEN only the pending script is executed and recorded.
    """
    mock_conn = MagicMock()
    mock_conn.transaction.side_effect = lambda: nullcontext()
    mock_conn.execute.return_value.fetchall.return_value = [("0001_initial_schema.sql",)]

    mock_connect_cm = MagicMock()
    mock_connect_cm.__enter__.return_value = mock_conn
    mock_connect_cm.__exit__.return_value = False

    scripts = [
        ("0001_initial_schema.sql", "CREATE TABLE IF NOT EXISTS t1 ();"),
        ("0002_next.sql", "CREATE TABLE IF NOT EXISTS t2 ();"),
    ]

    with (
        patch("aie_estimator.persistence.migrate.list_migration_scripts", return_value=scripts),
        patch("aie_estimator.persistence.migrate.psycopg.connect", return_value=mock_connect_cm),
    ):
        applied = apply_migrations({"host": "localhost", "port": 5432, "user": "u", "password": "p", "dbname": "db"})  # noqa: S106

    assert applied == ["0002_next.sql"]
    executed_sqls = [call.args[0] for call in mock_conn.execute.call_args_list]
    assert "CREATE TABLE IF NOT EXISTS t2 ();" in executed_sqls
    assert "CREATE TABLE IF NOT EXISTS t1 ();" not in executed_sqls


def test_apply_migrations_propagates_sql_error() -> None:
    """
    GIVEN a migration script execution that raises a database error
    WHEN apply_migrations is invoked
    THEN the exception is logged and re-raised.
    """
    mock_conn = MagicMock()
    mock_conn.transaction.side_effect = lambda: nullcontext()
    mock_conn.execute.side_effect = [MagicMock(), MagicMock(fetchall=lambda: []), RuntimeError("SQL syntax error")]

    mock_connect_cm = MagicMock()
    mock_connect_cm.__enter__.return_value = mock_conn
    mock_connect_cm.__exit__.return_value = False

    with (
        patch(
            "aie_estimator.persistence.migrate.list_migration_scripts",
            return_value=[("0001_initial_schema.sql", "BAD SQL;")],
        ),
        patch("aie_estimator.persistence.migrate.psycopg.connect", return_value=mock_connect_cm),
        pytest.raises(RuntimeError, match=r"SQL syntax error"),
    ):
        apply_migrations({"host": "localhost", "port": 5432, "user": "u", "password": "p", "dbname": "db"})  # noqa: S106


def test_migrate_main_raises_when_persistence_unconfigured() -> None:
    """
    GIVEN Settings with postgres_password set to None
    WHEN migrate.main is invoked
    THEN a ConnectionError is raised.
    """
    settings = Settings(app_env=Environment.TEST, postgres_password=None)
    with (
        patch("aie_estimator.persistence.migrate.get_settings", return_value=settings),
        patch("aie_estimator.persistence.migrate.configure_logging"),
        pytest.raises(ConnectionError, match=r"not configured"),
    ):
        main()


def test_migrate_main_invokes_apply_migrations() -> None:
    """
    GIVEN Settings with discrete PostgreSQL credentials
    WHEN migrate.main is invoked
    THEN apply_migrations is called with the connection kwargs dictionary.
    """
    settings = Settings(
        app_env=Environment.TEST,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="estimator",
        postgres_password=SecretStr("secret"),
        postgres_db="estimator",
    )
    with (
        patch("aie_estimator.persistence.migrate.get_settings", return_value=settings),
        patch("aie_estimator.persistence.migrate.configure_logging"),
        patch("aie_estimator.persistence.migrate.apply_migrations", return_value=[]) as mock_apply,
    ):
        main()

    mock_apply.assert_called_once_with(
        {
            "host": "localhost",
            "port": 5432,
            "user": "estimator",
            "password": "secret",
            "dbname": "estimator",
        }
    )
