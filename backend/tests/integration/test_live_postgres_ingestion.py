# tests/integration/test_live_postgres_ingestion.py

"""
Opt-in live PostgreSQL end-to-end integration test for Session 6.

Verifies idempotent SQL migrations (0001 and 0002), FastAPI ingestion job
lifecycle (POST /runs -> GET /jobs/{job_id}), quarantine upserts, hashed
pseudonym mapping persistence with source_names provenance, and GDPR Article 17
CLI erasure against a running estimator-postgres container.
"""

# --- Python Standard Library ---
import asyncio
from datetime import datetime
import sys
from uuid import UUID

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import psycopg
import pytest

# --- App modules ---
from aie_estimator.ingestion.pii.forget import main as forget_main
from aie_estimator.ingestion.pii.pseudonymizer import compute_pii_hash, resolve_pii_salt
from aie_estimator.foundation.persistence.migrate import apply_migrations
from aie_estimator.server import create_app
from aie_estimator.settings import Environment, Settings

pytestmark = pytest.mark.live_db


def test_live_postgres_migrations_ingestion_and_forget_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    GIVEN a running PostgreSQL instance configured in .env.dev
    WHEN migrations, POST /api/v1/ingestion/runs, GET /api/v1/ingestion/jobs/{job_id},
         and the forget CLI run end-to-end
    THEN migrations are idempotent, job/quarantine/mapping rows persist, and forget erases the mapping.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param capsys: Pytest stdout/stderr capture fixture.
    """
    dev_settings = Settings(app_env=Environment.DEV)
    conn_kwargs = dev_settings.get_database_kwargs()
    if conn_kwargs is None:
        pytest.skip("POSTGRES_PASSWORD is not configured for live PostgreSQL verification.")

    # Verify 0001 and 0002 migrations apply cleanly and are idempotent on repeat runs.
    apply_migrations(conn_kwargs)
    second_pass = apply_migrations(conn_kwargs)
    assert second_pass == []

    # Isolate live verification to PostgreSQL: disable Redis caches and OTel so containers aren't required,
    # and instantiate TestClient after monkeypatch (the shared `client` fixture starts lifespan beforehand).
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("ENABLE_EXACT_CACHE", "false")
    monkeypatch.setenv("ENABLE_SEMANTIC_CACHE", "false")
    monkeypatch.setenv("ENABLE_OTEL_TRACING", "false")

    # Capture DB clock before the run so `finally` deletes only mappings created by this test.
    with psycopg.connect(**conn_kwargs) as conn:
        row = conn.execute("SELECT now();").fetchone()
        assert row is not None
        test_started_at: datetime = row[0]

    job_id: UUID | None = None
    backend_options = {"loop_factory": asyncio.SelectorEventLoop} if sys.platform == "win32" else None
    try:
        with TestClient(create_app(), backend_options=backend_options) as client:
            run_response = client.post("/api/v1/ingestion/runs", json={"source_name": "historical_budgets"})
            assert run_response.status_code == 202
            job_id = UUID(run_response.json()["job_id"])

            job_response = client.get(f"/api/v1/ingestion/jobs/{job_id}")
            assert job_response.status_code == 200
            job_payload = job_response.json()
            assert job_payload["status"] == "completed"
            assert job_payload["documents_count"] == 2
            assert job_payload["validation_report"] is not None

        with psycopg.connect(**conn_kwargs) as conn:
            quarantine_row = conn.execute(
                "SELECT count(*) FROM public.quarantined_records WHERE job_id = %s;",
                (job_id,),
            ).fetchone()
            assert quarantine_row is not None and quarantine_row[0] >= 1

            target_hash = compute_pii_hash("Laura Miller", resolve_pii_salt(dev_settings))
            mapping_row = conn.execute(
                "SELECT source_names FROM public.pseudonym_mappings WHERE entity_type = %s AND original_hash = %s;",
                ("PERSON", target_hash),
            ).fetchone()
            assert mapping_row is not None
            assert "historical_budgets" in mapping_row[0]

            exit_code = forget_main(["--entity-type", "PERSON", "--value", "Laura Miller"])
            assert exit_code == 0
            assert "erased" in capsys.readouterr().out

            deleted_row = conn.execute(
                "SELECT 1 FROM public.pseudonym_mappings WHERE entity_type = %s AND original_hash = %s;",
                ("PERSON", target_hash),
            ).fetchone()
            assert deleted_row is None
    finally:
        # Deleting the job row cascades to `quarantined_records` via FK ON DELETE CASCADE.
        with psycopg.connect(**conn_kwargs) as conn, conn.transaction():
            if job_id is not None:
                conn.execute("DELETE FROM public.ingestion_jobs WHERE job_id = %s;", (job_id,))
            conn.execute("DELETE FROM public.pseudonym_mappings WHERE created_at >= %s;", (test_started_at,))
