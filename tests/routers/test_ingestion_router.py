# tests/routers/test_ingestion_router.py

# --- Python Standard Library ---
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

# --- Third Party Libraries ---
from fastapi import status
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.dependencies import (
    get_catalog,
    get_filesystem_loader,
    get_jobs_repository,
    get_parser_registry,
    get_pii_pseudonymizer,
    get_quarantined_records_repository,
)
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.parsers.registry import default_registry
from aie_estimator.persistence.repositories.jobs import Job, JobStatus
from aie_estimator.server import create_app


def _make_job(
    job_id: UUID,
    source_name: str = "historical_budgets",
    job_status: JobStatus = JobStatus.PENDING,
    documents_count: int = 0,
    error_message: str | None = None,
) -> Job:
    return Job(
        job_id=job_id,
        source_name=source_name,
        status=job_status,
        documents_count=documents_count,
        error_message=error_message,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC) if job_status in {JobStatus.COMPLETED, JobStatus.FAILED} else None,
    )


def _build_client(
    mock_repo: MagicMock,
    mock_quarantine_repo: MagicMock | None = None,
    mock_pseudonymizer: MagicMock | None = None,
) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_catalog] = lambda: load_catalog(Path("data/catalog/catalog.yaml"))
    app.dependency_overrides[get_filesystem_loader] = lambda: FileSystemLoader(Path("data/seed"))
    app.dependency_overrides[get_parser_registry] = default_registry
    app.dependency_overrides[get_jobs_repository] = lambda: mock_repo
    if mock_quarantine_repo is None:
        mock_quarantine_repo = MagicMock()
        mock_quarantine_repo.upsert_many = AsyncMock(return_value=0)
    app.dependency_overrides[get_quarantined_records_repository] = lambda: mock_quarantine_repo
    if mock_pseudonymizer is None:
        mock_pseudonymizer = MagicMock()
        mock_pseudonymizer.pseudonymize_documents = AsyncMock(side_effect=lambda docs, **_: list(docs))
    app.dependency_overrides[get_pii_pseudonymizer] = lambda: mock_pseudonymizer
    return TestClient(app)


def test_post_ingestion_run_dispatches_background_job() -> None:
    """
    GIVEN an include catalog source and mocked repositories
    WHEN POST /api/v1/ingestion/runs is called
    THEN HTTP 202 is returned, quarantined records are upserted, and the job completes with validation_report.
    """
    job_id = uuid4()
    mock_repo = MagicMock()
    mock_repo.create = AsyncMock(return_value=_make_job(job_id))
    mock_repo.mark_running = AsyncMock()
    mock_repo.mark_completed = AsyncMock()
    mock_repo.mark_failed = AsyncMock()

    mock_quarantine_repo = MagicMock()
    mock_quarantine_repo.upsert_many = AsyncMock(return_value=3)

    mock_pseudonymizer = MagicMock()
    mock_pseudonymizer.pseudonymize_documents = AsyncMock(side_effect=lambda docs, **_: list(docs))

    with _build_client(mock_repo, mock_quarantine_repo, mock_pseudonymizer) as client:
        response = client.post("/api/v1/ingestion/runs", json={"source_name": "historical_budgets"})

    assert response.status_code == status.HTTP_202_ACCEPTED
    payload = response.json()
    assert payload == {"job_id": str(job_id), "source_name": "historical_budgets", "status": "pending"}
    mock_repo.create.assert_awaited_once_with("historical_budgets")
    mock_repo.mark_running.assert_awaited_once_with(job_id)
    mock_pseudonymizer.pseudonymize_documents.assert_awaited_once()
    mock_quarantine_repo.upsert_many.assert_awaited_once()
    assert len(mock_quarantine_repo.upsert_many.await_args.kwargs["records"]) == 3
    mock_repo.mark_completed.assert_awaited_once()
    assert mock_repo.mark_completed.await_args.kwargs["documents_count"] == 2
    assert mock_repo.mark_completed.await_args.kwargs["validation_report"]["valid_rows"] == 2
    mock_repo.mark_failed.assert_not_awaited()


def test_post_ingestion_run_marks_failed_when_background_errors() -> None:
    """
    GIVEN a loader pointing to a missing data root
    WHEN POST /api/v1/ingestion/runs executes the background task
    THEN HTTP 202 is returned and the background task marks the job failed.
    """
    job_id = uuid4()
    mock_repo = MagicMock()
    mock_repo.create = AsyncMock(return_value=_make_job(job_id))
    mock_repo.mark_running = AsyncMock()
    mock_repo.mark_completed = AsyncMock()
    mock_repo.mark_failed = AsyncMock()

    app = create_app()
    app.dependency_overrides[get_catalog] = lambda: load_catalog(Path("data/catalog/catalog.yaml"))
    app.dependency_overrides[get_filesystem_loader] = lambda: FileSystemLoader(Path("data/nonexistent_root"))
    app.dependency_overrides[get_parser_registry] = default_registry
    app.dependency_overrides[get_jobs_repository] = lambda: mock_repo
    app.dependency_overrides[get_quarantined_records_repository] = lambda: MagicMock()
    app.dependency_overrides[get_pii_pseudonymizer] = lambda: None

    with TestClient(app) as client:
        response = client.post("/api/v1/ingestion/runs", json={"source_name": "historical_budgets"})

    assert response.status_code == status.HTTP_202_ACCEPTED
    mock_repo.mark_running.assert_awaited_once_with(job_id)
    mock_repo.mark_failed.assert_awaited_once()
    mock_repo.mark_completed.assert_not_awaited()


@pytest.mark.parametrize(
    ("source_name", "expected_status"),
    [
        ("unknown_source", status.HTTP_404_NOT_FOUND),
        ("meeting_transcripts", status.HTTP_400_BAD_REQUEST),
        ("official_rate_card", status.HTTP_400_BAD_REQUEST),
    ],
)
def test_post_ingestion_run_rejects_unknown_and_non_included_sources(source_name: str, expected_status: int) -> None:
    """
    GIVEN an unknown, review, or exclude source name
    WHEN POST /api/v1/ingestion/runs is called
    THEN HTTP 404 or 400 is returned without creating a job row.
    """
    mock_repo = MagicMock()
    mock_repo.create = AsyncMock()

    with _build_client(mock_repo) as client:
        response = client.post("/api/v1/ingestion/runs", json={"source_name": source_name})

    assert response.status_code == expected_status
    mock_repo.create.assert_not_awaited()


def test_get_ingestion_job_returns_job_or_404() -> None:
    """
    GIVEN a persisted job and an unknown job UUID
    WHEN GET /api/v1/ingestion/jobs/{job_id} is called
    THEN HTTP 200 returns the job payload and HTTP 404 is returned for missing jobs.
    """
    job_id = uuid4()
    completed_job = _make_job(job_id, job_status=JobStatus.COMPLETED, documents_count=6)
    mock_repo = MagicMock()
    mock_repo.get = AsyncMock(side_effect=[completed_job, None])

    with _build_client(mock_repo) as client:
        found = client.get(f"/api/v1/ingestion/jobs/{job_id}")
        missing = client.get(f"/api/v1/ingestion/jobs/{uuid4()}")

    assert found.status_code == status.HTTP_200_OK
    assert found.json()["documents_count"] == 6
    assert found.json()["status"] == "completed"
    assert missing.status_code == status.HTTP_404_NOT_FOUND


def test_post_ingestion_run_skips_pseudonymization_when_disabled() -> None:
    """
    GIVEN an include PII source with get_pii_pseudonymizer returning None
    WHEN POST /api/v1/ingestion/runs executes the background task
    THEN pseudonymization is skipped cleanly and the job completes.
    """
    job_id = uuid4()
    mock_repo = MagicMock()
    mock_repo.create = AsyncMock(return_value=_make_job(job_id))
    mock_repo.mark_running = AsyncMock()
    mock_repo.mark_completed = AsyncMock()
    mock_repo.mark_failed = AsyncMock()

    mock_quarantine_repo = MagicMock()
    mock_quarantine_repo.upsert_many = AsyncMock(return_value=3)

    app = create_app()
    app.dependency_overrides[get_catalog] = lambda: load_catalog(Path("data/catalog/catalog.yaml"))
    app.dependency_overrides[get_filesystem_loader] = lambda: FileSystemLoader(Path("data/seed"))
    app.dependency_overrides[get_parser_registry] = default_registry
    app.dependency_overrides[get_jobs_repository] = lambda: mock_repo
    app.dependency_overrides[get_quarantined_records_repository] = lambda: mock_quarantine_repo
    app.dependency_overrides[get_pii_pseudonymizer] = lambda: None

    with TestClient(app) as client:
        response = client.post("/api/v1/ingestion/runs", json={"source_name": "historical_budgets"})

    assert response.status_code == status.HTTP_202_ACCEPTED
    mock_repo.mark_completed.assert_awaited_once()
    mock_repo.mark_failed.assert_not_awaited()
