# tests/persistence/repositories/test_jobs.py

# --- Python Standard Library ---
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.persistence.repositories.jobs import (
    MAX_ERROR_MESSAGE_LENGTH,
    Job,
    JobsRepository,
    JobStatus,
)


def _make_mock_pool(cursor: MagicMock) -> tuple[MagicMock, MagicMock]:
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=cursor)

    @asynccontextmanager
    async def _cursor(*args: Any, **kwargs: Any):
        yield cursor

    @asynccontextmanager
    async def _tx():
        yield

    @asynccontextmanager
    async def _connection():
        yield conn

    conn.cursor = _cursor
    conn.transaction = _tx
    pool = MagicMock()
    pool.connection = _connection
    return pool, conn


@pytest.mark.anyio
async def test_jobs_repository_create_returns_job() -> None:
    """
    GIVEN an AsyncConnectionPool returning an inserted Job model
    WHEN JobsRepository.create is awaited
    THEN the frozen Pydantic Job instance is returned matching the persisted fields.
    """
    expected_job = Job(
        job_id=uuid4(),
        source_name="historical_budgets",
        status=JobStatus.PENDING,
        documents_count=0,
        error_message=None,
        started_at=datetime.now(UTC),
        finished_at=None,
    )
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(return_value=expected_job)
    pool, _ = _make_mock_pool(cursor)

    job = await JobsRepository(pool).create("historical_budgets")

    assert job == expected_job
    assert cursor.execute.await_args.args[1][1] == "historical_budgets"
    assert cursor.execute.await_args.args[1][2] is JobStatus.PENDING


@pytest.mark.anyio
async def test_jobs_repository_create_raises_when_returning_empty() -> None:
    """
    GIVEN a pool cursor returning None from INSERT ... RETURNING
    WHEN JobsRepository.create is awaited
    THEN a RuntimeError is raised.
    """
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(return_value=None)
    pool, _ = _make_mock_pool(cursor)

    with pytest.raises(RuntimeError, match=r"Failed to insert"):
        await JobsRepository(pool).create("historical_budgets")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("row", "expected_present"),
    [
        (
            Job(
                job_id=uuid4(),
                source_name="rate_cards",
                status="completed",
                documents_count=4,
                error_message=None,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            ),
            True,
        ),
        (None, False),
    ],
    ids=["found", "missing"],
)
async def test_jobs_repository_get(row: Job | None, expected_present: bool) -> None:
    """
    GIVEN a Job row or None for a requested job_id
    WHEN JobsRepository.get is awaited
    THEN the Job instance is returned when found or None when absent.

    :param row: Simulated Job model or None.
    :param expected_present: Whether a Job should be returned.
    """
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(return_value=row)
    pool, _ = _make_mock_pool(cursor)

    result = await JobsRepository(pool).get(uuid4())

    assert (result is not None) is expected_present
    if row is not None:
        assert result == row


@pytest.mark.anyio
async def test_jobs_repository_status_transitions_and_truncation() -> None:
    """
    GIVEN an active JobsRepository
    WHEN mark_running, mark_completed, mark_failed, and fail_stale_jobs are awaited
    THEN SQL updates execute with expected parameters including 2048-char error truncation.
    """
    cursor = MagicMock()
    cursor.rowcount = 2
    pool, conn = _make_mock_pool(cursor)
    repo = JobsRepository(pool)
    job_id = uuid4()

    await repo.mark_running(job_id)
    await repo.mark_completed(job_id, documents_count=12)
    await repo.mark_failed(job_id, error_message="x" * (MAX_ERROR_MESSAGE_LENGTH + 50))
    recovered = await repo.fail_stale_jobs()

    assert recovered == 2
    calls = conn.execute.await_args_list
    assert calls[0].args[1] == (JobStatus.RUNNING, job_id)
    assert calls[1].args[1][0] is JobStatus.COMPLETED
    assert calls[1].args[1][1] == 12
    assert calls[1].args[1][3] == job_id
    assert calls[2].args[1][0] is JobStatus.FAILED
    assert len(calls[2].args[1][1]) == MAX_ERROR_MESSAGE_LENGTH
    assert calls[2].args[1][3] == job_id
    assert calls[3].args[1][0] is JobStatus.FAILED
    assert calls[3].args[1][3] == [JobStatus.PENDING, JobStatus.RUNNING]
