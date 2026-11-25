# src/aie_estimator/foundation/persistence/repositories/jobs.py

# --- Python Standard Library ---
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Final
from uuid import UUID, uuid4

# --- Third Party Libraries ---
from psycopg.rows import class_row
from psycopg.sql import SQL, Composable, Composed, Identifier
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

MAX_ERROR_MESSAGE_LENGTH: Final[int] = 2048
STALE_JOB_ERROR_MESSAGE: Final[str] = "Job interrupted by application restart or shutdown."


class JobStatus(StrEnum):
    """Lifecycle states for an ``ingestion_jobs`` record."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Immutable domain view of an ``ingestion_jobs`` row."""

    __tablename__: ClassVar[str] = "ingestion_jobs"

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    source_name: str
    status: JobStatus
    documents_count: int
    validation_report: dict[str, Any] | None = None
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None


JOB_TABLE: Final[Identifier] = Identifier("public", Job.__tablename__)
_JOB_COLUMNS: Final[Composable] = SQL(", ").join(map(Identifier, Job.model_fields))

_INSERT_JOB_SQL: Final[Composed] = SQL(
    "INSERT INTO {table} (job_id, source_name, status) VALUES (%s, %s, %s) RETURNING {cols};"
).format(table=JOB_TABLE, cols=_JOB_COLUMNS)
_SELECT_JOB_SQL: Final[Composed] = SQL("SELECT {cols} FROM {table} WHERE job_id = %s;").format(table=JOB_TABLE, cols=_JOB_COLUMNS)

_MARK_RUNNING_SQL: Final[Composed] = SQL("UPDATE {table} SET status = %s WHERE job_id = %s;").format(table=JOB_TABLE)
_MARK_COMPLETED_SQL: Final[Composed] = SQL(
    "UPDATE {table} SET status = %s, documents_count = %s, validation_report = %s, finished_at = %s WHERE job_id = %s;"
).format(table=JOB_TABLE)
_MARK_FAILED_SQL: Final[Composed] = SQL(
    "UPDATE {table} SET status = %s, error_message = %s, finished_at = %s WHERE job_id = %s;"
).format(table=JOB_TABLE)
_FAIL_STALE_JOBS_SQL: Final[Composed] = SQL(
    "UPDATE {table} SET status = %s, error_message = %s, finished_at = %s WHERE status = ANY(%s);"
).format(table=JOB_TABLE)


class JobsRepository:
    """Async raw-SQL repository managing ``ingestion_jobs`` lifecycle state."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        """
        Initialize the repository with an active asynchronous connection pool.

        :param pool: Opened psycopg AsyncConnectionPool instance.
        """
        self._pool = pool

    async def create(self, source_name: str) -> Job:
        """
        Insert a new ``pending`` ingestion job for the given source.

        :param source_name: Catalog source identifier being ingested.

        :return: Created immutable Job domain record.
        :raises RuntimeError: If the insert fails to return the created row.
        """
        job_id = uuid4()
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=class_row(Job)) as cur:
            await cur.execute(_INSERT_JOB_SQL, (job_id, source_name, JobStatus.PENDING))
            job = await cur.fetchone()
        if job is None:
            raise RuntimeError(f"Failed to insert ingestion job for source '{source_name}'.")
        log.info("ingestion_job_created", job_id=str(job.job_id), source_name=source_name)
        return job

    async def get(self, job_id: UUID) -> Job | None:
        """
        Retrieve an ingestion job by its unique identifier.

        :param job_id: UUID of the ingestion job.

        :return: Matching Job record, or None if not found.
        """
        async with self._pool.connection() as conn, conn.cursor(row_factory=class_row(Job)) as cur:
            await cur.execute(_SELECT_JOB_SQL, (job_id,))
            return await cur.fetchone()

    async def mark_running(self, job_id: UUID) -> None:
        """
        Transition an ingestion job to ``running``.

        :param job_id: UUID of the ingestion job.
        """
        await self._execute_update(_MARK_RUNNING_SQL, (JobStatus.RUNNING, job_id))
        log.info("ingestion_job_updated", job_id=str(job_id), status=JobStatus.RUNNING)

    async def mark_completed(
        self,
        job_id: UUID,
        *,
        documents_count: int,
        validation_report: dict[str, Any] | None = None,
    ) -> None:
        """
        Transition an ingestion job to ``completed`` with the emitted document count and report.

        :param job_id: UUID of the ingestion job.
        :param documents_count: Total canonical documents emitted by the run.
        :param validation_report: Optional JSON-serializable cleaning and validation summary.
        """
        finished_at = datetime.now(UTC)
        report_param = Jsonb(validation_report) if validation_report is not None else None
        await self._execute_update(_MARK_COMPLETED_SQL, (JobStatus.COMPLETED, documents_count, report_param, finished_at, job_id))
        log.info("ingestion_job_updated", job_id=str(job_id), status=JobStatus.COMPLETED, documents_count=documents_count)

    async def mark_failed(self, job_id: UUID, *, error_message: str) -> None:
        """
        Transition an ingestion job to ``failed`` with a truncated diagnostic message.

        :param job_id: UUID of the ingestion job.
        :param error_message: Error description to persist (truncated to 2048 chars).
        """
        finished_at = datetime.now(UTC)
        truncated = error_message[:MAX_ERROR_MESSAGE_LENGTH]
        await self._execute_update(_MARK_FAILED_SQL, (JobStatus.FAILED, truncated, finished_at, job_id))
        log.info("ingestion_job_updated", job_id=str(job_id), status=JobStatus.FAILED)

    async def fail_stale_jobs(self) -> int:
        """
        Mark any orphaned ``pending`` or ``running`` jobs from previous runs as ``failed``.

        :return: Number of stale jobs marked as failed.
        """
        finished_at = datetime.now(UTC)
        recovered_count = await self._execute_update(
            _FAIL_STALE_JOBS_SQL,
            (JobStatus.FAILED, STALE_JOB_ERROR_MESSAGE, finished_at, [JobStatus.PENDING, JobStatus.RUNNING]),
        )
        log.info("ingestion_jobs_recovered", count=recovered_count)
        return recovered_count

    async def _execute_update(self, sql_query: Composed, params: tuple[Any, ...]) -> int:
        async with self._pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(sql_query, params)
            return cursor.rowcount
