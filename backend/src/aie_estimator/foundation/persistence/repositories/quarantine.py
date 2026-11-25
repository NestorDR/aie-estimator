# src/aie_estimator/foundation/persistence/repositories/quarantine.py

"""
Async raw-SQL repository for idempotent ``quarantined_records`` persistence.

Uses ``ON CONFLICT (source_name, record_key, content_hash) DO UPDATE`` so
re-running ingestion over an unchanged corpus refreshes ``job_id`` and increments
``seen_count`` without creating duplicate quarantine rows.
"""

# --- Python Standard Library ---
from datetime import datetime
from typing import Any, ClassVar, Final
from uuid import UUID

# --- Third Party Libraries ---
from psycopg.sql import SQL, Composed, Identifier
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)


class QuarantinedRecordInput(BaseModel):
    """Immutable payload emitted by cleaning/validation for a quarantined record."""

    model_config = ConfigDict(frozen=True)

    record_key: str
    content_hash: str
    failure_reasons: list[str]
    raw_payload: dict[str, Any]


class QuarantinedRecord(BaseModel):
    """Immutable domain view of a ``quarantined_records`` row."""

    __tablename__: ClassVar[str] = "quarantined_records"

    model_config = ConfigDict(frozen=True)

    id: int
    job_id: UUID
    source_name: str
    record_key: str
    content_hash: str
    failure_reasons: list[str]
    raw_payload: dict[str, Any]
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime


QUARANTINED_TABLE: Final[Identifier] = Identifier("public", QuarantinedRecord.__tablename__)

_UPSERT_QUARANTINED_SQL: Final[Composed] = SQL(
    "INSERT INTO {table} (job_id, source_name, record_key, content_hash, failure_reasons, raw_payload) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (source_name, record_key, content_hash) DO UPDATE SET "
    "job_id = EXCLUDED.job_id, "
    "failure_reasons = EXCLUDED.failure_reasons, "
    "raw_payload = EXCLUDED.raw_payload, "
    "seen_count = {table}.seen_count + 1, "
    "last_seen_at = now();"
).format(table=QUARANTINED_TABLE)


class QuarantinedRecordsRepository:
    """Async raw-SQL repository managing idempotent ``quarantined_records`` persistence."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        """
        Initialize the repository with an active asynchronous connection pool.

        :param pool: Opened psycopg AsyncConnectionPool instance.
        """
        self._pool = pool

    async def upsert_many(
        self,
        *,
        job_id: UUID,
        source_name: str,
        records: list[QuarantinedRecordInput],
    ) -> int:
        """
        Idempotently upsert quarantined records for an ingestion run.

        :param job_id: Ingestion job UUID that observed the records.
        :param source_name: Audited catalog source identifier.
        :param records: List of quarantined record payloads to persist.

        :return: Number of quarantined records processed.
        """
        if not records:
            return 0

        params = [
            (
                job_id,
                source_name,
                record.record_key,
                record.content_hash,
                Jsonb(record.failure_reasons),
                Jsonb(record.raw_payload),
            )
            for record in records
        ]
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.executemany(_UPSERT_QUARANTINED_SQL, params)

        log.info("quarantined_records_upserted", job_id=str(job_id), source_name=source_name, count=len(records))
        return len(records)
