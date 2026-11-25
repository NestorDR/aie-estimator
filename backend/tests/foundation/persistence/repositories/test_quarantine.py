# tests/foundation/persistence/repositories/test_quarantine.py

# --- Python Standard Library ---
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

# --- Third Party Libraries ---
from psycopg.types.json import Jsonb
import pytest

# --- App modules ---
from aie_estimator.foundation.persistence.repositories.quarantine import (
    QuarantinedRecordInput,
    QuarantinedRecordsRepository,
)


def _make_mock_pool(cursor: MagicMock) -> MagicMock:
    conn = MagicMock()

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
    return pool


@pytest.mark.anyio
async def test_quarantined_records_upsert_many_empty_is_noop() -> None:
    """
    GIVEN an empty sequence of QuarantinedRecordInput
    WHEN upsert_many is awaited
    THEN zero is returned without opening a database connection.
    """
    pool = MagicMock()
    count = await QuarantinedRecordsRepository(pool).upsert_many(
        job_id=uuid4(),
        source_name="historical_budgets",
        records=[],
    )
    assert count == 0
    pool.connection.assert_not_called()


@pytest.mark.anyio
async def test_quarantined_records_upsert_many_executes_idempotent_batch() -> None:
    """
    GIVEN quarantined records for an ingestion job
    WHEN upsert_many is awaited
    THEN executemany runs with Jsonb-wrapped failure_reasons and raw_payload and returns the record count.
    """
    cursor = MagicMock()
    cursor.executemany = AsyncMock()
    pool = _make_mock_pool(cursor)
    job_id = uuid4()
    records = [
        QuarantinedRecordInput(
            record_key="BUDGET-2024-0003",
            content_hash="abc123",
            failure_reasons=["not_nullable"],
            raw_payload={"budget_id": "BUDGET-2024-0003", "client_name": None},
        )
    ]

    count = await QuarantinedRecordsRepository(pool).upsert_many(
        job_id=job_id,
        source_name="historical_budgets",
        records=records,
    )

    assert count == 1
    cursor.executemany.assert_awaited_once()
    params = cursor.executemany.await_args.args[1]
    assert len(params) == 1
    assert params[0][0] == job_id
    assert params[0][1] == "historical_budgets"
    assert params[0][2] == "BUDGET-2024-0003"
    assert params[0][3] == "abc123"
    assert isinstance(params[0][4], Jsonb)
    assert isinstance(params[0][5], Jsonb)
