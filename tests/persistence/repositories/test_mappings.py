# tests/persistence/repositories/test_mappings.py

# --- Python Standard Library ---
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.persistence.repositories.mappings import Mapping, MappingsRepository


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
async def test_mappings_repository_lookup_or_create_returns_existing_without_factory() -> None:
    """
    GIVEN an existing pseudonym mapping row
    WHEN lookup_or_create is awaited
    THEN the existing Mapping is returned without invoking new_pseudonym_factory.
    """
    expected = Mapping(entity_type="PERSON", original_hash="hash123", pseudonym="Alice Smith", created_at=datetime.now(UTC))
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(return_value=expected)
    pool, _ = _make_mock_pool(cursor)
    factory = MagicMock(return_value="Unused Name")

    result = await MappingsRepository(pool).lookup_or_create("PERSON", "hash123", factory)

    assert result == expected
    factory.assert_not_called()
    cursor.execute.assert_awaited_once()


@pytest.mark.anyio
async def test_mappings_repository_lookup_or_create_inserts_on_miss() -> None:
    """
    GIVEN no existing mapping row
    WHEN lookup_or_create is awaited and INSERT ... RETURNING succeeds
    THEN the factory is called once and the inserted Mapping is returned.
    """
    expected = Mapping(entity_type="EMAIL", original_hash="hash456", pseudonym="anon@example.test", created_at=datetime.now(UTC))
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(side_effect=[None, expected])
    pool, _ = _make_mock_pool(cursor)
    factory = MagicMock(return_value="anon@example.test")

    result = await MappingsRepository(pool).lookup_or_create("EMAIL", "hash456", factory)

    factory.assert_called_once()
    assert result == expected


@pytest.mark.anyio
async def test_mappings_repository_lookup_or_create_resolves_concurrent_conflict() -> None:
    """
    GIVEN a cache miss followed by a concurrent ON CONFLICT DO NOTHING insert returning None
    WHEN lookup_or_create falls back to SELECT
    THEN the winning concurrent Mapping row is returned.
    """
    winner = Mapping(entity_type="PERSON", original_hash="hash789", pseudonym="Winner Name", created_at=datetime.now(UTC))
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(side_effect=[None, None, winner])
    pool, _ = _make_mock_pool(cursor)
    factory = MagicMock(return_value="Loser Name")

    result = await MappingsRepository(pool).lookup_or_create("PERSON", "hash789", factory)

    assert result == winner


@pytest.mark.anyio
async def test_mappings_repository_lookup_or_create_raises_if_winner_absent() -> None:
    """
    GIVEN a miss, an empty RETURNING result, and an empty fallback SELECT
    WHEN lookup_or_create is awaited
    THEN a RuntimeError is raised.
    """
    cursor = MagicMock()
    cursor.execute = AsyncMock(return_value=None)
    cursor.fetchone = AsyncMock(side_effect=[None, None, None])
    pool, _ = _make_mock_pool(cursor)

    with pytest.raises(RuntimeError, match=r"Missing pseudonym mapping"):
        await MappingsRepository(pool).lookup_or_create("PERSON", "hash000", lambda: "Name")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("rowcount", "expected_deleted"),
    [(1, True), (0, False)],
    ids=["deleted", "absent"],
)
async def test_mappings_repository_forget(rowcount: int, expected_deleted: bool) -> None:
    """
    GIVEN a DELETE rowcount of 1 or 0
    WHEN MappingsRepository.forget is awaited
    THEN True is returned when a row was erased and False otherwise.

    :param rowcount: Simulated SQL rowcount.
    :param expected_deleted: Expected boolean return value.
    """
    cursor = MagicMock()
    cursor.rowcount = rowcount
    pool, _ = _make_mock_pool(cursor)

    assert await MappingsRepository(pool).forget("PERSON", "hash123") is expected_deleted
