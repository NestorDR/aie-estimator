# src/aie_estimator/persistence/repositories/mappings.py

# --- Python Standard Library ---
from collections.abc import Callable
from datetime import datetime
from typing import ClassVar, Final

# --- Third Party Libraries ---
from psycopg.rows import class_row
from psycopg.sql import SQL, Composable, Composed, Identifier
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)


class Mapping(BaseModel):
    """Immutable domain view of a ``pseudonym_mappings`` row."""

    __tablename__: ClassVar[str] = "pseudonym_mappings"

    model_config = ConfigDict(frozen=True)

    entity_type: str
    original_hash: str
    pseudonym: str
    created_at: datetime


MAPPING_TABLE: Final[Identifier] = Identifier("public", Mapping.__tablename__)
_MAPPING_COLUMNS: Final[Composable] = SQL(", ").join(map(Identifier, Mapping.model_fields))

_SELECT_MAPPING_SQL: Final[Composed] = SQL(
    "SELECT {cols} FROM {table} WHERE entity_type = %s AND original_hash = %s;"
).format(table=MAPPING_TABLE, cols=_MAPPING_COLUMNS)
_UPSERT_MAPPING_SQL: Final[Composed] = SQL(
    "INSERT INTO {table} (entity_type, original_hash, pseudonym) "
    "VALUES (%s, %s, %s) ON CONFLICT (entity_type, original_hash) DO NOTHING RETURNING {cols};"
).format(table=MAPPING_TABLE, cols=_MAPPING_COLUMNS)
_DELETE_MAPPING_SQL: Final[Composed] = SQL(
    "DELETE FROM {table} WHERE entity_type = %s AND original_hash = %s;"
).format(table=MAPPING_TABLE)


class MappingsRepository:
    """Async raw-SQL repository managing GDPR-auditable ``pseudonym_mappings`` rows."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        """
        Initialize the repository with an active asynchronous connection pool.

        :param pool: Opened psycopg AsyncConnectionPool instance.
        """
        self._pool = pool

    async def lookup(self, entity_type: str, original_hash: str) -> Mapping | None:
        """
        Look up an existing pseudonym mapping by entity type and salted hash.

        :param entity_type: PII entity category (e.g., ``PERSON``, ``EMAIL``).
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: Matching Mapping record, or None if not present.
        """
        async with self._pool.connection() as conn, conn.cursor(row_factory=class_row(Mapping)) as cur:
            await cur.execute(_SELECT_MAPPING_SQL, (entity_type, original_hash))
            return await cur.fetchone()

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
    ) -> Mapping:
        """
        Return an existing pseudonym mapping or insert a new one atomically.

        :param entity_type: PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.
        :param new_pseudonym_factory: Zero-argument callable generating a pseudonym only on misses.

        :return: Deterministic winning Mapping record.
        :raises RuntimeError: If neither the insert nor the conflict fallback locates a row.
        """
        existing = await self.lookup(entity_type, original_hash)
        if existing is not None:
            return existing

        pseudonym = new_pseudonym_factory()
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=class_row(Mapping)) as cur:
            await cur.execute(_UPSERT_MAPPING_SQL, (entity_type, original_hash, pseudonym))
            mapping = await cur.fetchone()

        if mapping is not None:
            log.info("pseudonym_mapping_created", entity_type=entity_type, original_hash=original_hash)
            return mapping

        winner = await self.lookup(entity_type, original_hash)
        if winner is None:
            raise RuntimeError(f"Missing pseudonym mapping after conflict for entity_type='{entity_type}'.")
        return winner

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        """
        Delete a pseudonym mapping row to fulfill GDPR Article 17 erasure requests.

        :param entity_type: PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: True if a row was deleted, False when no matching mapping existed.
        """
        async with self._pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(_DELETE_MAPPING_SQL, (entity_type, original_hash))
            deleted = cursor.rowcount > 0

        if deleted:
            log.info("pseudonym_mapping_deleted", entity_type=entity_type, original_hash=original_hash)
        return deleted
