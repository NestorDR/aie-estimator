# src/aie_estimator/foundation/persistence/repositories/mappings.py

"""
Async PostgreSQL repository for GDPR-auditable pseudonym mappings.

``lookup_or_create`` is called per detected entity and must be idempotent under contention:
two concurrent writers for the same ``(entity_type, original_hash)`` must converge to the
same pseudonym or corpus-wide consistency breaks.

Rather than explicit table locking, convergence relies on PostgreSQL's
``UNIQUE (entity_type, original_hash)`` constraint (``INSERT ... ON CONFLICT DO NOTHING
RETURNING`` with a fallback ``SELECT`` on conflict) while atomically appending unseen
catalog ``source_names`` and returning immutable domain models rather than raw driver rows.
Plaintext PII never enters this table—only salted HMAC-SHA256 digests and replacement tokens.
"""

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
    source_names: tuple[str, ...] = ()
    created_at: datetime


MAPPING_TABLE: Final[Identifier] = Identifier("public", Mapping.__tablename__)
_MAPPING_COLUMNS: Final[Composable] = SQL(", ").join(map(Identifier, Mapping.model_fields))

_SELECT_MAPPING_SQL: Final[Composed] = SQL(
    "SELECT {cols} FROM {table} WHERE entity_type = %s AND original_hash = %s;"
).format(table=MAPPING_TABLE, cols=_MAPPING_COLUMNS)
_UPSERT_MAPPING_SQL: Final[Composed] = SQL(
    "INSERT INTO {table} (entity_type, original_hash, pseudonym, source_names) "
    "VALUES (%s, %s, %s, %s) ON CONFLICT (entity_type, original_hash) DO NOTHING RETURNING {cols};"
).format(table=MAPPING_TABLE, cols=_MAPPING_COLUMNS)
# Guard with `NOT (%s = ANY(source_names))` so concurrent appends stay idempotent without duplicate entries.
_APPEND_SOURCE_SQL: Final[Composed] = SQL(
    "UPDATE {table} SET source_names = array_append(source_names, %s) "
    "WHERE entity_type = %s AND original_hash = %s AND NOT (%s = ANY(source_names)) RETURNING {cols};"
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

        :param entity_type: PII entity category (e.g., ``PERSON``, ``EMAIL_ADDRESS``).
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: Matching Mapping record, or None if not present.
        """
        async with self._pool.connection() as conn, conn.cursor(row_factory=class_row(Mapping)) as cur:
            await cur.execute(_SELECT_MAPPING_SQL, (entity_type, original_hash))
            return await cur.fetchone()

    async def _append_source_name(self, mapping: Mapping, source_name: str) -> Mapping:
        """
        Append an unseen catalog source identifier to an existing mapping row atomically.

        :param mapping: Existing Mapping record.
        :param source_name: Catalog source identifier to record.

        :return: Updated Mapping record containing ``source_name``.
        """
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=class_row(Mapping)) as cur:
            await cur.execute(
                _APPEND_SOURCE_SQL,
                (source_name, mapping.entity_type, mapping.original_hash, source_name),
            )
            updated = await cur.fetchone()

        if updated is not None:
            log.info(
                "pseudonym_mapping_updated",
                entity_type=mapping.entity_type,
                original_hash=mapping.original_hash,
                source_name=source_name,
            )
            return updated
        # updated is None when a concurrent writer already appended source_name before our UPDATE ran.
        return mapping

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
        *,
        source_name: str | None = None,
    ) -> Mapping:
        """
        Return an existing pseudonym mapping or insert a new one atomically, recording source provenance.

        :param entity_type: PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.
        :param new_pseudonym_factory: Zero-argument callable generating a pseudonym only on misses.
        :param source_name: Optional catalog source identifier to record in ``source_names``.

        :return: Deterministic winning Mapping record.
        :raises RuntimeError: If neither the insert nor the conflict fallback locates a row.
        """
        # Read first so cache hits skip Faker generation and write transactions; only append unseen sources.
        existing = await self.lookup(entity_type, original_hash)
        if existing is not None:
            if source_name is not None and source_name not in existing.source_names:
                return await self._append_source_name(existing, source_name)
            return existing

        pseudonym = new_pseudonym_factory()
        initial_sources = [source_name] if source_name is not None else []
        async with self._pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=class_row(Mapping)) as cur:
            await cur.execute(_UPSERT_MAPPING_SQL, (entity_type, original_hash, pseudonym, initial_sources))
            mapping = await cur.fetchone()

        if mapping is not None:
            log.info(
                "pseudonym_mapping_created",
                entity_type=entity_type,
                original_hash=original_hash,
                source_name=source_name,
            )
            return mapping

        # ON CONFLICT DO NOTHING returned no row: another concurrent writer won the insert race.
        winner = await self.lookup(entity_type, original_hash)
        if winner is None:
            raise RuntimeError(f"Missing pseudonym mapping after conflict for entity_type='{entity_type}'.")
        if source_name is not None and source_name not in winner.source_names:
            return await self._append_source_name(winner, source_name)
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
