# src/aie_estimator/ingestion/pii/mapping_store.py

"""
Asynchronous mapping store abstraction backing consistent PII pseudonymization.

Only the salted HMAC-SHA256 digest of a detected value ever reaches persistence; plaintext
PII never leaves the pseudonymizer. That asymmetry makes GDPR Article 17 right-to-be-forgotten
auditable: deleting a row by ``(entity_type, original_hash)`` destroys reversibility without
ever having stored raw personal data.

Both persistent (PostgreSQL) and in-memory implementations guarantee idempotency across
repeated calls and track catalog ``source_name`` provenance so offline tests exercise the
same async contract without requiring a live database.
"""

# --- Python Standard Library ---
from collections.abc import Callable
from typing import Protocol, runtime_checkable

# --- App modules ---
from aie_estimator.persistence.repositories.mappings import MappingsRepository


@runtime_checkable
class AsyncMappingStore(Protocol):
    """Async contract for idempotent pseudonym persistence and GDPR Article 17 erasure."""

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
        *,
        source_name: str | None = None,
    ) -> str:
        """
        Return the existing pseudonym for ``(entity_type, original_hash)`` or persist a new one.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.
        :param new_pseudonym_factory: Zero-argument generator called only on cache misses.
        :param source_name: Optional catalog source identifier for provenance tracking.

        :return: Deterministic pseudonym string.
        """
        ...  # pragma: no cover

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        """
        Remove the mapping for ``(entity_type, original_hash)`` to honor GDPR Article 17.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: True if a mapping existed and was deleted, False otherwise.
        """
        ...  # pragma: no cover


class PostgresMappingStore:
    """Persistent ``AsyncMappingStore`` delegating to ``MappingsRepository``."""

    def __init__(self, repository: MappingsRepository) -> None:
        """
        Initialize the store with an async PostgreSQL ``MappingsRepository``.

        :param repository: Active MappingsRepository bound to ``AsyncConnectionPool``.
        """
        self._repository = repository

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
        *,
        source_name: str | None = None,
    ) -> str:
        """
        Resolve or create a durable pseudonym mapping in PostgreSQL.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.
        :param new_pseudonym_factory: Zero-argument generator invoked only on misses.
        :param source_name: Optional catalog source identifier recorded in ``source_names``.

        :return: Winning deterministic pseudonym string.
        """
        mapping = await self._repository.lookup_or_create(
            entity_type=entity_type,
            original_hash=original_hash,
            new_pseudonym_factory=new_pseudonym_factory,
            source_name=source_name,
        )
        return mapping.pseudonym

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        """
        Delete a mapping row from PostgreSQL.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: True when a row was deleted, False when absent.
        """
        return await self._repository.forget(entity_type, original_hash)


class InMemoryMappingStore:
    """In-memory ``AsyncMappingStore`` for deterministic offline tests and scripts."""

    def __init__(self) -> None:
        """Initialize an empty in-memory pseudonym registry."""
        self._mappings: dict[tuple[str, str], str] = {}

    async def lookup_or_create(
        self,
        entity_type: str,
        original_hash: str,
        new_pseudonym_factory: Callable[[], str],
        *,
        source_name: str | None = None,
    ) -> str:
        """
        Return an existing in-memory pseudonym or generate and store a new one.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.
        :param new_pseudonym_factory: Zero-argument generator invoked only on misses.
        :param source_name: Optional catalog source identifier (accepted for protocol parity).

        :return: Deterministic pseudonym string.
        """
        # In-memory store does not persist provenance; keep parameter for AsyncMappingStore protocol parity.
        del source_name
        key = (entity_type, original_hash)
        if key not in self._mappings:
            self._mappings[key] = new_pseudonym_factory()
        return self._mappings[key]

    async def forget(self, entity_type: str, original_hash: str) -> bool:
        """
        Delete an entry from the in-memory store.

        :param entity_type: Canonical PII entity category.
        :param original_hash: Salted HMAC-SHA256 digest of the original value.

        :return: True if the key existed and was removed, False otherwise.
        """
        return self._mappings.pop((entity_type, original_hash), None) is not None
