# src/aie_estimator/ingestion/pii/forget.py

"""
GDPR Article 17 erasure service helper and CLI entrypoint.

Hashes the requested plaintext entity value with the configured HMAC-SHA256 salt, deletes
the corresponding mapping row from the backing ``AsyncMappingStore``, and emits a
structured audit event without ever logging plaintext PII.
"""

# --- Python Standard Library ---
import argparse
import asyncio
import sys
from typing import Final

# --- Third Party Libraries ---
from pydantic import SecretStr
import structlog

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.pii.mapping_store import AsyncMappingStore, PostgresMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import compute_pii_hash, resolve_pii_salt
from aie_estimator.logging import configure_logging
from aie_estimator.persistence.database import close_pool, open_pool
from aie_estimator.persistence.repositories.mappings import MappingsRepository
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)


async def forget_entity(
    entity_type: PiiFlag | str,
    original_value: str,
    *,
    salt: SecretStr | str,
    store: AsyncMappingStore,
) -> bool:
    """
    Erase a pseudonym mapping for ``(entity_type, original_value)`` and emit a GDPR audit event.

    :param entity_type: Canonical ``PiiFlag`` or entity type string.
    :param original_value: Plaintext PII value requested for erasure (never logged).
    :param salt: Secret HMAC salt matching the salt used during pseudonymization.
    :param store: Active ``AsyncMappingStore`` holding pseudonym mappings.

    :return: True if a mapping row was deleted, False when no matching row existed.
    :raises ValueError: If ``salt`` is empty or ``original_value`` is empty.
    """
    if not original_value:
        raise ValueError("original_value must not be empty.")

    normalized_type = entity_type.value if isinstance(entity_type, PiiFlag) else entity_type
    original_hash = compute_pii_hash(original_value, salt)
    deleted = await store.forget(normalized_type, original_hash)

    if deleted:
        log.info("ingestion_pii_erasure_completed", entity_type=normalized_type, original_hash=original_hash)
    else:
        log.info("ingestion_pii_erasure_not_found", entity_type=normalized_type, original_hash=original_hash)
    return deleted


async def _run_cli_forget(entity_type: PiiFlag, original_value: str) -> bool:
    """
    Open the PostgreSQL pool, execute ``forget_entity``, and close the pool cleanly.

    :param entity_type: Validated ``PiiFlag`` category.
    :param original_value: Plaintext PII value to erase.

    :return: True if a mapping row was deleted, False otherwise.
    """
    settings = get_settings()
    configure_logging(settings)
    salt = resolve_pii_salt(settings)
    pool = await open_pool(settings)
    try:
        store = PostgresMappingStore(MappingsRepository(pool))
        return await forget_entity(entity_type, original_value, salt=salt, store=store)
    finally:
        await close_pool(pool)


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint for executing GDPR Article 17 erasure against PostgreSQL.

    :param argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    :return: Exit code ``0`` on completion (whether erased or absent).
    """
    parser = argparse.ArgumentParser(
        prog="python -m aie_estimator.ingestion.pii.forget",
        description="Erase a GDPR pseudonym mapping by entity type and plaintext value.",
    )
    parser.add_argument(
        "--entity-type",
        required=True,
        choices=[flag.value for flag in PiiFlag],
        help="Canonical PiiFlag entity type (e.g., PERSON, EMAIL_ADDRESS, CLIENT_CODE).",
    )
    parser.add_argument(
        "--value",
        required=True,
        help="Plaintext PII value to hash and erase from pseudonym_mappings.",
    )
    args = parser.parse_args(argv)
    entity_type = PiiFlag(args.entity_type)
    # psycopg async connections require SelectorEventLoop on Windows (ProactorEventLoop is unsupported).
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    deleted = asyncio.run(_run_cli_forget(entity_type, args.value), loop_factory=loop_factory)
    status_label = "erased" if deleted else "not found"
    print(f"GDPR mapping {status_label} for entity_type={entity_type.value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
