# src/aie_estimator/ingestion/inspect_budgets.py

"""
Read-only offline CLI inspector for cleaned, pseudonymized Session 6 budget JSON.

Runs the canonical ``ingest_source`` pipeline in memory and optionally applies
``ConsistentPseudonymizer`` backed by ``InMemoryMappingStore`` so developers can
visually inspect or pipe the validated ``CleanedBudget`` JSON array on ``stdout``
without writing to PostgreSQL or disk. Diagnostic logs are emitted to ``stderr``
via ``structlog`` using homogeneous ``ingestion_inspection_*`` event names.
"""

# --- Python Standard Library ---
import argparse
import asyncio
from contextlib import redirect_stdout
import json
import logging
import sys
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.orchestrator import IngestionRejected, ingest_source
from aie_estimator.ingestion.parsers.budget_json import CleanedBudget
from aie_estimator.ingestion.parsers.registry import default_registry
from aie_estimator.ingestion.pii.analyzer import PiiAnalyzer, PresidioPiiAnalyzer
from aie_estimator.ingestion.pii.mapping_store import InMemoryMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer, resolve_pii_salt
from aie_estimator.logging import configure_logging
from aie_estimator.settings import Settings, get_settings

DEFAULT_SOURCE_NAME: Final[str] = "historical_budgets"

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)


async def inspect_cleaned_budgets(
    *,
    settings: Settings | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    pseudonymize: bool = True,
    analyzer: PiiAnalyzer | None = None,
) -> list[CleanedBudget]:
    """
    Ingest, clean, validate, and optionally pseudonymize budgets in memory.

    :param settings: Optional Settings override; defaults to ``get_settings()``.
    :param source_name: Logical catalog source name to inspect.
    :param pseudonymize: Whether to apply in-memory Presidio PII pseudonymization.
    :param analyzer: Optional PiiAnalyzer override (useful for deterministic tests).

    :return: List of validated ``CleanedBudget`` models.
    :raises IngestionRejected: If the source is unknown or its decision is not ``include``.
    :raises ValueError: If pseudonymization is enabled without a configured HMAC salt.
    """
    active_settings = settings or get_settings()
    log.info("ingestion_inspection_started", source_name=source_name, pseudonymize=pseudonymize)

    catalog = load_catalog(active_settings.catalog_path)
    result = ingest_source(
        catalog=catalog,
        source_name=source_name,
        loader=FileSystemLoader(active_settings.ingestion_data_root),
        registry=default_registry(),
    )
    # Respect catalog sensitivity stamped by normalize_documents: skip Presidio on non-PII sources.
    should_pseudonymize = pseudonymize and any(doc.metadata.contains_pii for doc in result.documents)

    if should_pseudonymize:
        salt = resolve_pii_salt(active_settings)
        active_analyzer: PiiAnalyzer = analyzer or PresidioPiiAnalyzer(
            spacy_model=active_settings.pii_spacy_model,
            language=active_settings.pii_language,
            score_threshold=active_settings.pii_confidence_threshold,
        )
        # In-memory store guarantees this read-only inspector never mutates PostgreSQL mappings.
        pseudonymizer = ConsistentPseudonymizer(
            analyzer=active_analyzer,
            mapping_store=InMemoryMappingStore(),
            salt=salt,
            faker_locale=active_settings.pii_faker_locale,
        )
        documents = await pseudonymizer.pseudonymize_documents(result.documents, source_name=source_name)
    else:
        documents = result.documents

    # Validate post-pseudonymization so emitted JSON enforces the strict CleanedBudget contract on scrubbed values.
    budgets = [CleanedBudget.model_validate(doc.metadata.extra) for doc in documents]
    log.info(
        "ingestion_inspection_completed",
        source_name=source_name,
        budgets_count=len(budgets),
        quarantined_count=len(result.quarantined),
        discarded_count=int(result.report.get("discarded_rows", 0)),
        pseudonymized=should_pseudonymize,
    )
    return budgets


def main(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint printing cleaned, pseudonymized budgets as JSON to ``stdout``.

    :param argv: Optional CLI argument list (defaults to ``sys.argv[1:]``).

    :return: Exit code ``0`` on success, ``1`` on catalog/configuration failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m aie_estimator.ingestion.inspect_budgets",
        description="Inspect cleaned, pseudonymized Session 6 budgets as JSON on stdout (read-only).",
    )
    parser.add_argument(
        "--no-pii",
        action="store_true",
        help="Skip PII pseudonymization and output cleaned raw budgets.",
    )
    args = parser.parse_args(argv)

    # Route early structlog bootstrap events emitted by get_settings() (before
    # configure_logging binds stdlib logging to stderr) so stdout stays valid JSON from byte 0.
    with redirect_stdout(sys.stderr):
        settings = get_settings()
    configure_logging(settings)
    # Suppress verbose spaCy/Presidio unmapped-entity warnings (e.g. CARDINAL, PRODUCT) on stderr.
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

    try:
        budgets = asyncio.run(
            inspect_cleaned_budgets(
                settings=settings,
                pseudonymize=not args.no_pii,
            )
        )
    except (IngestionRejected, FileNotFoundError, ValueError) as exc:
        log.error(
            "ingestion_inspection_failed",
            source_name=DEFAULT_SOURCE_NAME,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = [budget.model_dump(mode="json") for budget in budgets]
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
