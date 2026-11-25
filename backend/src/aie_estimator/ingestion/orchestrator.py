# src/aie_estimator/ingestion/orchestrator.py

"""
Orchestrator: glues catalog + loader + parser + normalizer into a single ingestion run.

The contract is intentionally narrow:

    ingest_source(*, catalog, source_name, loader, registry) -> list[Document]

Three guarantees:

1. The source must exist in the catalog and have ``decision == include``.
   Anything else raises ``IngestionRejected`` — the router maps that to 400.
2. The function is *synchronous* and free of DB dependencies so it can be offloaded
   via ``asyncio.to_thread(...)`` from ``ingestion_router.py``.
3. ``ingestion_router.py`` owns the async ``JobsRepository`` lifecycle
   (``pending -> running -> completed | failed``) on the ``SelectorEventLoop``.
"""

# --- Python Standard Library ---
from datetime import UTC, datetime
from typing import Any

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field

# --- App modules ---
from aie_estimator.ingestion.catalog.models import CatalogDecision, CatalogSource, DataCatalog
from aie_estimator.ingestion.cleaning.budget_records import clean_budget_records, serialize_record_payload
from aie_estimator.ingestion.cleaning.policy import validate_with_policy
from aie_estimator.ingestion.cleaning.text import clean_text_documents
from aie_estimator.ingestion.documents.models import Document
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.normalizers.canonical import normalize_documents
from aie_estimator.ingestion.parsers.budget_json import BudgetJsonParser
from aie_estimator.ingestion.parsers.protocol import ParseContext
from aie_estimator.ingestion.parsers.registry import ParserRegistry
from aie_estimator.persistence.repositories.quarantine import QuarantinedRecordInput


class IngestionResult(BaseModel):
    """Immutable outcome of a synchronous ingestion run."""

    model_config = ConfigDict(frozen=True)

    documents: list[Document]
    quarantined: list[QuarantinedRecordInput] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)


class IngestionRejected(Exception):  # noqa: N818
    """Raised when an ingestion run is requested for an unknown or non-included catalog source."""

    def __init__(self, message: str, *, reason: str = "ingestion_rejected") -> None:
        """
        Initialize the rejection error with a structured reason and message.

        :param message: Human-readable explanation of why the source was rejected.
        :param reason: Machine-readable rejection reason code.
        """
        super().__init__(message)
        self.reason = reason
        self.message = message


def require_included_source(catalog: DataCatalog, source_name: str) -> CatalogSource:
    """
    Resolve ``source_name`` from ``catalog`` and verify its decision is ``include``.

    :param catalog: Validated DataCatalog snapshot.
    :param source_name: Logical source identifier to validate.

    :return: Approved CatalogSource entry.
    :raises IngestionRejected: If the source is unknown or its decision is not ``include``.
    """
    source = catalog.find(source_name)
    if source is None:
        raise IngestionRejected(
            f"Source '{source_name}' was not found in catalog.",
            reason="unknown_source",
        )
    if source.decision is not CatalogDecision.INCLUDE:
        raise IngestionRejected(
            f"Source '{source_name}' has decision='{source.decision.value}'; only 'include' sources can be ingested.",
            reason="source_not_included",
        )
    return source


def ingest_source(
    *,
    catalog: DataCatalog,
    source_name: str,
    loader: FileSystemLoader,
    registry: ParserRegistry,
) -> IngestionResult:
    """
    Execute ``loaders -> parsers -> cleaning -> normalizers`` for a single catalog source.

    :param catalog: Validated DataCatalog snapshot.
    :param source_name: Logical name of the source to ingest.
    :param loader: Filesystem loader resolving source blobs.
    :param registry: ParserRegistry providing the parser for ``source.format``.

    :return: IngestionResult with canonical Documents, quarantined records, and validation report.
    :raises IngestionRejected: If the source is unknown or its decision is not ``include``.
    """
    source = require_included_source(catalog, source_name)
    parser = registry.get(source.format)
    context = ParseContext(
        source=source,
        source_version=catalog.version,
        ingested_at=datetime.now(UTC),
    )
    blobs = list(loader.iter_blobs(source.location, {source.format}))

    # Tabular JSON budgets must be cleaned, deduplicated, and validated as a batch
    # before Markdown rendering so dirty values and divergent revisions never enter Document.text.
    if isinstance(parser, BudgetJsonParser):
        raw_records = [parser.decode_record(blob) for blob in blobs]
        cleaned = clean_budget_records(raw_records)
        validation = validate_with_policy(cleaned)
        rendered_docs = [
            parser.render_record(
                serialize_record_payload(row),
                source_location=str(row.get("_source_location") or source.location),
                context=context,
            )
            for _, row in validation.valid.iterrows()
        ]
        documents = normalize_documents(
            rendered_docs,
            source=source,
            source_version=context.source_version,
            ingested_at=context.ingested_at,
        )
        return IngestionResult(
            documents=documents,
            quarantined=validation.quarantine_inputs,
            report=validation.report,
        )

    parsed_docs: list[Document] = []
    for blob in blobs:
        for doc in parser.parse(blob, context):
            # Stamp authoritative blob path before batch text cleaning and normalization.
            parsed_docs.append(
                doc.model_copy(
                    update={"metadata": doc.metadata.model_copy(update={"source_location": blob.relative_path})}
                )
            )

    text_result = clean_text_documents(parsed_docs)
    normalized_docs = normalize_documents(
        text_result.documents,
        source=source,
        source_version=context.source_version,
        ingested_at=context.ingested_at,
    )
    return IngestionResult(documents=normalized_docs, quarantined=[], report=text_result.report)
