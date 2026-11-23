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

# --- App modules ---
from aie_estimator.ingestion.catalog.models import CatalogDecision, CatalogSource, DataCatalog
from aie_estimator.ingestion.documents.models import Document
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.normalizers.canonical import normalize_documents
from aie_estimator.ingestion.parsers.protocol import ParseContext
from aie_estimator.ingestion.parsers.registry import ParserRegistry


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
) -> list[Document]:
    """
    Execute the ``loaders -> parsers -> normalizers`` pipeline for a single catalog source.

    :param catalog: Validated DataCatalog snapshot.
    :param source_name: Logical name of the source to ingest.
    :param loader: Filesystem loader resolving source blobs.
    :param registry: ParserRegistry providing the parser for ``source.format``.

    :return: List of canonicalized Document instances.
    :raises IngestionRejected: If the source is unknown or its decision is not ``include``.
    """
    source = require_included_source(catalog, source_name)

    parser = registry.get(source.format)
    context = ParseContext(
        source=source,
        source_version=catalog.version,
        ingested_at=datetime.now(UTC),
    )

    documents: list[Document] = []
    for blob in loader.iter_blobs(source.location, {source.format}):
        parsed = parser.parse(blob, context)
        documents.extend(
            normalize_documents(
                parsed,
                source=source,
                source_version=context.source_version,
                ingested_at=context.ingested_at,
                source_location=blob.relative_path,
            )
        )
    return documents
