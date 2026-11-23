# src/aie_estimator/ingestion/normalizers/canonical.py

"""
Placeholder canonicalization stage for the ``loaders -> parsers -> normalizers`` flow.

Currently enforces authoritative catalog metadata override and SHA-256 ``content_hash``
calculation. Later, we will introduce dedicated cleaning and schema validation
before this stage to address the four recurring families of dirtiness in enterprise RAG
systems:

- Heterogeneity of format: ``€ vs EUR vs eur``
- Duplicates with divergence: ``Same id, different totals``
- Disguised nulls: ``'N/A', 'TBD', '', '-'``
- Out-of-range values: ``total: -50000, hours: 9999999``
"""

# --- Python Standard Library ---
from collections.abc import Iterable
from datetime import datetime

# --- App modules ---
from aie_estimator.ingestion.catalog.models import CatalogSource
from aie_estimator.ingestion.documents.models import Document, compute_content_hash


def normalize_documents(
    documents: Iterable[Document],
    *,
    source: CatalogSource,
    source_version: str,
    ingested_at: datetime,
    source_location: str,
) -> list[Document]:
    """
    Canonicalize parsed Documents under authoritative catalog metadata and SHA-256 content hashes.

    :param documents: Iterable of parsed Document instances.
    :param source: Authoritative CatalogSource entry governing the run.
    :param source_version: Active DataCatalog version string.
    :param ingested_at: UTC timestamp for the current ingestion run.
    :param source_location: Resolved relative file path applied to each document.

    :return: List of canonicalized Document instances.
    """
    normalized: list[Document] = []
    for document in documents:
        normalized_metadata = document.metadata.model_copy(
            update={
                "source_name": source.name,
                "source_version": source_version,
                "source_location": source_location,
                "ingested_at": ingested_at,
                "lineage": source.lineage,
                "contains_pii": source.sensitivity.has_pii,
                "pii_flags": source.sensitivity.pii_flags,
                "access_level": source.sensitivity.access_level,
            }
        )
        normalized.append(
            document.model_copy(
                update={
                    "content_hash": compute_content_hash(document.text),
                    "metadata": normalized_metadata,
                }
            )
        )
    return normalized
