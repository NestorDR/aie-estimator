# src/aie_estimator/ingestion/documents/models.py

"""
The canonical ``Document`` model.

Designed flat on purpose: a single ``DocumentMetadata`` sub-object holds the
structured fields that every consumer needs; everything format-specific lives
in ``extra: dict[str, Any]``. That escape hatch is what lets us keep the
contract uniform across JSON, TXT, XLSX and future formats without inventing
sub-types per source.

The ``id`` is the deterministic identifier assigned per document; it must be
stable across re-ingestions so the downstream index can dedupe.
"""

# --- Python Standard Library ---
from datetime import datetime
import hashlib
from typing import Any

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field

# --- App modules ---
from aie_estimator.ingestion.catalog.models import AccessLevel, PiiFlag


def compute_content_hash(text: str) -> str:
    """
    Compute the deterministic SHA-256 hex digest for document text.

    Used by downstream deduplication and incremental re-indexing to detect unchanged
    or duplicate content without comparing full document bodies.

    :param text: Canonical document text content.

    :return: Lowercase 64-character SHA-256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentMetadata(BaseModel):
    """Canonical provenance, sensitivity, and format-specific metadata for an ingested document."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_version: str
    source_location: str
    ingested_at: datetime
    lineage: tuple[str, ...] = ()
    contains_pii: bool = False
    pii_flags: tuple[PiiFlag, ...] = ()
    access_level: AccessLevel = "internal"
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """Canonical document record emitted by the offline ingestion pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    content_hash: str = ""
    metadata: DocumentMetadata
