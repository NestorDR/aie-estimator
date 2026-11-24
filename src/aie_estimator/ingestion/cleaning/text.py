# src/aie_estimator/ingestion/cleaning/text.py

"""
Minimal non-tabular text cleaning and validation for ``Document`` streams.

Applies Unicode ``NFKC`` (Normalization Form Compatibility Composition) to collapse compatibility-equivalent 
characters into standard canonical forms (e.g. fullwidth ``"Ｈｅｌｌｏ"`` -> ``"Hello"``,
ligatures ``"ﬁ"`` -> ``"fi"``, non-breaking space ``\\u00a0`` -> ``" "``) so embeddings and null-placeholder
checks treat visually identical tokens uniformly. 

Strips control characters and trailing line whitespace without collapsing intentional line breaks, 
then filters disguised nulls and undersized snippets prior to canonical normalization.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final
import unicodedata

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.ingestion.documents.models import Document
from aie_estimator.utils.constants import NULL_PLACEHOLDERS

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

DEFAULT_MIN_TEXT_LENGTH: Final[int] = 10


@dataclass(frozen=True)
class TextCleaningResult:
    """Output of non-tabular text cleaning and length/placeholder filtering."""

    documents: list[Document]
    filtered_count: int
    report: dict[str, Any]


def clean_text_documents(
    documents: Iterable[Document],
    *,
    min_length: int = DEFAULT_MIN_TEXT_LENGTH,
) -> TextCleaningResult:
    """
    Normalize non-tabular document text and filter placeholders or undersized entries.

    :param documents: Iterable of parsed Document instances.
    :param min_length: Minimum required character count after normalization.

    :return: TextCleaningResult with valid normalized documents and summary report.
    """
    valid: list[Document] = []
    filtered_null = 0
    filtered_short = 0
    total = 0

    for doc in documents:
        total += 1
        cleaned = _normalize_text(doc.text)
        if cleaned.lower() in NULL_PLACEHOLDERS:
            filtered_null += 1
            continue
        if len(cleaned) < min_length:
            filtered_short += 1
            continue
        valid.append(doc.model_copy(update={"text": cleaned}))

    filtered_count = filtered_null + filtered_short
    report: dict[str, Any] = {
        "input_rows": total,
        "valid_rows": len(valid),
        "quarantined_rows": 0,
        "discarded_rows": filtered_count,
        "failures_by_check": {
            key: count
            for key, count in (("disguised_null", filtered_null), ("min_length", filtered_short))
            if count > 0
        },
    }
    log.info(
        "ingestion_cleaning_completed",
        cleaner="text",
        input_rows=total,
        valid_rows=len(valid),
        quarantined_rows=0,
        discarded_rows=filtered_count,
    )
    return TextCleaningResult(documents=valid, filtered_count=filtered_count, report=report)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    # Strip C0/C1 control characters (e.g. \x00) while preserving newlines and tabs for Markdown/turn structure.
    without_controls = "".join(
        ch for ch in normalized if ch in {"\n", "\t"} or unicodedata.category(ch)[0] != "C"
    )
    lines = [line.rstrip() for line in without_controls.splitlines()]
    return "\n".join(lines).strip()
