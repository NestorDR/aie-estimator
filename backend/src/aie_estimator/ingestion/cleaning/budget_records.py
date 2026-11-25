# src/aie_estimator/ingestion/cleaning/budget_records.py

"""
Tabular cleaning and content-hash deduplication for raw JSON budget records.

Applies narrow, side-effect-free transformations over decoded JSON payloads *before*
Pandera validation and Markdown rendering (nulls before dedup, coercions before
schema checks):

- **Null placeholders** — maps ``"TBD"``, ``"N/A"``, ``""``, etc. to ``pd.NA``
  so disguised nulls do not enter embeddings as real content.
- **Currency normalization** — collapses aliases/casing (``"eur"``, ``"€"`` ->
  ``"EUR"``) while keeping unknown codes uppercase for schema rejection.
- **Date coercion** — parses mixed ISO (``2024-02-14``) and ``DD/MM/YYYY``
  (``12/03/2024``) dates with ``errors="coerce"`` so unparseable dates become
  ``NaT`` for downstream routing rather than crashing the batch.
- **Numeric coercion** — coerces string/number ``total_amount`` values to
  ``float64`` with ``errors="coerce"``.
- **Content-hash deduplication** — hashes canonical business fields with SHA-256
  and groups by ``budget_id``; identical-hash duplicates collapse silently,
  while divergent revisions (same ``budget_id``, different ``content_hash``)
  route to ``divergent_df`` for quarantine review.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final

# --- Third Party Libraries ---
import pandas as pd

# --- App modules ---
from aie_estimator.utils.constants import NULL_PLACEHOLDERS

TEXT_NULL_COLUMNS: Final[tuple[str, ...]] = ("client_name", "contact", "contact_email", "notes")
CURRENCY_ALIASES: Final[dict[str, str]] = {
    "€": "EUR",
    "eur": "EUR",
    "euros": "EUR",
    "$": "USD",
    "usd": "USD",
    "£": "GBP",
    "gbp": "GBP",
}


@dataclass(frozen=True)
class BudgetCleaningResult:
    """Output of tabular budget cleaning and content-hash deduplication."""

    clean_df: pd.DataFrame
    divergent_df: pd.DataFrame
    exact_duplicates_collapsed: int
    divergent_duplicates_quarantined: int


def clean_budget_records(records: Iterable[dict[str, Any]]) -> BudgetCleaningResult:
    """
    Normalize raw budget dictionaries and separate exact vs. divergent duplicates.

    :param records: Iterable of decoded JSON budget dictionaries.

    :return: BudgetCleaningResult with deduplicated ``clean_df`` and ``divergent_df``.
    """
    df = pd.DataFrame(list(records))
    if df.empty:
        return BudgetCleaningResult(
            clean_df=df,
            divergent_df=df.copy(),
            exact_duplicates_collapsed=0,
            divergent_duplicates_quarantined=0,
        )

    out = df.copy()

    # Map disguised nulls to pd.NA while preserving casing on valid strings.
    for column in TEXT_NULL_COLUMNS:
        if column in out.columns:
            out[column] = out[column].apply(_normalize_nullable_text)

    # Canonicalize currency aliases/casing; unknown codes stay uppercase for schema rejection.
    if "currency" in out.columns:
        out["currency"] = out["currency"].apply(_normalize_currency)

    # Coerce ISO and DD/MM/YYYY dates into UTC-naive timestamps; failures become NaT.
    if "signed_at" in out.columns:
        out["signed_at"] = pd.to_datetime(out["signed_at"], format="mixed", dayfirst=True, errors="coerce")

    # Coerce numeric amounts permissively.
    if "total_amount" in out.columns:
        out["total_amount"] = pd.to_numeric(out["total_amount"], errors="coerce").astype("float64")

    # Compute deterministic SHA-256 content_hash across business fields.
    # _source_location is excluded so identical payloads in different files yield the same hash.
    out["content_hash"] = [_compute_row_hash(row) for _, row in out.iterrows()]

    # Deduplicate by budget_id: collapse exact hashes, quarantine divergent revisions.
    if "budget_id" not in out.columns:
        return BudgetCleaningResult(
            clean_df=out.reset_index(drop=True),
            divergent_df=out.iloc[0:0].copy(),
            exact_duplicates_collapsed=0,
            divergent_duplicates_quarantined=0,
        )

    keep_indices: list[Any] = []
    divergent_indices: list[Any] = []
    exact_collapsed = 0

    # Preserve input order and retain NaN/missing budget_id rows (dropna=False) so Pandera routes them.
    for _, group in out.groupby("budget_id", sort=False, dropna=False):
        if len(group) == 1:
            keep_indices.append(group.index[0])
            continue
        if group["content_hash"].nunique(dropna=False) == 1:
            keep_indices.append(group.index[0])
            exact_collapsed += len(group) - 1
        else:
            # Never guess a winner among conflicting revisions; quarantine every row for human review.
            divergent_indices.extend(group.index.tolist())

    clean_df = out.loc[keep_indices].reset_index(drop=True)
    divergent_df = out.loc[divergent_indices].reset_index(drop=True)
    return BudgetCleaningResult(
        clean_df=clean_df,
        divergent_df=divergent_df,
        exact_duplicates_collapsed=exact_collapsed,
        divergent_duplicates_quarantined=len(divergent_df),
    )


def _normalize_nullable_text(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NULL_PLACEHOLDERS:
            return pd.NA
        return stripped
    return value


def _normalize_currency(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    if text.lower() in NULL_PLACEHOLDERS:
        return pd.NA
    return CURRENCY_ALIASES.get(text.lower(), text.upper())


def serialize_record_payload(
    row: pd.Series,
    *,
    exclude_keys: frozenset[str] = frozenset({"_source_location"}),
) -> dict[str, Any]:
    """
    Convert a DataFrame row into a JSON-serializable dictionary.

    :param row: Budget DataFrame row.
    :param exclude_keys: Column names omitted from the serialized dictionary.

    :return: Dictionary with missing values mapped to ``None`` and timestamps to ISO strings.
    """
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if key in exclude_keys:
            continue
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            payload[str(key)] = None
        elif isinstance(value, pd.Timestamp):
            payload[str(key)] = value.isoformat()
        else:
            payload[str(key)] = value
    return payload


def _compute_row_hash(row: pd.Series) -> str:
    payload = serialize_record_payload(row, exclude_keys=frozenset({"content_hash", "_source_location"}))
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
