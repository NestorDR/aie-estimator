# src/aie_estimator/ingestion/cleaning/policy.py

"""
Validation policy router: repair (prior stage) / quarantine / discard.

Uses Pandera ``lazy=True`` so validation collects every failure across all rows
and columns in one pass rather than aborting on the first error. That full
failure set lets the router partition cleaned records into three buckets and
populate a JSON-serializable ``report`` persisted on ``ingestion_jobs``:

* ``valid``       — passes every schema and cross-column check.
* ``quarantined`` — recoverable failures (e.g., ``client_name`` mapped to ``pd.NA``)
  and divergent ``budget_id`` duplicates; persisted in ``quarantined_records``
  for human review without entering RAG.
* ``discarded``   — structural or out-of-range failures (e.g., negative
  ``total_amount``, malformed IDs, phase-sum mismatches); dropped and counted
  in ``report`` without occupying quarantine storage.
"""

# --- Python Standard Library ---
from dataclasses import dataclass, field
from typing import Any, Final

# --- Third Party Libraries ---
import pandas as pd
import pandera.pandas as pa
import structlog

# --- App modules ---
from aie_estimator.ingestion.cleaning.budget_records import BudgetCleaningResult, serialize_record_payload
from aie_estimator.ingestion.cleaning.schemas import BudgetRecord
from aie_estimator.foundation.persistence.repositories.quarantine import QuarantinedRecordInput

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

# Quarantine = recoverable or ambiguous records kept for human review;
QUARANTINE_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "not_nullable",          # required text field (e.g. client_name) became pd.NA after null cleaning
        "no_default",            # required column value missing without default value
        "column_in_dataframe",   # expected column missing from input DataFrame
        "str_length",            # string length outside allowed min/max limits
    }
)
# Discard = structurally invalid or out-of-range records dropped with metrics.
DISCARD_CHECKS: Final[frozenset[str]] = frozenset(
    {
        "in_range",                  # numeric value outside allowed [min, max] limits
        "less_than_or_equal_to",     # total_amount exceeds upper limit
        "greater_than_or_equal_to",  # total_amount is negative
        "isin",                      # unsupported currency outside allowed set
        "str_matches",               # malformed budget_id or client_code pattern
        "phases_sum_matches_total",  # cross-column mismatch: sum(phases.amount) != total_amount
    }
)


@dataclass(frozen=True)
class ValidationResult:
    """Partitioned validation output and JSON-serializable summary report."""

    valid: pd.DataFrame
    quarantined: pd.DataFrame
    discarded: pd.DataFrame
    quarantine_inputs: list[QuarantinedRecordInput] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def validate_with_policy(
    cleaning_result: BudgetCleaningResult,
    schema: pa.DataFrameSchema = BudgetRecord,
) -> ValidationResult:
    """
    Validate cleaned budget rows lazily and route failures by policy.

    :param cleaning_result: Output of :func:`clean_budget_records`.
    :param schema: Pandera DataFrameSchema to enforce.

    :return: ValidationResult containing valid, quarantined, and discarded partitions.
    """
    clean_df = cleaning_result.clean_df
    divergent_df = cleaning_result.divergent_df
    failures_by_check: dict[str, int] = {}

    if clean_df.empty:
        valid_df = clean_df.copy()
        quarantined_df = clean_df.iloc[0:0].copy()
        discarded_df = clean_df.iloc[0:0].copy()
        quarantine_inputs: list[QuarantinedRecordInput] = []
    else:
        try:
            valid_df = schema.validate(clean_df, lazy=True)
            quarantined_df = clean_df.iloc[0:0].copy()
            discarded_df = clean_df.iloc[0:0].copy()
            quarantine_inputs = []
        except pa.errors.SchemaErrors as exc:
            valid_df, quarantined_df, discarded_df, schema_reasons = _route_schema_errors(
                clean_df, exc, failures_by_check
            )
            quarantine_inputs = _build_quarantine_inputs(quarantined_df, schema_reasons)

    if not divergent_df.empty:
        failures_by_check["divergent_duplicate"] = len(divergent_df)
        quarantine_inputs.extend(
            _build_quarantine_inputs(divergent_df, {i: ["divergent_duplicate"] for i in divergent_df.index})
        )
        quarantined_df = pd.concat([quarantined_df, divergent_df], ignore_index=True)

    input_rows = len(clean_df) + len(divergent_df) + cleaning_result.exact_duplicates_collapsed
    report: dict[str, Any] = {
        "input_rows": input_rows,
        "valid_rows": len(valid_df),
        "quarantined_rows": len(quarantined_df),
        "discarded_rows": len(discarded_df),
        "exact_duplicates_collapsed": cleaning_result.exact_duplicates_collapsed,
        "divergent_duplicates_quarantined": cleaning_result.divergent_duplicates_quarantined,
        "failures_by_check": failures_by_check,
    }

    log.info(
        "ingestion_cleaning_completed",
        cleaner="tabular",
        input_rows=input_rows,
        valid_rows=len(valid_df),
        quarantined_rows=len(quarantined_df),
        discarded_rows=len(discarded_df),
    )

    return ValidationResult(
        valid=valid_df,
        quarantined=quarantined_df,
        discarded=discarded_df,
        quarantine_inputs=quarantine_inputs,
        report=report,
    )


def _route_schema_errors(
    df: pd.DataFrame,
    err: pa.errors.SchemaErrors,
    failures_by_check: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    indices_quarantine: set[int] = set()
    indices_discard: set[int] = set()
    reasons_by_index: dict[int, list[str]] = {}
    seen_row_checks: set[tuple[int | None, str]] = set()

    for _, row in err.failure_cases.iterrows():
        check_name = str(row.get("check", "unknown"))
        idx = row.get("index")
        row_idx: int | None = None
        if idx is not None and not pd.isna(idx):
            try:
                row_idx = int(idx)
            except (TypeError, ValueError):
                row_idx = None

        # Pandera emits one failure_cases row per column for DataFrame-level checks;
        # deduplicate (row_idx, check_name) so metrics count once per failing row.
        if (row_idx, check_name) not in seen_row_checks:
            seen_row_checks.add((row_idx, check_name))
            failures_by_check[check_name] = failures_by_check.get(check_name, 0) + 1

        # Schema-level errors (e.g. missing columns) have no row index but are still recorded above.
        if row_idx is None:
            continue

        reasons_by_index.setdefault(row_idx, [])
        if check_name not in reasons_by_index[row_idx]:
            reasons_by_index[row_idx].append(check_name)

        if any(disc in check_name for disc in DISCARD_CHECKS):
            indices_discard.add(row_idx)
        else:
            # Known QUARANTINE_CHECKS and any unrecognized check default to quarantine for review.
            indices_quarantine.add(row_idx)

    # Discard always wins if a row hits both discard and quarantine checks.
    indices_quarantine -= indices_discard
    discarded_mask = df.index.isin(indices_discard)
    quarantined_mask = df.index.isin(indices_quarantine)
    valid_mask = ~(discarded_mask | quarantined_mask)

    return (
        df.loc[valid_mask].reset_index(drop=True),
        df.loc[quarantined_mask].copy(),
        df.loc[discarded_mask].reset_index(drop=True),
        reasons_by_index,
    )


def _build_quarantine_inputs(
    df: pd.DataFrame,
    reasons_by_index: dict[int, list[str]],
) -> list[QuarantinedRecordInput]:
    items: list[QuarantinedRecordInput] = []
    for idx, row in df.iterrows():
        row_idx = int(idx) if isinstance(idx, (int, float)) else 0
        record_key = str(row.get("budget_id") or row.get("_source_location") or f"row-{row_idx}")
        content_hash = str(row.get("content_hash") or "")
        reasons = reasons_by_index.get(row_idx, ["unknown"])
        items.append(
            QuarantinedRecordInput(
                record_key=record_key,
                content_hash=content_hash,
                failure_reasons=reasons,
                raw_payload=serialize_record_payload(row),
            )
        )
    return items
