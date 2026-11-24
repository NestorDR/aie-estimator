# src/aie_estimator/ingestion/cleaning/schemas.py

"""
Pandera v2 schema contract for cleaned budget DataFrames.

Enforces ``strict=True`` and ``lazy=True`` validation with cross-column
``sum(phases.amount) == total_amount`` checks while declaring optional
rendering metadata columns (``phases``, ``contact``, ``contact_email``,
``notes``, ``content_hash``, ``_source_location``) so Markdown rendering
retains full context without triggering ``column_not_in_schema`` errors.
"""

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

BUDGET_ID_PATTERN: Final[str] = r"^BUDGET-\d{4}-\d{4}$"
CLIENT_CODE_PATTERN: Final[str] = r"^CLI-\d{4}$"
ALLOWED_CURRENCIES: Final[tuple[str, ...]] = ("EUR", "USD", "GBP")
MAX_BUDGET_AMOUNT: Final[float] = 10_000_000.0
_PHASE_SUM_TOLERANCE: Final[float] = 1e-2


def _check_phases_sum_matches_total(df: pd.DataFrame) -> pd.Series:
    """
    Verify that ``sum(phases.amount)`` matches ``total_amount`` when phases are present.

    :param df: Cleaned budget DataFrame.

    :return: Boolean Series indexed like ``df`` indicating per-row compliance.
    """
    if "phases" not in df.columns or "total_amount" not in df.columns:
        return pd.Series(True, index=df.index)

    results: list[bool] = []
    for _, row in df.iterrows():
        phases = row["phases"]
        total = row["total_amount"]
        # Skip rows without phases or with coerced NaN totals so column-level checks report the root cause.
        if not isinstance(phases, list) or not phases or pd.isna(total):
            results.append(True)
            continue
        try:
            phase_sum = sum(float(phase["amount"]) for phase in phases)
        except (KeyError, TypeError, ValueError):
            results.append(False)
            continue
        results.append(abs(phase_sum - float(total)) <= _PHASE_SUM_TOLERANCE)
    return pd.Series(results, index=df.index)


BudgetRecord: Final[DataFrameSchema] = DataFrameSchema(
    columns={
        "budget_id": Column(
            str,
            checks=Check.str_matches(BUDGET_ID_PATTERN),
            nullable=False,
            required=True,
        ),
        "client_name": Column(
            str,
            checks=Check.str_length(min_value=2, max_value=200),
            nullable=False,
            required=True,
        ),
        "client_code": Column(
            str,
            checks=Check.str_matches(CLIENT_CODE_PATTERN),
            nullable=False,
            required=True,
        ),
        "currency": Column(
            str,
            checks=Check.isin(ALLOWED_CURRENCIES),
            nullable=False,
            required=True,
        ),
        "total_amount": Column(
            float,
            checks=[Check.ge(0.0), Check.le(MAX_BUDGET_AMOUNT)],
            nullable=False,
            required=True,
        ),
        "signed_at": Column(
            pa.DateTime,
            nullable=False,
            required=True,
        ),
        "phases": Column(object, nullable=True, required=False),
        "contact": Column(str, nullable=True, required=False),
        "contact_email": Column(str, nullable=True, required=False),
        "notes": Column(str, nullable=True, required=False),
        "content_hash": Column(str, nullable=True, required=False),
        "_source_location": Column(str, nullable=True, required=False),
    },
    checks=[
        Check(
            _check_phases_sum_matches_total,
            name="phases_sum_matches_total",
        )
    ],
    strict=True,
    coerce=False,
)
