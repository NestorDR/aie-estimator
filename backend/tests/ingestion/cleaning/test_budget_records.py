# tests/ingestion/cleaning/test_budget_records.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
import pandas as pd
import pytest

# --- App modules ---
from aie_estimator.ingestion.cleaning.budget_records import clean_budget_records


def test_clean_budget_records_empty_input() -> None:
    """
    GIVEN an empty iterable of budget records
    WHEN clean_budget_records is called
    THEN empty clean and divergent DataFrames with zero duplicate counts are returned.
    """
    result = clean_budget_records([])
    assert result.clean_df.empty
    assert result.divergent_df.empty
    assert result.exact_duplicates_collapsed == 0
    assert result.divergent_duplicates_quarantined == 0


@pytest.mark.parametrize(
    ("raw_client", "raw_currency", "raw_date", "raw_amount", "expected_na_client", "expected_currency"),
    [
        ("TBD", "eur", "12/03/2024", "28000", True, "EUR"),
        ("N/A", "€", "2024-02-14", 45000, True, "EUR"),
        ("Acme Spain Ltd.", "$", "2024-05-02", "12500.5", False, "USD"),
    ],
    ids=["tbd-eur-dmy", "na-euro-iso", "valid-usd-float"],
)
def test_clean_budget_records_normalizes_heterogeneity_and_nulls(
    raw_client: str,
    raw_currency: str,
    raw_date: str,
    raw_amount: str | int,
    expected_na_client: bool,
    expected_currency: str,
) -> None:
    """
    GIVEN raw budget records with disguised nulls, currency variants, mixed dates, and string amounts
    WHEN clean_budget_records runs
    THEN disguised nulls map to pd.NA, currencies normalize, dates parse, and amounts coerce to float.
    """
    record: dict[str, Any] = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": raw_client,
        "client_code": "CLI-0001",
        "currency": raw_currency,
        "total_amount": raw_amount,
        "signed_at": raw_date,
    }
    result = clean_budget_records([record])
    row = result.clean_df.iloc[0]
    assert pd.isna(row["client_name"]) is expected_na_client
    assert row["currency"] == expected_currency
    assert pd.notna(row["signed_at"])
    assert isinstance(row["total_amount"], float)


def test_clean_budget_records_collapses_exact_and_isolates_divergent_duplicates() -> None:
    """
    GIVEN exact duplicate records and divergent revisions sharing a budget_id
    WHEN clean_budget_records runs
    THEN exact duplicates collapse to one row and all divergent rows move to divergent_df.
    """
    exact_a = {
        "budget_id": "BUDGET-2024-0001",
        "client_name": "Acme",
        "client_code": "CLI-0001",
        "currency": "EUR",
        "total_amount": 10000,
        "signed_at": "2024-02-14",
        "_source_location": "budgets/a1.json",
    }
    exact_b = {**exact_a, "_source_location": "budgets/a2.json"}
    div_v1 = {
        "budget_id": "BUDGET-2024-0005",
        "client_name": "Toledo",
        "client_code": "CLI-0288",
        "currency": "EUR",
        "total_amount": 30000,
        "signed_at": "2024-04-10",
    }
    div_v2 = {**div_v1, "total_amount": 32000, "signed_at": "2024-04-12"}

    result = clean_budget_records([exact_a, exact_b, div_v1, div_v2])
    assert len(result.clean_df) == 1
    assert result.clean_df.iloc[0]["budget_id"] == "BUDGET-2024-0001"
    assert result.exact_duplicates_collapsed == 1
    assert len(result.divergent_df) == 2
    assert result.divergent_duplicates_quarantined == 2
