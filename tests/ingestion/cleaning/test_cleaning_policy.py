# tests/ingestion/cleaning/test_policy.py

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Any

# --- App modules ---
from aie_estimator.ingestion.cleaning.budget_records import clean_budget_records
from aie_estimator.ingestion.cleaning.policy import validate_with_policy


def test_validate_with_policy_routes_seed_budgets_accurately() -> None:
    """
    GIVEN all 6 seed budget JSON files under data/seed/budgets
    WHEN clean_budget_records and validate_with_policy execute
    THEN 0001/0002 are valid, 0003 and 0005-v1/v2 are quarantined, and 0004 is discarded.
    """
    seed_dir = Path("data/seed/budgets")
    records: list[dict[str, Any]] = []
    for path in sorted(seed_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_source_location"] = f"budgets/{path.name}"
        records.append(payload)

    cleaned = clean_budget_records(records)
    result = validate_with_policy(cleaned)

    assert sorted(result.valid["budget_id"].tolist()) == ["BUDGET-2024-0001", "BUDGET-2024-0002"]
    assert sorted(result.quarantined["budget_id"].tolist()) == [
        "BUDGET-2024-0003",
        "BUDGET-2024-0005",
        "BUDGET-2024-0005",
    ]
    assert result.discarded["budget_id"].tolist() == ["BUDGET-2024-0004"]
    assert result.report["input_rows"] == 6
    assert result.report["valid_rows"] == 2
    assert result.report["quarantined_rows"] == 3
    assert result.report["discarded_rows"] == 1
    assert result.report["divergent_duplicates_quarantined"] == 2
    assert len(result.quarantine_inputs) == 3


def test_validate_with_policy_discards_phase_sum_mismatch() -> None:
    """
    GIVEN a budget record whose phases.amount sum disagrees with total_amount
    WHEN validate_with_policy runs
    THEN the cross-column check routes the row to discarded.
    """
    bad_record = {
        "budget_id": "BUDGET-2024-0099",
        "client_name": "Acme Corp",
        "client_code": "CLI-0099",
        "currency": "EUR",
        "total_amount": 50000,
        "signed_at": "2024-05-01",
        "phases": [{"name": "Discovery", "weeks": 2, "amount": 10000}],
    }
    cleaned = clean_budget_records([bad_record])
    result = validate_with_policy(cleaned)

    assert result.valid.empty
    assert result.quarantined.empty
    assert result.discarded["budget_id"].tolist() == ["BUDGET-2024-0099"]
    assert result.report["failures_by_check"]["phases_sum_matches_total"] == 1
