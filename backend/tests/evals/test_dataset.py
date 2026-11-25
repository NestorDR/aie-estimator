# tests/evals/test_dataset.py

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Any

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import ProjectType
from aie_estimator.sessions.tier_resolver import Tier
from evals.dataset import GoldenCase, load_dataset


def test_load_default_dataset_covers_all_project_types_and_tiers() -> None:
    """
    GIVEN the default Golden Dataset JSON file
    WHEN ``load_dataset()`` is called with no arguments
    THEN all entries validate and cover every ``ProjectType`` and ``Tier`` (both explicit and auto-resolved).
    """
    cases = load_dataset()

    assert len(cases) >= 10
    assert {case.project_type for case in cases} == set(ProjectType)
    assert {case.expected_tier for case in cases if case.expected_tier is not None} == set(Tier)
    assert any(case.tier is None for case in cases)
    assert any(case.tier is not None for case in cases)
    assert any(case.expected_out_of_scope for case in cases)


def test_load_dataset_from_custom_path(tmp_path: Path) -> None:
    """
    GIVEN a custom JSON file with a valid GoldenCase entry
    WHEN ``load_dataset(path)`` is called
    THEN it returns the parsed ``GoldenCase`` list.

    :param tmp_path: Temporary directory fixture provided by pytest.
    """
    payload = [
        {
            "id": "custom-001",
            "description": "Build a lightweight internal inventory dashboard with role-based login.",
            "project_type": "internal_tool",
            "expected_cost_range_eur": [5000, 15000],
        }
    ]
    dataset_file = tmp_path / "custom_dataset.json"
    dataset_file.write_text(json.dumps(payload), encoding="utf-8")

    cases = load_dataset(dataset_file)

    assert len(cases) == 1
    assert isinstance(cases[0], GoldenCase)
    assert cases[0].id == "custom-001"
    assert cases[0].expected_cost_range_eur == (5000, 15000)


@pytest.mark.parametrize(
    ("overrides", "error_keyword"),
    [
        ({"project_type": "blockchain_ai"}, r"project_type"),
        ({"tier": "director"}, r"tier"),
        ({"expected_hours_range": [500, 100]}, r"expected_hours_range"),
        ({"expected_cost_range_eur": [-10, 1000]}, r"expected_cost_range_eur"),
        ({"expected_duration_weeks_range": [12, 4]}, r"expected_duration_weeks_range"),
        ({"expected_phase_count_range": [0, 4]}, r"expected_phase_count_range"),
    ],
    ids=[
        "invalid_project_type",
        "invalid_tier",
        "inverted_hours_range",
        "negative_cost_range",
        "inverted_duration_range",
        "zero_phase_count_range",
    ],
)
def test_golden_case_rejects_invalid_inputs(overrides: dict[str, Any], error_keyword: str) -> None:
    """
    GIVEN a ``GoldenCase`` payload with an invalid field or range
    WHEN ``GoldenCase.model_validate`` is called
    THEN a ``ValidationError`` matching the offending field is raised.

    :param overrides: Field overrides applied to the valid base payload.
    :param error_keyword: Regex keyword expected in the raised ``ValidationError``.
    """
    base: dict[str, Any] = {
        "id": "invalid-case",
        "description": "Valid project description with at least twenty characters.",
        "project_type": "web_saas",
    }
    base.update(overrides)

    with pytest.raises(ValidationError, match=error_keyword):
        GoldenCase.model_validate(base)
