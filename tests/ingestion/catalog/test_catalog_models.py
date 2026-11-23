# tests/ingestion/catalog/test_catalog_models.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import (
    CatalogDecision,
    CatalogSource,
    DataCatalog,
    PiiFlag,
    QualityScore,
    Sensitivity,
)


def _make_source(**overrides: Any) -> dict[str, Any]:
    """Build a valid raw source payload with optional field overrides."""
    base: dict[str, Any] = {
        "name": "historical_budgets",
        "location": "budgets",
        "format": "json",
        "quality": {"completeness": 4, "consistency": 3, "actuality": 5, "reliability": 4},
        "sensitivity": {"has_pii": True, "pii_flags": ["EMAIL", "PERSON"], "access_level": "confidential"},
        "decision": "include",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("scores", "expected_ready"),
    [
        ((3, 3, 3, 3), True),
        ((5, 5, 5, 5), True),
        ((5, 5, 2, 5), False),
        ((1, 4, 4, 4), False),
    ],
    ids=["all_three", "all_five", "actuality_two", "completeness_one"],
)
def test_quality_score_is_rag_ready(scores: tuple[int, int, int, int], expected_ready: bool) -> None:
    """
    GIVEN four quality dimension scores
    WHEN is_rag_ready is evaluated
    THEN it returns True only when every dimension is at least 3.
    """
    quality = QualityScore(
        completeness=scores[0],
        consistency=scores[1],
        actuality=scores[2],
        reliability=scores[3],
    )
    assert quality.is_rag_ready is expected_ready


@pytest.mark.parametrize(
    ("overrides", "match_pattern"),
    [
        ({"name": "InvalidName"}, r"snake_case"),
        ({"name": ""}, r"snake_case"),
        ({"decision": "review", "decision_reason": ""}, r"requires a non-empty decision_reason"),
        ({"decision": "exclude", "decision_reason": "   "}, r"requires a non-empty decision_reason"),
        ({"unexpected_field": "nope"}, r"Extra inputs are not permitted"),
        (
            {"sensitivity": {"has_pii": True, "pii_flags": ["UNKNOWN_FLAG"], "access_level": "internal"}},
            r"Input should be",
        ),
        (
            {"sensitivity": {"has_pii": False, "pii_flags": [], "access_level": "secret"}},
            r"Input should be",
        ),
        ({"quality": {"completeness": 6, "consistency": 3, "actuality": 3, "reliability": 3}}, r"less than or equal to 5"),
    ],
    ids=[
        "uppercase_name",
        "empty_name",
        "review_missing_reason",
        "exclude_whitespace_reason",
        "extra_field_forbidden",
        "invalid_pii_flag",
        "invalid_access_level",
        "score_above_five",
    ],
)
def test_catalog_source_validation_errors(overrides: dict[str, Any], match_pattern: str) -> None:
    """
    GIVEN an invalid CatalogSource payload
    WHEN CatalogSource.model_validate is called
    THEN ValidationError is raised matching the expected constraint.
    """
    with pytest.raises(ValidationError, match=match_pattern):
        CatalogSource.model_validate(_make_source(**overrides))


def test_data_catalog_rejects_duplicate_source_names() -> None:
    """
    GIVEN two sources sharing the same name
    WHEN DataCatalog.model_validate is called
    THEN ValidationError is raised for duplicate source names.
    """
    payload = {
        "version": "1.0.0",
        "sources": [_make_source(name="budgets_json"), _make_source(name="budgets_json")],
    }
    with pytest.raises(ValidationError, match=r"Duplicate source name"):
        DataCatalog.model_validate(payload)


def test_data_catalog_included_sources_and_find() -> None:
    """
    GIVEN a catalog with include, review, and exclude sources
    WHEN included_sources and find are called
    THEN only include sources are returned and find resolves by name or returns None.
    """
    catalog = DataCatalog.model_validate(
        {
            "version": "1.0.0",
            "sources": [
                _make_source(name="budgets_json", decision="include"),
                _make_source(name="transcripts_txt", decision="review", decision_reason="Needs sign-off"),
                _make_source(name="rate_card_xlsx", decision="exclude", decision_reason="Obsolete 2024 rates"),
            ],
        }
    )

    included = catalog.included_sources()
    assert [source.name for source in included] == ["budgets_json"]
    assert catalog.find("transcripts_txt") is not None
    assert catalog.find("transcripts_txt").decision is CatalogDecision.REVIEW
    assert catalog.find("missing") is None
    assert Sensitivity(has_pii=True, pii_flags=(PiiFlag.EMAIL,)).pii_flags == (PiiFlag.EMAIL,)
