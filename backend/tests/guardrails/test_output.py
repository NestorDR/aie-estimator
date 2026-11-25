# tests/guardrails/test_output.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.schemas.estimations import (
    ComponentEstimation,
    CostRange,
    DeveloperEstimation,
    DurationRange,
    ExecutiveEstimation,
    HoursRange,
    Phase,
    PmEstimation,
    TeamComposition,
)


def test_enforce_scope_response_high_confidence_unchanged() -> None:
    """
    GIVEN an estimation result with confidence >= 30%
    WHEN enforce_scope_response is executed
    THEN the result is returned unchanged.
    """
    phase = Phase(
        name="Implementation",
        duration_weeks=4,
        cost_eur=10000,
        summary="Implementation of backend API and frontend views.",
    )
    result = PmEstimation(
        summary="Standard web application estimation.",
        confidence_pct=85,
        phases=[phase],
        team_composition=TeamComposition(developers=2, designers=1),
        total_duration_weeks=4,
        total_cost_eur=10000,
    )

    filtered = enforce_scope_response(result)
    assert filtered is result
    assert filtered.confidence_pct == 85
    assert filtered.total_cost_eur == 10000


def test_enforce_scope_response_already_out_of_scope_unchanged() -> None:
    """
    GIVEN an estimation result with low confidence that already starts with 'Out of scope:'
    WHEN enforce_scope_response is executed
    THEN the result is returned unchanged.
    """
    phase = Phase(
        name="Not estimated",
        duration_weeks=1,
        cost_eur=0,
        summary="Cannot be estimated due to missing requirements.",
    )
    result = PmEstimation(
        summary="Out of scope: project is unrelated to software engineering.",
        confidence_pct=15,
        phases=[phase],
        team_composition=TeamComposition(developers=0, designers=0),
        total_duration_weeks=1,
        total_cost_eur=0,
    )

    filtered = enforce_scope_response(result)
    assert filtered is result
    assert filtered.confidence_pct == 15
    assert filtered.summary.startswith("Out of scope:")


def test_enforce_scope_response_low_confidence_unmarked_normalized() -> None:
    """
    GIVEN an estimation result with low confidence (< 30%) lacking the 'Out of scope:' prefix
    WHEN enforce_scope_response is executed
    THEN the result is normalized with the prefix, duration=1, cost=0, and a placeholder phase.
    """
    initial_phase = Phase(
        name="Discovery",
        duration_weeks=2,
        cost_eur=5000,
        summary="Exploratory discovery phase for ambiguous prompt.",
    )
    unmarked_result = PmEstimation.model_construct(
        summary="The prompt lacks key specifications and team sizing details.",
        confidence_pct=20,
        phases=[initial_phase],
        team_composition=TeamComposition(developers=1, designers=0),
        total_duration_weeks=2,
        total_cost_eur=5000,
    )

    filtered = enforce_scope_response(unmarked_result)

    assert filtered is not unmarked_result
    assert filtered.confidence_pct == 20
    assert filtered.summary.startswith("Out of scope:")
    assert "Original model rationale" in filtered.summary
    assert filtered.total_duration_weeks == 1
    assert filtered.total_cost_eur == 0
    assert len(filtered.phases) == 1
    assert filtered.phases[0].name == "Not estimated"
    assert filtered.phases[0].cost_eur == 0
    assert filtered.phases[0].duration_weeks == 1


@pytest.mark.parametrize(
    ("raw_result", "expected_cls"),
    [
        (
            DeveloperEstimation.model_construct(
                summary="Unclear requirements.",
                confidence_pct=10,
                components=[ComponentEstimation(name="API", estimated_hours=40)],
                total_hours_range=HoursRange(min_hours=30, max_hours=50),
            ),
            DeveloperEstimation,
        ),
        (
            PmEstimation.model_construct(
                summary="Unclear requirements.",
                confidence_pct=15,
                phases=[Phase(name="Build", duration_weeks=2, cost_eur=4000, summary="Build phase work.")],
                team_composition=TeamComposition(developers=1, designers=0),
                total_duration_weeks=2,
                total_cost_eur=4000,
            ),
            PmEstimation,
        ),
        (
            ExecutiveEstimation.model_construct(
                summary="Unclear requirements.",
                confidence_pct=20,
                headline_cost_range=CostRange(min_eur=5000, max_eur=10000),
                headline_duration_range=DurationRange(min_weeks=2, max_weeks=4),
                confidence_level="low",
                top_three_risks=["Vague scope"],
                go_no_go_recommendation="Hold.",
            ),
            ExecutiveEstimation,
        ),
    ],
)
def test_enforce_scope_response_normalizes_all_tier_schemas(
    raw_result: DeveloperEstimation | PmEstimation | ExecutiveEstimation,
    expected_cls: type,
) -> None:
    """
    GIVEN any tier estimate with confidence < 30 and no 'Out of scope:' prefix
    WHEN enforce_scope_response is executed
    THEN it returns a valid normalized instance of the same tier schema.
    """
    filtered = enforce_scope_response(raw_result)

    assert isinstance(filtered, expected_cls)
    assert filtered.summary.startswith("Out of scope:")
    assert filtered.confidence_pct == raw_result.confidence_pct
