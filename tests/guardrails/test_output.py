# tests/guardrails/test_output.py

# --- App modules ---
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.schemas.estimations import EstimationResult, Phase


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
    result = EstimationResult(
        summary="Standard web application estimation.",
        confidence_pct=85,
        phases=[phase],
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
    result = EstimationResult(
        summary="Out of scope: project is unrelated to software engineering.",
        confidence_pct=15,
        phases=[phase],
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
    # Create an initial valid result at boundary
    initial_phase = Phase(
        name="Discovery",
        duration_weeks=2,
        cost_eur=5000,
        summary="Exploratory discovery phase for ambiguous prompt.",
    )
    # Bypass model_validator on instantiation by using construct or prefixing, or creating with confidence 20
    # Note: EstimationResult validator requires 'Out of scope:' when confidence < 30 on instantiation.
    # We construct an instance directly to simulate edge-case outputs or deserialized raw results.
    unmarked_result = EstimationResult.model_construct(
        summary="The prompt lacks key specifications and team sizing details.",
        confidence_pct=20,
        phases=[initial_phase],
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
