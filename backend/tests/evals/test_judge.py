# tests/evals/test_judge.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.domain.schemas.estimations import (
    EstimationResponse,
    Milestone,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
    TokenUsage,
)
from aie_estimator.generation.conversation.tier_resolver import Tier
from evals.dataset import GoldenCase
from evals.judge import JudgeVerdict, evaluate_with_judge, render_judge_prompt


def _make_sample_case_and_response() -> tuple[GoldenCase, EstimationResponse]:
    """Build a minimal ``GoldenCase`` and ``EstimationResponse`` pair for judge tests."""
    case = GoldenCase(
        id="judge-001",
        description="Build an internal HR onboarding tool with SSO login and Slack alerts.",
        project_type=ProjectType.INTERNAL_TOOL,
        expected_tier=Tier.PM,
    )
    pm_result = PmEstimation(
        summary="Phased HR onboarding portal with SSO and Slack integration.",
        confidence_pct=82,
        phases=[Phase(name="Core", duration_weeks=4, cost_eur=16000, summary="SSO and Slack workflows.")],
        milestones=[Milestone(name="Launch", target_week=4, deliverable="Live HR portal")],
        team_composition=TeamComposition(developers=2, designers=1, qa_engineers=0, tech_lead=True),
        blockers=[],
        total_duration_weeks=4,
        total_cost_eur=16000,
    )
    response = EstimationResponse(
        result=pm_result,
        tier=Tier.PM,
        tier_rule="explicit_override",
        prompt_version="v3",
        cached=False,
        model="reasoner-fast",
        provider="openai",
        cost_usd=0.001,
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    return case, response


def test_render_judge_prompt_includes_tier_and_description() -> None:
    """
    GIVEN a ``GoldenCase`` and ``EstimationResponse``
    WHEN ``render_judge_prompt`` renders ``judge.j2`` under ``StrictUndefined``
    THEN the output contains the resolved tier, project description, and serialized JSON.
    """
    case, response = _make_sample_case_and_response()

    rendered = render_judge_prompt(case, response)

    assert "pm stakeholder" in rendered
    assert case.description in rendered
    assert "Phased HR onboarding portal" in rendered


@pytest.mark.parametrize(
    ("ratings", "expected_score", "expected_passed"),
    [
        ((5, 5, 5), 1.0, True),
        ((4, 4, 4), 0.75, True),
        ((3, 3, 3), 0.5, False),
        ((1, 1, 1), 0.0, False),
    ],
    ids=["all_5_passes", "all_4_passes", "all_3_fails_threshold", "all_1_zero"],
)
@pytest.mark.anyio
async def test_evaluate_with_judge_normalizes_likert_scores(
    monkeypatch: pytest.MonkeyPatch,
    ratings: tuple[int, int, int],
    expected_score: float,
    expected_passed: bool,
) -> None:
    """
    GIVEN a mocked ``call_llm`` returning a ``JudgeVerdict`` with 1-5 Likert scores
    WHEN ``evaluate_with_judge`` is awaited
    THEN it normalizes the scores to ``[0.0, 1.0]`` and applies the 0.7 pass threshold.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param ratings: Tuple of ``(coherence, actionability, tier_appropriateness)`` ratings.
    :param expected_score: Expected normalized score.
    :param expected_passed: Expected pass/fail boolean.
    """
    case, response = _make_sample_case_and_response()
    verdict = JudgeVerdict(
        coherence=ratings[0],
        actionability=ratings[1],
        tier_appropriateness=ratings[2],
        reasoning="Structured phases and team staffing match the PM tier.",
    )
    mock_call_llm = AsyncMock(return_value={"result": verdict})
    monkeypatch.setattr("evals.judge.call_llm", mock_call_llm)

    metric = await evaluate_with_judge(case, response)

    assert metric.name == "llm_judge"
    assert metric.score == pytest.approx(expected_score)
    assert metric.passed is expected_passed
    assert "Structured phases" in metric.details
    mock_call_llm.assert_awaited_once()


@pytest.mark.anyio
async def test_evaluate_with_judge_handles_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a mocked ``call_llm`` that raises an exception
    WHEN ``evaluate_with_judge`` is awaited
    THEN it returns a failed ``MetricResult`` with ``score=0.0`` without raising.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    case, response = _make_sample_case_and_response()
    monkeypatch.setattr("evals.judge.call_llm", AsyncMock(side_effect=RuntimeError("upstream timeout")))

    metric = await evaluate_with_judge(case, response)

    assert metric.name == "llm_judge"
    assert metric.score == 0.0
    assert metric.passed is False
    assert "upstream timeout" in metric.details
