# tests/services/test_critic_boss_services.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.acb import CriticFeedback, CriticIssue
from aie_estimator.schemas.estimations import (
    LOW_CONFIDENCE_THRESHOLD,
    ComponentEstimation,
    DeveloperEstimation,
    HoursRange,
)
from aie_estimator.services.boss_service import Boss
from aie_estimator.services.critic_service import Critic
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE


def _sample_developer_estimate(confidence_pct: int = 80) -> DeveloperEstimation:
    """Build a minimal valid DeveloperEstimation for Critic and Boss tests."""
    return DeveloperEstimation(
        summary="Initial developer estimate covering API and database.",
        confidence_pct=confidence_pct,
        components=[ComponentEstimation(name="Backend API", estimated_hours=60)],
        total_hours_range=HoursRange(min_hours=50, max_hours=75),
    )


def _blocking_feedback(verdict: str = "needs_iteration") -> CriticFeedback:
    """Build a CriticFeedback with one major defect."""
    return CriticFeedback(
        verdict=verdict,  # type: ignore[arg-type]
        issues=[
            CriticIssue(
                category="missing_component",
                severity="major",
                field_path="components",
                description="Authentication module is missing from components.",
                suggested_fix="Add an Auth component with estimated hours.",
            )
        ],
        confidence_in_review=85,
    )


@pytest.mark.anyio
async def test_critic_review_returns_structured_feedback_and_telemetry() -> None:
    """
    GIVEN call_llm returning a valid CriticFeedback and usage metadata
    WHEN Critic.review is awaited
    THEN it returns CriticOutcome with parsed feedback, tokens, and cost_usd.
    """
    expected = _blocking_feedback()
    mock_call = AsyncMock(
        return_value={
            "result": expected,
            "model": DEFAULT_FAST_PROFILE,
            "provider": "litellm",
            "cost_usd": 0.0012,
            "usage": {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
        }
    )

    with patch("aie_estimator.services.critic_service.call_llm", new=mock_call):
        critic = Critic(model=DEFAULT_FAST_PROFILE)
        outcome = await critic.review(
            transcript="Build a secure SaaS API with OAuth2.",
            metadata=ProjectMetadata(project_name="SaaS API"),
            tier=Tier.DEVELOPER,
            result=_sample_developer_estimate(),
        )

    assert outcome.feedback.verdict == "needs_iteration"
    assert outcome.input_tokens == 120
    assert outcome.output_tokens == 60
    assert outcome.cost_usd == pytest.approx(0.0012)


@pytest.mark.anyio
async def test_critic_review_degrades_gracefully_on_provider_error() -> None:
    """
    GIVEN call_llm raising a RuntimeError
    WHEN Critic.review is awaited
    THEN it degrades gracefully to accept with confidence_in_review=0.
    """
    with patch(
        "aie_estimator.services.critic_service.call_llm",
        new=AsyncMock(side_effect=RuntimeError("Upstream timeout")),
    ):
        critic = Critic(model=DEFAULT_FAST_PROFILE)
        outcome = await critic.review(
            transcript="Build a secure SaaS API with OAuth2.",
            metadata=ProjectMetadata(),
            tier=Tier.DEVELOPER,
            result=_sample_developer_estimate(),
        )

    assert outcome.feedback.verdict == "accept"
    assert outcome.feedback.issues == []
    assert outcome.feedback.confidence_in_review == 0


@pytest.mark.anyio
async def test_boss_accept_on_first_pass() -> None:
    """
    GIVEN Critic accepting on iteration 0
    WHEN Boss.run executes
    THEN it returns the Actor draft unmodified with final_decision='accept' and iterations_run=1.
    """
    draft = _sample_developer_estimate()
    actor = AsyncMock(return_value=draft)
    critic = AsyncMock(return_value=CriticFeedback(verdict="accept", issues=[], confidence_in_review=92))

    result, trace = await Boss(max_iterations=2).run(actor=actor, critic=critic)

    assert result == draft
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 1
    assert trace.iterations[0].decision_after == "accept"
    assert actor.await_count == 1
    assert critic.await_count == 1


@pytest.mark.anyio
async def test_boss_iterates_with_feedback_then_accepts() -> None:
    """
    GIVEN Critic returning needs_iteration on round 0 and accept on round 1
    WHEN Boss.run executes with max_iterations=2
    THEN Boss passes round 0 feedback to Actor on round 1 and accepts with iterations_run=2.
    """
    draft_v1 = _sample_developer_estimate(confidence_pct=70)
    draft_v2 = _sample_developer_estimate(confidence_pct=88)
    feedback_v1 = _blocking_feedback("needs_iteration")

    actor = AsyncMock(side_effect=[draft_v1, draft_v2])
    critic = AsyncMock(
        side_effect=[
            feedback_v1,
            CriticFeedback(verdict="accept", issues=[], confidence_in_review=90),
        ]
    )

    result, trace = await Boss(max_iterations=2).run(actor=actor, critic=critic)

    assert result == draft_v2
    assert trace.final_decision == "accept"
    assert trace.iterations_run == 2
    assert [it.decision_after for it in trace.iterations] == ["iterate", "accept"]
    assert actor.await_args_list[0].args[0] is None
    assert actor.await_args_list[1].args[0] == feedback_v1


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("verdicts", "expected_runs"),
    [
        (["reject"], 1),
        (["needs_iteration", "needs_iteration"], 2),
    ],
)
async def test_boss_synthesizes_fallback_on_reject_or_budget_exhaustion(
    verdicts: list[str],
    expected_runs: int,
) -> None:
    """
    GIVEN Critic rejecting immediately or exhausting max_iterations=2
    WHEN Boss.run executes
    THEN Boss returns final_decision='synthesize', prepends caveats, and floors confidence at 30.
    """
    draft = _sample_developer_estimate(confidence_pct=40)
    actor = AsyncMock(return_value=draft)
    critic = AsyncMock(side_effect=[_blocking_feedback(v) for v in verdicts])

    result, trace = await Boss(max_iterations=2).run(actor=actor, critic=critic)

    assert trace.final_decision == "synthesize"
    assert trace.iterations_run == expected_runs
    assert "Open caveats from independent review" in result.summary
    assert "missing_component" in result.summary
    assert result.confidence_pct == LOW_CONFIDENCE_THRESHOLD
    assert result.confidence_pct >= LOW_CONFIDENCE_THRESHOLD
