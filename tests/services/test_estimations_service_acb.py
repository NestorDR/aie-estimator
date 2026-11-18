# tests/services/test_estimations_service_acb.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.acb import CriticFeedback, CriticIssue
from aie_estimator.schemas.estimations import (
    DetailLevel,
    OutputFormat,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
)
from aie_estimator.services.boss_service import Boss
from aie_estimator.services.critic_service import Critic
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.sessions.metadata_extractor import MetadataExtractionOutcome
from aie_estimator.sessions.models import ProjectMetadata, Session
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.utils.constants import DEFAULT_FAST_PROFILE


def _pm_estimate(summary: str = "PM estimate for B2B platform.", confidence_pct: int = 84) -> PmEstimation:
    """Build a minimal valid PmEstimation for ACB service tests."""
    return PmEstimation(
        summary=summary,
        confidence_pct=confidence_pct,
        phases=[
            Phase(
                name="Core Build",
                duration_weeks=4,
                cost_eur=16_000,
                summary="Implement backend API and data layer.",
            )
        ],
        team_composition=TeamComposition(developers=2, designers=1),
        total_duration_weeks=4,
        total_cost_eur=16_000,
    )


def _llm_payload(result: Any, input_tokens: int, output_tokens: int, cost_usd: float) -> dict[str, Any]:
    """Build a normalized provider response dictionary."""
    return {
        "result": result,
        "model": DEFAULT_FAST_PROFILE,
        "provider": "litellm",
        "cost_usd": cost_usd,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


@pytest.mark.anyio
async def test_generate_session_estimation_acb_iterates_and_aggregates_telemetry() -> None:
    """
    GIVEN an Actor draft with a major defect on round 0 and a clean revision on round 1
    WHEN EstimationService.generate_session_estimation_acb executes
    THEN Boss iterates once, aggregates Actor + Critic + metadata tokens/costs, and records only the final turn.
    """
    session = Session(session_id="test-session-acb-1")
    draft_v1 = _pm_estimate("Draft v1 missing QA.", confidence_pct=72)
    draft_v2 = _pm_estimate("Draft v2 with QA included.", confidence_pct=88)
    feedback_v1 = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="missing_component",
                severity="major",
                field_path="phases",
                description="QA phase is missing.",
                suggested_fix="Include QA effort in the build phase.",
            )
        ],
        confidence_in_review=85,
    )
    feedback_v2 = CriticFeedback(verdict="accept", issues=[], confidence_in_review=92)

    actor_calls = AsyncMock(
        side_effect=[
            _llm_payload(draft_v1, input_tokens=300, output_tokens=150, cost_usd=0.0010),
            _llm_payload(draft_v2, input_tokens=350, output_tokens=160, cost_usd=0.0012),
        ]
    )
    critic_calls = AsyncMock(
        side_effect=[
            _llm_payload(feedback_v1, input_tokens=200, output_tokens=80, cost_usd=0.0005),
            _llm_payload(feedback_v2, input_tokens=210, output_tokens=50, cost_usd=0.0004),
        ]
    )
    metadata_mock = AsyncMock(
        return_value=MetadataExtractionOutcome(
            metadata=ProjectMetadata(project_name="B2B Platform"),
            input_tokens=90,
            output_tokens=30,
            cost_usd=0.0002,
        )
    )

    service = EstimationService()
    with (
        patch("aie_estimator.services.estimations_service.call_llm", new=actor_calls),
        patch("aie_estimator.services.critic_service.call_llm", new=critic_calls),
        patch("aie_estimator.services.estimations_service.update_metadata", new=metadata_mock),
    ):
        response = await service.generate_session_estimation_acb(
            session=session,
            transcript="Estimate a multi-tenant B2B platform with QA.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            tier_override=Tier.PM,
            critic=Critic(model=DEFAULT_FAST_PROFILE),
            boss=Boss(max_iterations=2),
        )

    assert response["result"] == draft_v2
    assert response["boss_trace"].final_decision == "accept"
    assert response["boss_trace"].iterations_run == 2
    assert response["usage"]["input_tokens"] == 300 + 350 + 200 + 210
    assert response["usage"]["output_tokens"] == 150 + 160 + 80 + 50
    assert response["usage"]["total_tokens"] == 1500
    assert response["usage"]["preprocessing_input_tokens"] == 90
    assert response["usage"]["preprocessing_output_tokens"] == 30
    assert response["cost_usd"] == pytest.approx(0.0033)
    assert len(session.history.messages) == 2
    assert "Draft v2 with QA included." in session.history.messages[1].content
    # Second Actor call must receive <critic_feedback> from round 0
    second_actor_messages = actor_calls.await_args_list[1].kwargs["messages"]
    assert "<critic_feedback>" in second_actor_messages[-1]["content"]

    obs = session.last_turn_observed
    assert obs is not None
    assert session.turn_index == 1
    assert obs.turn_index == 1
    assert obs.tokens_in == 300 + 350 + 200 + 210 + 90
    assert obs.tokens_out == 150 + 160 + 80 + 50 + 30
    assert obs.cost_usd == pytest.approx(0.0033)
    assert obs.cache_hit_kind == "none"
    assert obs.last_resolved_tier == "pm"
