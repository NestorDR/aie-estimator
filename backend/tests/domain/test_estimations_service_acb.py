# tests/domain/test_estimations_service_acb.py

# --- Python Standard Library ---
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.domain.schemas.acb import CriticFeedback, CriticIssue
from aie_estimator.domain.schemas.estimations import (
    DetailLevel,
    OutputFormat,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
)
from aie_estimator.generation.agentic.boss_service import Boss
from aie_estimator.generation.agentic.critic_service import Critic
from aie_estimator.domain.estimations_service import EstimationService
from aie_estimator.generation.conversation.metadata_extractor import MetadataExtractionOutcome
from aie_estimator.generation.conversation.models import ProjectMetadata, Session
from aie_estimator.generation.conversation.tier_resolver import Tier
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


def _llm_payload(result: Any, input_tokens: int, output_tokens: int, cost_usd: float | None) -> dict[str, Any]:
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
        patch("aie_estimator.domain.estimations_service.call_llm", new=actor_calls),
        patch("aie_estimator.generation.agentic.critic_service.call_llm", new=critic_calls),
        patch("aie_estimator.domain.estimations_service.update_metadata", new=metadata_mock),
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
    assert "preprocessing_input_tokens" not in response["usage"]
    assert "preprocessing_output_tokens" not in response["usage"]
    assert response["cost_usd"] == pytest.approx(0.0031)
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


@pytest.mark.parametrize(
    ("actor_costs", "critic_payload", "expected_response_cost", "expected_obs_cost"),
    [
        (
            [0.0010, None],
            _llm_payload(
                CriticFeedback(
                    verdict="needs_iteration",
                    issues=[
                        CriticIssue(
                            category="missing_component",
                            severity="major",
                            field_path="phases",
                            description="Missing QA.",
                            suggested_fix="Add QA.",
                        )
                    ],
                    confidence_in_review=80,
                ),
                input_tokens=150,
                output_tokens=60,
                cost_usd=0.0005,
            ),
            None,
            None,
        ),
        (
            [0.0010],
            _llm_payload(
                CriticFeedback(verdict="accept", issues=[], confidence_in_review=90),
                input_tokens=150,
                output_tokens=60,
                cost_usd=None,
            ),
            0.0010,
            0.0012,
        ),
        (
            [0.0010],
            RuntimeError("Critic unavailable"),
            0.0010,
            0.0012,
        ),
    ],
)
@pytest.mark.anyio
async def test_generate_session_estimation_acb_cost_preservation(
    actor_costs: list[float | None],
    critic_payload: dict[str, Any] | Exception,
    expected_response_cost: float | None,
    expected_obs_cost: float | None,
) -> None:
    """
    GIVEN mixed priced/unpriced Actor and Critic outcomes
    WHEN EstimationService.generate_session_estimation_acb executes
    THEN unpriced Actor calls preserve cost_usd=None while priced Actor turns sum non-null Critic and auxiliary costs.

    :param actor_costs: Per-iteration Actor USD cost sequence.
    :param critic_payload: Critic call return payload or exception triggering fail-open fallback.
    :param expected_response_cost: Expected generation cost on the returned ACB dictionary.
    :param expected_obs_cost: Expected full-turn cost on TurnObservation.
    """
    session = Session(session_id="test-session-acb-cost")
    draft = _pm_estimate()
    actor_calls = AsyncMock(
        side_effect=[_llm_payload(draft, input_tokens=200, output_tokens=100, cost_usd=cost) for cost in actor_costs]
    )
    critic_effects: list[Any] = [critic_payload]
    if len(actor_costs) > 1:
        critic_effects.append(
            _llm_payload(
                CriticFeedback(verdict="accept", issues=[], confidence_in_review=90),
                input_tokens=150,
                output_tokens=50,
                cost_usd=0.0004,
            )
        )
    critic_calls = AsyncMock(side_effect=critic_effects)
    metadata_mock = AsyncMock(
        return_value=MetadataExtractionOutcome(
            metadata=ProjectMetadata(project_name="B2B Platform"),
            input_tokens=50,
            output_tokens=20,
            cost_usd=0.0002,
        )
    )

    service = EstimationService()
    with (
        patch("aie_estimator.domain.estimations_service.call_llm", new=actor_calls),
        patch("aie_estimator.generation.agentic.critic_service.call_llm", new=critic_calls),
        patch("aie_estimator.domain.estimations_service.update_metadata", new=metadata_mock),
    ):
        response = await service.generate_session_estimation_acb(
            session=session,
            transcript="Estimate a multi-tenant B2B platform with QA.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            tier_override=Tier.PM,
            critic=Critic(model=DEFAULT_FAST_PROFILE),
            boss=Boss(max_iterations=len(actor_costs)),
        )

    assert response["cost_usd"] == (
        pytest.approx(expected_response_cost) if expected_response_cost is not None else None
    )
    assert session.last_turn_observed is not None
    assert session.last_turn_observed.cost_usd == (
        pytest.approx(expected_obs_cost) if expected_obs_cost is not None else None
    )


@pytest.mark.anyio
async def test_generate_session_estimation_acb_logs_per_dispatch_latency_and_trimmed_completion() -> None:
    """
    GIVEN a two-iteration ACB turn
    WHEN EstimationService.generate_session_estimation_acb completes
    THEN estimation_generation_completed logs per-dispatch latency and estimation_acb_completed omits duplicate telemetry.
    """
    session = Session(session_id="test-session-acb-logs")
    draft_v1 = _pm_estimate("Draft v1.", confidence_pct=72)
    draft_v2 = _pm_estimate("Draft v2.", confidence_pct=88)
    feedback_v1 = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="missing_component",
                severity="major",
                field_path="phases",
                description="Missing QA.",
                suggested_fix="Add QA.",
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
    metadata_mock = AsyncMock(return_value=MetadataExtractionOutcome(metadata=ProjectMetadata()))
    log_spy = MagicMock()
    fake_time = SimpleNamespace(perf_counter=MagicMock(side_effect=[10.0, 11.0, 11.125, 15.0, 15.25, 16.5]))

    service = EstimationService()
    with (
        patch("aie_estimator.domain.estimations_service.call_llm", new=actor_calls),
        patch("aie_estimator.generation.agentic.critic_service.call_llm", new=critic_calls),
        patch("aie_estimator.domain.estimations_service.update_metadata", new=metadata_mock),
        patch("aie_estimator.domain.estimations_service.log.info", new=log_spy),
        patch("aie_estimator.domain.estimations_service.time", new=fake_time),
    ):
        await service.generate_session_estimation_acb(
            session=session,
            transcript="Estimate a multi-tenant B2B platform with QA.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            tier_override=Tier.PM,
            critic=Critic(model=DEFAULT_FAST_PROFILE),
            boss=Boss(max_iterations=2),
        )

    generation_calls = [
        call for call in log_spy.call_args_list if call.args == ("estimation_generation_completed",)
    ]
    assert [call.kwargs["latency_ms"] for call in generation_calls] == [125, 250]

    assert session.last_turn_observed is not None
    assert session.last_turn_observed.latency_ms == 6500

    acb_completed_calls = [
        call for call in log_spy.call_args_list if call.args == ("estimation_acb_completed",)
    ]
    assert len(acb_completed_calls) == 1
    assert acb_completed_calls[0].kwargs == {
        "session_id": session.session_id,
        "tier": "pm",
        "final_decision": "accept",
        "iterations_run": 2,
    }
