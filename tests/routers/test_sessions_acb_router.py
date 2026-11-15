# tests/routers/test_sessions_acb_router.py

# --- Python Standard Library ---
from typing import Any, Final
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
from fastapi.testclient import TestClient

# --- App modules ---
from aie_estimator.schemas.acb import CriticFeedback
from aie_estimator.schemas.estimations import Phase, PmEstimation, TeamComposition
from aie_estimator.sessions.metadata_extractor import MetadataExtractionOutcome
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE

_VALID_FORM_DATA: Final[dict[str, str]] = {
    "transcript": "We are building MyProject, a multi-tenant B2B analytics platform with FastAPI and PostgreSQL.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def test_estimate_in_session_acb_returns_200_with_boss_trace(client: TestClient) -> None:
    """
    GIVEN an active session
    WHEN POST /api/v1/sessions/{session_id}/estimate-acb is called
    THEN HTTP 200 is returned with ACBResponse containing result and boss_trace.
    """
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    pm_result = PmEstimation(
        summary="PM estimate for MyProject.",
        confidence_pct=86,
        phases=[
            Phase(
                name="Backend & API",
                duration_weeks=4,
                cost_eur=14_000,
                summary="Build FastAPI services and PostgreSQL schema.",
            )
        ],
        team_composition=TeamComposition(developers=2, designers=1),
        total_duration_weeks=4,
        total_cost_eur=14_000,
    )
    critic_feedback = CriticFeedback(verdict="accept", issues=[], confidence_in_review=91)

    actor_mock = AsyncMock(
        return_value={
            "result": pm_result,
            "model": DEFAULT_FAST_PROFILE,
            "provider": "litellm",
            "cost_usd": 0.0010,
            "usage": {"input_tokens": 250, "output_tokens": 120, "total_tokens": 370},
        }
    )
    critic_mock = AsyncMock(
        return_value={
            "result": critic_feedback,
            "model": DEFAULT_FAST_PROFILE,
            "provider": "litellm",
            "cost_usd": 0.0005,
            "usage": {"input_tokens": 180, "output_tokens": 60, "total_tokens": 240},
        }
    )
    metadata_mock = AsyncMock(
        return_value=MetadataExtractionOutcome(
            metadata=ProjectMetadata(project_name="MyProject"),
            input_tokens=40,
            output_tokens=20,
            cost_usd=0.0001,
        )
    )

    with (
        patch("aie_estimator.services.estimations_service.call_llm", new=actor_mock),
        patch("aie_estimator.services.critic_service.call_llm", new=critic_mock),
        patch("aie_estimator.services.estimations_service.update_metadata", new=metadata_mock),
    ):
        response = client.post(
            f"/api/v1/sessions/{session_id}/estimate-acb",
            data=_VALID_FORM_DATA,
            headers={"X-User-Tier": "pm"},
        )

    assert response.status_code == 200
    payload: dict[str, Any] = response.json()
    assert payload["tier"] == "pm"
    assert payload["boss_trace"]["final_decision"] == "accept"
    assert payload["boss_trace"]["iterations_run"] == 1
    assert payload["usage"]["total_tokens"] == 610


def test_estimate_in_session_acb_returns_404_for_unknown_session(client: TestClient) -> None:
    """
    GIVEN a nonexistent session_id
    WHEN POST /api/v1/sessions/{session_id}/estimate-acb is called
    THEN HTTP 404 Not Found is returned.
    """
    response = client.post(
        "/api/v1/sessions/00000000-0000-0000-0000-000000000000/estimate-acb",
        data=_VALID_FORM_DATA,
    )

    assert response.status_code == 404
