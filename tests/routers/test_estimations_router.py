# tests/routers/test_estimations_router.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import MagicMock

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import pytest


@pytest.mark.parametrize(
    "invalid_model",
    [
        "string",
        "invalid-model-name",
        "llama-3-70b",
        "gemini-2.5-flash",
    ],
    ids=[
        "swagger_default_string",
        "arbitrary_unsupported_model",
        "unsupported_llama",
        "unsupported_gemini",
    ],
)
def test_create_estimation_invalid_model_returns_422(
    client: TestClient,
    sample_transcription: str,
    invalid_model: str,
) -> None:
    """
    GIVEN an invalid or placeholder model identifier
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 422 Unprocessable Entity is returned with validation error details.

    :param client: Shared TestClient fixture.
    :param sample_transcription: Valid meeting transcript fixture.
    :param invalid_model: Target invalid model identifier.
    """
    payload: dict[str, Any] = {
        "transcription": sample_transcription,
        "model": invalid_model,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422

    body = response.json()
    errors = body.get("errors", [])
    assert any(err.get("loc") == ["body", "model"] for err in errors)
    assert any("Input should be" in err.get("msg", "") for err in errors)


def test_create_estimation_valid_model_override_accepted(
    client: TestClient,
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a valid model identifier from MODEL_CATALOG
    WHEN a POST request is sent to /api/v1/estimate
    THEN the request passes schema validation and succeeds with HTTP 200.

    :param client: Shared TestClient fixture.
    :param sample_transcription: Valid meeting transcript fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = MagicMock(
        return_value={
            "estimation": "Project Breakdown:\n- Backend: 40h\nTotal: 40h",
            "model": "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            "preprocessing": "none",
            "extracted_requirements": None,
            "latency_ms": 250,
        }
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.llm_service.generate_estimation", mock_generate)

    payload: dict[str, Any] = {
        "transcription": sample_transcription,
        "model": "gpt-4o-mini",
        "evaluate": False,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "gpt-4o-mini"
    assert data["provider"] == "openai"
    assert mock_generate.call_count == 1
