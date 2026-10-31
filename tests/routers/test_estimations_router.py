# tests/routers/test_estimations_router.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.schemas.models import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM


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
    mock_generate = AsyncMock(
        return_value={
            "estimation": "Project Breakdown:\n- Backend: 40h\nTotal: 40h",
            "model": DEFAULT_GPT_LLM,
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
        "model": DEFAULT_GPT_LLM,
        "evaluate": False,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == DEFAULT_GPT_LLM
    assert data["provider"] == "openai"
    assert mock_generate.call_count == 1


@pytest.mark.parametrize(
    ("cost_input", "expected_cost"),
    [
        (0.00125, 0.00125),
        (None, None),
    ],
    ids=["calculated_cost_present", "cost_null_default"],
)
def test_create_estimation_cost_usd_serialization(
    client: TestClient,
    sample_transcription: str,
    cost_input: float | None,
    expected_cost: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an LLM response containing calculated cost_usd or None
    WHEN a POST request is processed by /api/v1/estimate
    THEN the response serializes cost_usd accurately in JSON.

    :param client: Shared TestClient fixture.
    :param sample_transcription: Valid meeting transcript fixture.
    :param cost_input: Cost value from LLM service response.
    :param expected_cost: Expected serialized cost value.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value={
            "estimation": "Breakdown:\n- API: 10h\nTotal: 10h",
            "model": DEFAULT_FAST_PROFILE,
            "provider": "litellm",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
            },
            "preprocessing": "none",
            "extracted_requirements": None,
            "latency_ms": 180,
            "cost_usd": cost_input,
        }
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.llm_service.generate_estimation", mock_generate)

    payload: dict[str, Any] = {
        "transcription": sample_transcription,
        "model": DEFAULT_FAST_PROFILE,
        "engine": "litellm",
        "evaluate": False,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["cost_usd"] == expected_cost
    assert data["model"] == DEFAULT_FAST_PROFILE


def test_create_estimation_native_engine_dispatch(
    client: TestClient,
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an estimation request specifying engine='native'
    WHEN a POST request is processed by /api/v1/estimate
    THEN the request executes through the native engine successfully.

    :param client: Shared TestClient fixture.
    :param sample_transcription: Valid meeting transcript fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value={
            "estimation": "Native Breakdown:\n- Service: 15h\nTotal: 15h",
            "model": DEFAULT_GPT_LLM,
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            "preprocessing": "none",
            "extracted_requirements": None,
            "latency_ms": 150,
            "cost_usd": None,
        }
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.llm_service.generate_estimation", mock_generate)

    payload: dict[str, Any] = {
        "transcription": sample_transcription,
        "model": DEFAULT_GPT_LLM,
        "engine": "native",
        "evaluate": False,
    }
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == DEFAULT_GPT_LLM
    assert data["provider"] == "openai"
    assert mock_generate.call_count == 1
    assert mock_generate.call_args.args[1].engine == "native"

