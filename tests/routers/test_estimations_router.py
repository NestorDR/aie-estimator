# tests/routers/test_estimations_router.py

# --- Python Standard Library ---
from collections.abc import Callable
from typing import Any, Final
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.guardrails.input import InputGuardrailViolation
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM

VALID_PAYLOAD: Final[dict[str, Any]] = {
    "description": "This is a comprehensive description of an inventory management SaaS system.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def test_create_estimation_invalid_field_returns_422(client: TestClient) -> None:
    """
    GIVEN an estimation payload with an invalid field value
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 422 Unprocessable Content is returned with field-level validation details.

    Per-field validation cases live in tests/schemas/test_estimations.py.

    :param client: Shared TestClient fixture.
    """
    response = client.post("/api/v1/estimate", json={**VALID_PAYLOAD, "description": "Too short"})
    assert response.status_code == 422

    errors = response.json().get("errors", [])
    assert any(err.get("loc") == ["body", "description"] for err in errors)
    assert any("at least 20 characters" in err.get("msg", "") for err in errors)


def test_create_estimation_valid_request_returns_200(
    client: TestClient,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a valid estimation request
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 200 is returned with result, prompt_version, model, provider, and usage metadata.

    :param client: Shared TestClient fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            model=DEFAULT_GPT_LLM,
            provider="openai",
            cost_usd=0.0025,
            usage={"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    with patch("aie_estimator.routers.estimations_router.render_estimation_prompt") as mock_render:
        mock_render.return_value = ("System instructions", "User project description")

        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["result"]["total_cost_eur"] == 20000
        assert data["result"]["total_duration_weeks"] == 8
        assert data["result"]["confidence_pct"] == 85
        assert len(data["result"]["phases"]) == 2
        assert data["cached"] is False
        assert data["prompt_version"] == "v1"
        assert data["model"] == DEFAULT_GPT_LLM
        assert data["provider"] == "openai"
        assert data["cost_usd"] == 0.0025
        assert data["usage"] == {
            "input_tokens": 120,
            "output_tokens": 60,
            "total_tokens": 180,
            "preprocessing_input_tokens": 0,
            "preprocessing_output_tokens": 0,
        }

        # Verify render_estimation_prompt was called with resolved active version "v1"
        mock_render.assert_called_once()
        _, call_kwargs = mock_render.call_args
        assert call_kwargs.get("version") == "v1"

        # Verify llm_service was called with rendered prompts
        mock_generate.assert_called_once()
        gen_args, gen_kwargs = mock_generate.call_args
        called_system = gen_kwargs.get("system_prompt") or (gen_args[0] if gen_args else None)
        called_user = gen_kwargs.get("user_prompt") or (gen_args[1] if len(gen_args) > 1 else None)
        assert called_system == "System instructions"
        assert called_user == "User project description"


@pytest.mark.parametrize(
    ("payload_override", "result_override", "expected_response", "expected_engine"),
    [
        (
            {"model": DEFAULT_GPT_LLM},
            {"model": DEFAULT_GPT_LLM, "provider": "openai"},
            {"model": DEFAULT_GPT_LLM, "provider": "openai", "cost_usd": None},
            "litellm",
        ),
        (
            {"model": DEFAULT_FAST_PROFILE},
            {"model": DEFAULT_FAST_PROFILE, "provider": "litellm", "cost_usd": 0.00125},
            {"model": DEFAULT_FAST_PROFILE, "provider": "litellm", "cost_usd": 0.00125},
            "litellm",
        ),
        (
            {"engine": "native"},
            {"model": DEFAULT_GPT_LLM, "provider": "openai"},
            {"model": DEFAULT_GPT_LLM, "provider": "openai"},
            "native",
        ),
    ],
    ids=["model_override_null_cost", "profile_override_with_cost", "native_engine"],
)
def test_create_estimation_request_options_pass_through(
    client: TestClient,
    payload_override: dict[str, Any],
    result_override: dict[str, Any],
    expected_response: dict[str, Any],
    expected_engine: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a request overriding the model, profile, or engine
    WHEN a POST request is sent to /api/v1/estimate
    THEN the service receives the engine and model, and the response reflects the model, provider, and cost.

    :param client: Shared TestClient fixture.
    :param payload_override: Request fields added to the valid payload.
    :param result_override: Overrides for the mocked service result.
    :param expected_response: Response fields expected in the JSON body.
    :param expected_engine: Engine expected in the options passed to the service.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(return_value=make_estimation_result(**result_override))
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    response = client.post("/api/v1/estimate", json={**VALID_PAYLOAD, **payload_override})
    assert response.status_code == 200

    data = response.json()
    assert {key: data[key] for key in expected_response} == expected_response

    mock_generate.assert_called_once()
    passed_opts = mock_generate.call_args.args[2]
    assert passed_opts.engine == expected_engine
    assert passed_opts.model == payload_override.get("model", passed_opts.model)


def test_create_estimation_reflects_configured_active_prompt_version(
    client: TestClient,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a custom active prompt version resolved by resolve_prompt_version
    WHEN a POST request is processed by /api/v1/estimate
    THEN the rendered prompt and response metadata reflect that configured active version.

    :param client: Shared TestClient fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            estimation="Result with active version v2 candidate.",
            model=DEFAULT_GPT_LLM,
            provider="openai",
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)
    monkeypatch.setattr("aie_estimator.routers.estimations_router.resolve_prompt_version", lambda: "v2-candidate")

    with patch("aie_estimator.routers.estimations_router.render_estimation_prompt") as mock_render:
        mock_render.return_value = ("System prompt v2", "User prompt v2")

        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["prompt_version"] == "v2-candidate"

        mock_render.assert_called_once()
        _, call_kwargs = mock_render.call_args
        assert call_kwargs.get("version") == "v2-candidate"


def test_create_estimation_input_guardrail_violation_returns_400(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an estimation request that triggers an input guardrail violation
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 400 Bad Request is returned with structured reason and message in the detail object.

    :param client: Shared TestClient fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        side_effect=InputGuardrailViolation(
            "Suspicious instruction-like text detected: 'ignore previous instructions'",
            reason="prompt_injection",
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["reason"] == "prompt_injection"
    assert "Suspicious instruction-like text" in data["detail"]["message"]

