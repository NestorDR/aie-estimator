# tests/services/test_llm_service.py

# --- Python Standard Library ---
from typing import Any, Final
from unittest.mock import MagicMock

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import BadRequestError, RateLimitError
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.services import llm_service
from aie_estimator.services.llm_service import (
    GenerationOptions,
    LLMServiceError,
    _call_anthropic,
    _call_openai,
    _reset_clients,
    get_anthropic_client,
    get_openai_client,
)
from aie_estimator.settings import Environment, Settings

_SAMPLE_TRANSCRIPT: Final[str] = (
    "Client: We need a scalable notification service supporting email and SMS.\n"
    "Team: We can implement webhooks with AWS SNS/SES.\n"
    "Client: Launch target is within 3 weeks."
)


def test_client_caching_and_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured API credentials
    WHEN client accessor functions are invoked multiple times
    THEN the client instances are cached and reused until explicitly reset.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-openai-key"),
        anthropic_api_key=SecretStr("sk-test-anthropic-key"),
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    openai_1 = get_openai_client()
    openai_2 = get_openai_client()
    assert openai_1 is openai_2

    anthropic_1 = get_anthropic_client()
    anthropic_2 = get_anthropic_client()
    assert anthropic_1 is anthropic_2

    _reset_clients()
    openai_3 = get_openai_client()
    assert openai_3 is not openai_1


def test_client_missing_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN unconfigured API credentials
    WHEN client accessor functions are invoked
    THEN LLMServiceError is raised indicating missing configuration.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    empty_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=None,
        anthropic_api_key=None,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: empty_settings)
    _reset_clients()

    with pytest.raises(LLMServiceError, match=r"OpenAI API key"):
        get_openai_client()

    with pytest.raises(LLMServiceError, match=r"Anthropic API key"):
        get_anthropic_client()


@pytest.mark.parametrize(
    ("model", "is_reasoning"),
    [
        ("gpt-4o-mini", False),
        ("gpt-4o", False),
        ("o3-mini", True),
        ("gpt-6-luna", True),
    ],
    ids=["gpt4o_mini_standard", "gpt4o_standard", "o3_reasoning", "gpt6_reasoning"],
)
def test_call_openai_model_parameter_selection(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    is_reasoning: bool,
) -> None:
    """
    GIVEN a target model identifier
    WHEN _call_openai executes against the Responses API
    THEN reasoning models receive reasoning effort while non-reasoning models receive temperature.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param model: Target model identifier.
    :param is_reasoning: Whether model is classified as reasoning-capable.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = "completed"
    mock_resp.output_text = "Detailed estimation content."
    mock_resp.model = model
    mock_resp.usage.input_tokens = 250
    mock_resp.usage.output_tokens = 150
    mock_resp.usage.total_tokens = 400
    mock_client.responses.create.return_value = mock_resp

    monkeypatch.setattr("aie_estimator.services.llm_service.get_openai_client", lambda: mock_client)

    result = _call_openai(
        messages=_SAMPLE_TRANSCRIPT,
        system_prompt="You are a CAG estimator.",
        model=model,
        max_tokens=3000,
        temperature=0.3,
        effort="medium",
    )

    assert result["estimation"] == "Detailed estimation content."
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 400

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == model
    assert call_kwargs["input"] == _SAMPLE_TRANSCRIPT
    assert call_kwargs["instructions"] == "You are a CAG estimator."
    assert call_kwargs["max_output_tokens"] == 3000

    if is_reasoning:
        assert call_kwargs.get("reasoning") == {"effort": "medium"}
        assert "temperature" not in call_kwargs
    else:
        assert call_kwargs.get("temperature") == 0.3
        assert "reasoning" not in call_kwargs


def test_call_openai_temperature_bad_request_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an OpenAI call initially rejected due to an unsupported temperature parameter
    WHEN BadRequestError mentioning temperature is received
    THEN temperature is stripped and the call is retried successfully.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = "completed"
    mock_resp.output_text = "Recovered estimation."
    mock_resp.model = "o3-mini"
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    mock_resp.usage.total_tokens = 150

    err_response = MagicMock()
    err_response.status_code = 400
    temp_err = BadRequestError(
        message="Unsupported parameter: temperature is not supported by this model",
        response=err_response,
        body={"error": {"message": "temperature unsupported"}},
    )
    mock_client.responses.create.side_effect = [temp_err, mock_resp]

    monkeypatch.setattr("aie_estimator.services.llm_service.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("time.sleep", lambda _: None)

    result = _call_openai(
        messages=_SAMPLE_TRANSCRIPT,
        system_prompt=None,
        model="gpt-4o",  # Standard model string passing temperature initially
        max_tokens=2000,
        temperature=0.3,
    )

    assert result["estimation"] == "Recovered estimation."
    assert mock_client.responses.create.call_count == 2
    second_call_kwargs = mock_client.responses.create.call_args_list[1].kwargs
    assert "temperature" not in second_call_kwargs


def test_call_openai_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN persistent transient rate limits from OpenAI
    WHEN retry attempts exceed the maximum configured count
    THEN LLMServiceError is raised with logged diagnostic context.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_client = MagicMock()
    err_response = MagicMock()
    err_response.status_code = 429
    rate_err = RateLimitError(
        message="Rate limit reached",
        response=err_response,
        body={"error": {"message": "Rate limit"}},
    )
    mock_client.responses.create.side_effect = rate_err

    monkeypatch.setattr("aie_estimator.services.llm_service.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(LLMServiceError, match=r"OpenAI call failed"):
        _call_openai(
            messages=_SAMPLE_TRANSCRIPT,
            system_prompt=None,
            model="gpt-4o-mini",
            max_tokens=1000,
        )


@pytest.mark.parametrize(
    ("thinking_budget", "expected_max_tokens"),
    [
        (None, 4000),
        (2048, max(4000, 2048 + 1024)),
    ],
    ids=["thinking_disabled", "thinking_enabled"],
)
def test_call_anthropic_thinking_and_temperature_exclusivity(
    monkeypatch: pytest.MonkeyPatch,
    thinking_budget: int | None,
    expected_max_tokens: int,
) -> None:
    """
    GIVEN Anthropic configuration with or without extended thinking budget
    WHEN _call_anthropic executes
    THEN thinking and temperature are strictly mutually exclusive and tokens are appropriately padded.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param thinking_budget: Extended thinking token budget.
    :param expected_max_tokens: Expected max_tokens parameter after padding.
    """
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.model = "claude-haiku-4-5-20251001"
    mock_resp.stop_reason = "end_turn"
    mock_resp.usage.input_tokens = 300
    mock_resp.usage.output_tokens = 200

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Anthropic estimation output."
    mock_resp.content = [text_block]
    mock_client.messages.create.return_value = mock_resp

    monkeypatch.setattr("aie_estimator.services.llm_service.get_anthropic_client", lambda: mock_client)

    result = _call_anthropic(
        messages=_SAMPLE_TRANSCRIPT,
        system_prompt="CAG instructions",
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        thinking_budget=thinking_budget,
        temperature=0.3,
    )

    assert result["estimation"] == "Anthropic estimation output."
    assert result["finish_reason"] == "end_turn"
    assert result["usage"]["total_tokens"] == 500

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == expected_max_tokens
    assert call_kwargs["messages"] == [{"role": "user", "content": _SAMPLE_TRANSCRIPT}]
    assert call_kwargs["system"] == "CAG instructions"

    if thinking_budget is not None:
        assert call_kwargs.get("thinking") == {"type": "enabled", "budget_tokens": thinking_budget}
        assert "temperature" not in call_kwargs
    else:
        assert call_kwargs.get("temperature") == 0.3
        assert "thinking" not in call_kwargs


def test_generate_estimation_two_phase_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an estimation request configured with two_phase preprocessing
    WHEN generate_estimation executes
    THEN extract_requirements executes phase 1 before the estimation generation phase.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-key"),
        llm_model="gpt-4o-mini",
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_extract_resp = {
        "estimation": "- Functional requirement: Webhooks\n- Non-functional requirement: High availability",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
    }
    mock_final_resp = {
        "estimation": "## Project summary\nFinal estimation breakdown.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
    }

    call_count = 0

    def mock_call_openai(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return mock_extract_resp if call_count == 1 else mock_final_resp

    monkeypatch.setattr("aie_estimator.services.llm_service._call_openai", mock_call_openai)

    opts = GenerationOptions(preprocessing="two_phase")
    result = llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts)

    assert result["preprocessing"] == "two_phase"
    assert "Webhooks" in str(result["extracted_requirements"])
    assert result["usage"]["preprocessing_input_tokens"] == 80
    assert result["usage"]["preprocessing_output_tokens"] == 40
    assert call_count == 2


def test_server_llm_service_error_handler(app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a failing LLM provider invocation
    WHEN a client sends a POST request to /api/v1/estimate
    THEN HTTP 500 is returned with a consistent, sanitized error message.

    :param app: FastAPI application fixture.
    :param client: TestClient fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """

    def mock_failing_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise LLMServiceError("Upstream provider connection refused.")

    monkeypatch.setattr("aie_estimator.services.llm_service.generate_estimation", mock_failing_generation)

    response = client.post(
        "/api/v1/estimate",
        json={"transcription": "This transcript is long enough to satisfy schema validation constraints for testing."},
    )

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error occurred while generating the estimation" in data["detail"].lower()


@pytest.mark.parametrize(
    ("model_override", "default_provider", "expected_called_provider"),
    [
        ("claude-haiku-4-5-20251001", "openai", "anthropic"),
        ("gpt-6-luna", "anthropic", "openai"),
        ("gpt-4o-mini", "anthropic", "openai"),
        (None, "openai", "openai"),
        (None, "anthropic", "anthropic"),
    ],
    ids=[
        "claude_haiku_routes_to_anthropic",
        "gpt6_routes_to_openai",
        "gpt4o_mini_routes_to_openai",
        "default_model_uses_env_openai",
        "default_model_uses_env_anthropic",
    ],
)
def test_generate_estimation_model_driven_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    model_override: str | None,
    default_provider: str,
    expected_called_provider: str,
) -> None:
    """
    GIVEN a model override and a default environment provider
    WHEN generate_estimation is executed
    THEN the call is dispatched to the provider matching the model's catalog specification.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param model_override: Target model identifier or None.
    :param default_provider: Configured environment default provider.
    :param expected_called_provider: Expected provider invoked ('openai' or 'anthropic').
    """
    default_model = "claude-haiku-4-5-20251001" if default_provider == "anthropic" else "gpt-4o-mini"
    test_settings = Settings(
        app_env=Environment.TEST,
        llm_model=default_model,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_openai = MagicMock(
        return_value={
            "estimation": "OpenAI estimate",
            "model": model_override or "gpt-4o-mini",
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }
    )
    mock_anthropic = MagicMock(
        return_value={
            "estimation": "Anthropic estimate",
            "model": model_override or "gpt-4o-mini",
            "provider": "anthropic",
            "finish_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }
    )

    monkeypatch.setattr("aie_estimator.services.llm_service._call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service._call_anthropic", mock_anthropic)

    opts = GenerationOptions(model=model_override)
    result = llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts)

    if expected_called_provider == "openai":
        assert mock_openai.call_count == 1
        assert mock_anthropic.call_count == 0
        assert result["provider"] == "openai"
    else:
        assert mock_anthropic.call_count == 1
        assert mock_openai.call_count == 0
        assert result["provider"] == "anthropic"


def test_generate_estimation_cache_hit_bypasses_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a cached estimation entry in EstimationCache
    WHEN generate_estimation is executed with matching parameters
    THEN the cached result is returned with cache_hit=True without calling any provider.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        llm_model="gpt-4o-mini",
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_cache = MagicMock()
    cached_payload: dict[str, Any] = {
        "estimation": "Cached software estimation breakdown.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
    }
    mock_cache.get.return_value = cached_payload
    monkeypatch.setattr("aie_estimator.services.llm_service.get_estimation_cache", lambda: mock_cache)

    mock_openai = MagicMock()
    mock_anthropic = MagicMock()
    monkeypatch.setattr("aie_estimator.services.llm_service._call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service._call_anthropic", mock_anthropic)

    result = llm_service.generate_estimation(_SAMPLE_TRANSCRIPT)

    assert result["cache_hit"] is True
    assert result["estimation"] == "Cached software estimation breakdown."
    mock_openai.assert_not_called()
    mock_anthropic.assert_not_called()
