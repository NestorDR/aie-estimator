# tests/services/test_llm_service.py

# --- Python Standard Library ---
import asyncio
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
import litellm
from openai import BadRequestError, RateLimitError
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.schemas.models import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
)
from aie_estimator.services import LLMServiceError, llm_service
from aie_estimator.services.litellm_provider import call_litellm_router, get_litellm_router
from aie_estimator.services.llm_service import GenerationOptions
from aie_estimator.services.native_providers import (
    call_anthropic,
    call_openai,
    get_anthropic_client,
    get_openai_client,
)
from aie_estimator.settings import Environment, Settings

_SAMPLE_TRANSCRIPT: Final[str] = (
    "Client: We need a scalable notification service supporting email and SMS.\n"
    "Team: We can implement webhooks with AWS SNS/SES.\n"
    "Client: Launch target is within 3 weeks."
)


@pytest.fixture(autouse=True)
def mock_default_cache(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """
    Prevent unit tests in test_llm_service from opening real TCP sockets to Redis on cache lookups.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Configured AsyncMock simulating a cache miss by default.
    """
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = None
    mock_cache.close.return_value = None
    monkeypatch.setattr("aie_estimator.services.llm_service.get_estimation_cache", lambda: mock_cache)
    return mock_cache


def test_client_caching_and_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured API credentials
    WHEN client accessor functions are invoked multiple times
    THEN the client instances are cached via lru_cache and reused until explicitly cleared.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-openai-key"),
        anthropic_api_key=SecretStr("sk-test-anthropic-key"),
    )
    monkeypatch.setattr("aie_estimator.services.native_providers.get_settings", lambda: test_settings)

    openai_1 = get_openai_client()
    openai_2 = get_openai_client()
    assert openai_1 is openai_2

    anthropic_1 = get_anthropic_client()
    anthropic_2 = get_anthropic_client()
    assert anthropic_1 is anthropic_2

    get_openai_client.cache_clear()
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
    monkeypatch.setattr("aie_estimator.services.native_providers.get_settings", lambda: empty_settings)
    get_openai_client.cache_clear()
    get_anthropic_client.cache_clear()

    with pytest.raises(LLMServiceError, match=r"OpenAI API key"):
        get_openai_client()

    with pytest.raises(LLMServiceError, match=r"Anthropic API key"):
        get_anthropic_client()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "is_reasoning"),
    [
        ("custom-standard-model", False),
        (DEFAULT_GPT_LLM, True),
    ],
    ids=["custom_standard", "gpt6_reasoning"],
)
async def test_call_openai_model_parameter_selection(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    is_reasoning: bool,
) -> None:
    """
    GIVEN a target model identifier
    WHEN call_openai executes against the Responses API
    THEN reasoning models receive reasoning effort while non-reasoning models receive temperature.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param model: Target model identifier.
    :param is_reasoning: Whether model is classified as reasoning-capable.
    """
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status = "completed"
    mock_resp.output_text = "Detailed estimation content."
    mock_resp.model = model
    mock_resp.usage.input_tokens = 250
    mock_resp.usage.output_tokens = 150
    mock_resp.usage.total_tokens = 400
    mock_client.responses.create.return_value = mock_resp

    monkeypatch.setattr("aie_estimator.services.native_providers.get_openai_client", lambda: mock_client)

    result = await call_openai(
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


@pytest.mark.anyio
async def test_call_openai_temperature_bad_request_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an OpenAI call initially rejected due to an unsupported temperature parameter
    WHEN BadRequestError mentioning temperature is received
    THEN temperature is stripped and the call is retried successfully.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status = "completed"
    mock_resp.output_text = "Recovered estimation."
    mock_resp.model = "custom-standard-model"
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

    async def mock_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("aie_estimator.services.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    result = await call_openai(
        messages=_SAMPLE_TRANSCRIPT,
        system_prompt=None,
        model="custom-standard-model",
        max_tokens=2000,
        temperature=0.3,
    )

    assert result["estimation"] == "Recovered estimation."
    assert mock_client.responses.create.call_count == 2
    second_call_kwargs = mock_client.responses.create.call_args_list[1].kwargs
    assert "temperature" not in second_call_kwargs


@pytest.mark.anyio
async def test_call_openai_retry_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN persistent transient rate limits from OpenAI
    WHEN retry attempts exceed the maximum configured count
    THEN LLMServiceError is raised with logged diagnostic context.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_client = AsyncMock()
    err_response = MagicMock()
    err_response.status_code = 429
    rate_err = RateLimitError(
        message="Rate limit reached",
        response=err_response,
        body={"error": {"message": "Rate limit"}},
    )
    mock_client.responses.create.side_effect = rate_err

    async def mock_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("aie_estimator.services.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    with pytest.raises(LLMServiceError, match=r"OpenAI call failed"):
        await call_openai(
            messages=_SAMPLE_TRANSCRIPT,
            system_prompt=None,
            model=DEFAULT_GPT_LLM,
            max_tokens=1000,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    [
        (None, None),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
    ],
    ids=[
        "thinking_disabled",
        "effort_low",
        "effort_medium",
        "effort_high",
    ],
)
async def test_call_anthropic_thinking_and_temperature_exclusivity(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str | None,
    expected_effort: str | None,
) -> None:
    """
    GIVEN Anthropic configuration with or without qualitative reasoning effort
    WHEN call_anthropic executes
    THEN adaptive thinking and temperature are strictly mutually exclusive.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param reasoning_effort: Explicit qualitative reasoning effort tier.
    :param expected_effort: Expected effort tier passed to output_config.
    """
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.model = DEFAULT_CLAUDE_LLM
    mock_resp.stop_reason = "end_turn"
    mock_resp.usage.input_tokens = 300
    mock_resp.usage.output_tokens = 200

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Anthropic estimation output."
    mock_resp.content = [text_block]
    mock_client.messages.create.return_value = mock_resp

    monkeypatch.setattr("aie_estimator.services.native_providers.get_anthropic_client", lambda: mock_client)

    result = await call_anthropic(
        messages=_SAMPLE_TRANSCRIPT,
        system_prompt="CAG instructions",
        model=DEFAULT_CLAUDE_LLM,
        max_tokens=4000,
        temperature=0.3,
        reasoning_effort=reasoning_effort,
    )

    assert result["estimation"] == "Anthropic estimation output."
    assert result["finish_reason"] == "end_turn"
    assert result["usage"]["total_tokens"] == 500

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 4000
    assert call_kwargs["messages"] == [{"role": "user", "content": _SAMPLE_TRANSCRIPT}]
    assert call_kwargs["system"] == "CAG instructions"

    if expected_effort is not None:
        assert call_kwargs.get("thinking") == {"type": "adaptive"}
        assert call_kwargs.get("output_config") == {"effort": expected_effort}
        assert "temperature" not in call_kwargs
    else:
        assert call_kwargs.get("temperature") == 0.3
        assert "thinking" not in call_kwargs
        assert "output_config" not in call_kwargs


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
        llm_model=DEFAULT_GPT_LLM,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_extract_resp = {
        "estimation": "- Functional requirement: Webhooks\n- Non-functional requirement: High availability",
        "model": DEFAULT_GPT_LLM,
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
    }
    mock_final_resp = {
        "estimation": "## Project summary\nFinal estimation breakdown.",
        "model": DEFAULT_GPT_LLM,
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
    }

    call_count = 0

    async def mock_call_openai(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return mock_extract_resp if call_count == 1 else mock_final_resp

    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_call_openai)

    opts = GenerationOptions(preprocessing="two_phase", engine="native")
    result = asyncio.run(llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts))

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

    async def mock_failing_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
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
        (DEFAULT_CLAUDE_LLM, "openai", "anthropic"),
        (DEFAULT_GPT_LLM, "anthropic", "openai"),
        (None, "openai", "openai"),
        (None, "anthropic", "anthropic"),
    ],
    ids=[
        "claude_haiku_routes_to_anthropic",
        "gpt6_routes_to_openai",
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
    default_model = DEFAULT_CLAUDE_LLM if default_provider == "anthropic" else DEFAULT_GPT_LLM
    test_settings = Settings(
        app_env=Environment.TEST,
        llm_model=default_model,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_openai = AsyncMock(
        return_value={
            "estimation": "OpenAI estimate",
            "model": model_override or DEFAULT_GPT_LLM,
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }
    )
    mock_anthropic = AsyncMock(
        return_value={
            "estimation": "Anthropic estimate",
            "model": model_override or DEFAULT_GPT_LLM,
            "provider": "anthropic",
            "finish_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        }
    )

    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service.call_anthropic", mock_anthropic)

    opts = GenerationOptions(model=model_override, engine="native")
    result = asyncio.run(llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts))

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
        llm_model=DEFAULT_GPT_LLM,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_cache = AsyncMock()
    cached_payload: dict[str, Any] = {
        "estimation": "Cached software estimation breakdown.",
        "model": DEFAULT_GPT_LLM,
        "provider": "openai",
        "finish_reason": "stop",
        "usage": {"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
    }
    mock_cache.get.return_value = cached_payload
    monkeypatch.setattr("aie_estimator.services.llm_service.get_estimation_cache", lambda: mock_cache)

    mock_openai = AsyncMock()
    mock_anthropic = AsyncMock()
    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service.call_anthropic", mock_anthropic)

    result = asyncio.run(llm_service.generate_estimation(_SAMPLE_TRANSCRIPT))

    assert result["cache_hit"] is True
    assert result["estimation"] == "Cached software estimation breakdown."
    mock_openai.assert_not_called()
    mock_anthropic.assert_not_called()


def test_get_litellm_router_credential_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN partial API key credentials in the environment
    WHEN get_litellm_router is initialized
    THEN models lacking required environment variables are omitted from the active router model list.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.delenv("SCALEWAY_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    get_litellm_router.cache_clear()
    router = get_litellm_router()
    active_models = [entry.get("litellm_params", {}).get("model") for entry in router.model_list]

    # Models needing missing keys must not be in active router list
    assert "openai/deepseek-ai/DeepSeek-V4-Flash" not in active_models
    assert "mistral/mistral-small-3.2-24b-instruct" not in active_models
    assert "openai/deepseek-ai/DeepSeek-R2" not in active_models
    # Models using OPENAI_API_KEY must be kept
    assert f"openai/{DEFAULT_GPT_LLM}" in active_models


def test_get_litellm_router_zero_credentials_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN no API keys configured in the environment
    WHEN get_litellm_router is initialized
    THEN LLMServiceError is raised indicating no valid credentials.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "SCALEWAY_API_KEY", "NEBIUS_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    get_litellm_router.cache_clear()
    with pytest.raises(LLMServiceError, match=r"No models in Router have valid credentials"):
        get_litellm_router()


def test_get_litellm_router_zero_telemetry_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN get_litellm_router initialization
    WHEN router bootstrapping executes
    THEN zero-telemetry and parameter normalization flags are enforced on litellm.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    get_litellm_router.cache_clear()
    get_litellm_router()

    assert litellm.telemetry is False
    assert litellm.drop_params is True
    assert litellm.modify_params is True


def test_call_router_cost_extraction_and_response_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a mocked LiteLLM Router completion response containing hidden cost params
    WHEN call_litellm_router is executed
    THEN response metadata, token counts, and cost_usd are extracted accurately.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "Estimated Effort: 42 hours."

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 250
    mock_usage.completion_tokens = 75
    mock_usage.total_tokens = 325

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = f"openai/{DEFAULT_GPT_LLM}"
    mock_response._hidden_params = {
        "response_cost": 0.00035,
        "custom_llm_provider": "openai",
    }

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("aie_estimator.services.litellm_provider.get_litellm_router", lambda: mock_router)

    result = asyncio.run(
        call_litellm_router(
            messages="Analyze this project transcript",
            system_prompt="You are an estimator",
            model=DEFAULT_FAST_PROFILE,
            max_tokens=2000,
        )
    )

    assert result["estimation"] == "Estimated Effort: 42 hours."
    assert result["model"] == f"openai/{DEFAULT_GPT_LLM}"
    assert result["provider"] == "litellm"
    assert result["finish_reason"] == "stop"
    assert result["cost_usd"] == 0.00035
    assert result["usage"]["input_tokens"] == 250
    assert result["usage"]["output_tokens"] == 75
    assert result["usage"]["total_tokens"] == 325


def test_generate_estimation_litellm_engine_routes_to_router(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN engine='litellm' in GenerationOptions
    WHEN generate_estimation is executed
    THEN the call dispatches through call_litellm_router and extracts cost_usd.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_router_call = AsyncMock(
        return_value={
            "estimation": "Router generated estimation breakdown.",
            "model": DEFAULT_FAST_PROFILE,
            "provider": "litellm",
            "finish_reason": "stop",
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "cost_usd": 0.00045,
        }
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.call_litellm_router", mock_router_call)

    opts = GenerationOptions(engine="litellm", model=DEFAULT_FAST_PROFILE)
    result = asyncio.run(llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts))

    mock_router_call.assert_called_once()
    assert result["estimation"] == "Router generated estimation breakdown."
    assert result["cost_usd"] == 0.00045
    assert result["provider"] == "litellm"


def test_generate_estimation_native_engine_rejects_unsupported_provider() -> None:
    """
    GIVEN engine='native' with a logical profile or unsupported model
    WHEN generate_estimation is executed
    THEN LLMServiceError is raised indicating the unsupported provider.
    """
    opts = GenerationOptions(engine="native", model=DEFAULT_FAST_PROFILE)
    with pytest.raises(LLMServiceError, match=r"Unsupported provider"):
        asyncio.run(llm_service.generate_estimation(_SAMPLE_TRANSCRIPT, opts))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    [
        (None, None),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
    ],
    ids=[
        "router_reasoning_disabled",
        "router_low",
        "router_medium",
        "router_high",
    ],
)
async def test_call_router_dispatches_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str | None,
    expected_effort: str | None,
) -> None:
    """
    GIVEN qualitative reasoning effort tier
    WHEN call_litellm_router executes
    THEN reasoning_effort is forwarded directly to router.acompletion and extra_body is omitted.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param reasoning_effort: Explicit qualitative reasoning effort tier.
    :param expected_effort: Expected qualitative tier forwarded to router.acompletion.
    """
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "Router estimation output."

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = f"openai/{DEFAULT_GPT_LLM}"
    mock_response._hidden_params = {}

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("aie_estimator.services.litellm_provider.get_litellm_router", lambda: mock_router)

    result = await call_litellm_router(
        messages="Analyze project",
        system_prompt="System prompt",
        model=DEFAULT_FAST_PROFILE,
        max_tokens=2000,
        reasoning_effort=reasoning_effort,
    )

    assert result["estimation"] == "Router estimation output."
    call_kwargs = mock_router.acompletion.call_args.kwargs
    assert "extra_body" not in call_kwargs

    if expected_effort is not None:
        assert call_kwargs.get("reasoning_effort") == expected_effort
    else:
        assert "reasoning_effort" not in call_kwargs
