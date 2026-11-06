# tests/services/test_native_providers.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from openai import BadRequestError, RateLimitError
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.services.providers.model_catalog import DEFAULT_CLAUDE_LLM, DEFAULT_GPT_LLM
from aie_estimator.services import LLMProviderError
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.services.providers.native_providers import (
    call_anthropic,
    call_openai,
    get_anthropic_client,
    get_openai_client,
)
from aie_estimator.settings import Environment, Settings


def test_clients_built_from_configured_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured API credentials
    WHEN client accessor functions are invoked
    THEN clients are built with the configured keys.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-openai-key"),
        anthropic_api_key=SecretStr("sk-test-anthropic-key"),
    )
    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_settings", lambda: test_settings)

    assert get_openai_client().api_key == "sk-test-openai-key"
    assert get_anthropic_client().api_key == "sk-test-anthropic-key"


def test_client_missing_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN unconfigured API credentials
    WHEN client accessor functions are invoked
    THEN LLMProviderError is raised indicating missing configuration.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    empty_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=None,
        anthropic_api_key=None,
    )
    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_settings", lambda: empty_settings)
    get_openai_client.cache_clear()
    get_anthropic_client.cache_clear()

    with pytest.raises(LLMProviderError, match=r"OpenAI API key"):
        get_openai_client()

    with pytest.raises(LLMProviderError, match=r"Anthropic API key"):
        get_anthropic_client()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "expected_params", "absent_params", "expected_cost"),
    [
        ("custom-standard-model", {"temperature": 0.3}, ("reasoning",), None),
        (DEFAULT_GPT_LLM, {"reasoning": {"effort": "medium"}}, ("temperature",), 0.0001),
    ],
    ids=["custom_standard", "gpt6_reasoning"],
)
async def test_call_openai_model_parameter_selection(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_params: dict[str, Any],
    absent_params: tuple[str, ...],
    expected_cost: float | None,
) -> None:
    """
    GIVEN a target model identifier
    WHEN call_openai executes against the Responses API
    THEN reasoning models receive reasoning effort while non-reasoning models receive temperature.

    :param sample_transcription: Sample transcript fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param model: Target model identifier.
    :param expected_params: Sampling parameters expected in the request.
    :param absent_params: Sampling parameters that must not be sent.
    :param expected_cost: Expected cost_usd (None when the model has no pricing entry).
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

    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_openai_client", lambda: mock_client)

    opts = GenerationOptions(
        model=model,
        max_tokens=3000,
        temperature=0.3,
        reasoning_effort="medium",
    )
    result = await call_openai(
        user_prompt=sample_transcription,
        system_prompt="You are a CAG estimator.",
        opts=opts,
    )

    assert result["estimation"] == "Detailed estimation content."
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 400
    assert result.get("cost_usd") == expected_cost

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["model"] == model
    assert call_kwargs["input"] == sample_transcription
    assert call_kwargs["instructions"] == "You are a CAG estimator."
    assert call_kwargs["max_output_tokens"] == 3000
    assert {k: call_kwargs.get(k) for k in expected_params} == expected_params
    assert all(k not in call_kwargs for k in absent_params)


@pytest.mark.anyio
async def test_call_openai_temperature_bad_request_stripping(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an OpenAI call initially rejected due to an unsupported temperature parameter
    WHEN BadRequestError mentioning temperature is received
    THEN temperature is stripped and the call is retried successfully.

    :param sample_transcription: Sample transcript fixture.
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

    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    opts = GenerationOptions(
        model="custom-standard-model",
        max_tokens=2000,
        temperature=0.3,
    )
    result = await call_openai(
        user_prompt=sample_transcription,
        system_prompt=None,
        opts=opts,
    )

    assert result["estimation"] == "Recovered estimation."
    assert mock_client.responses.create.call_count == 2
    second_call_kwargs = mock_client.responses.create.call_args_list[1].kwargs
    assert "temperature" not in second_call_kwargs


@pytest.mark.anyio
async def test_call_openai_retry_exhaustion(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN persistent transient rate limits from OpenAI
    WHEN retry attempts exceed the maximum configured count
    THEN LLMProviderError is raised with logged diagnostic context.

    :param sample_transcription: Sample transcript fixture.
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

    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    with pytest.raises(LLMProviderError, match=r"OpenAI call failed"):
        opts = GenerationOptions(model=DEFAULT_GPT_LLM, max_tokens=1000)
        await call_openai(
            user_prompt=sample_transcription,
            system_prompt=None,
            opts=opts,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_params", "absent_params"),
    [
        (None, {"temperature": 0.3}, ("thinking", "output_config")),
        ("medium", {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}, ("temperature",)),
    ],
    ids=["thinking_disabled", "effort_medium"],
)
async def test_call_anthropic_thinking_and_temperature_exclusivity(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str | None,
    expected_params: dict[str, Any],
    absent_params: tuple[str, ...],
) -> None:
    """
    GIVEN Anthropic configuration with or without qualitative reasoning effort
    WHEN call_anthropic executes
    THEN adaptive thinking and temperature are strictly mutually exclusive.

    :param sample_transcription: Sample transcript fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param reasoning_effort: Explicit qualitative reasoning effort tier.
    :param expected_params: Sampling parameters expected in the request.
    :param absent_params: Sampling parameters that must not be sent.
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

    monkeypatch.setattr("aie_estimator.services.providers.native_providers.get_anthropic_client", lambda: mock_client)

    opts = GenerationOptions(
        model=DEFAULT_CLAUDE_LLM,
        max_tokens=4000,
        temperature=0.3,
        reasoning_effort=reasoning_effort,
    )
    result = await call_anthropic(
        user_prompt=sample_transcription,
        system_prompt="CAG instructions",
        opts=opts,
    )

    assert result["estimation"] == "Anthropic estimation output."
    assert result["finish_reason"] == "end_turn"
    assert result["usage"]["total_tokens"] == 500

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 4000
    assert call_kwargs["messages"] == [{"role": "user", "content": sample_transcription}]
    assert call_kwargs["system"] == "CAG instructions"
    assert {k: call_kwargs.get(k) for k in expected_params} == expected_params
    assert all(k not in call_kwargs for k in absent_params)
