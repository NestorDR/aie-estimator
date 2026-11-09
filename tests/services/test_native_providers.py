# tests/services/test_native_providers.py

# --- Python Standard Library ---
import json
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from instructor.core import InstructorRetryException
from openai import BadRequestError, RateLimitError
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResult
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.services.providers.native_providers import (
    call_anthropic,
    call_openai,
    get_anthropic_client,
    get_openai_client,
)
from aie_estimator.settings import Environment, Settings
from aie_estimator.utils.model_constants import DEFAULT_CLAUDE_LLM, DEFAULT_GPT_LLM

VALID_RESULT_DICT: Final[dict[str, Any]] = {
    "summary": "Full-stack web application delivery breakdown.",
    "confidence_pct": 85,
    "phases": [
        {
            "name": "Design & Architecture",
            "duration_weeks": 2,
            "cost_eur": 5000,
            "summary": "Technical architecture specification and UX wireframes.",
        },
        {
            "name": "Core Implementation",
            "duration_weeks": 6,
            "cost_eur": 15000,
            "summary": "Frontend, backend services, database schema, and test automation.",
        },
    ],
    "total_duration_weeks": 8,
    "total_cost_eur": 20000,
}


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
    ("model", "expected_params", "absent_params", "expected_token_param", "expected_cost"),
    [
        (
            "custom-standard-model",
            {"temperature": 0.3},
            ("reasoning_effort", "max_completion_tokens"),
            "max_tokens",
            None,
        ),
        (
            DEFAULT_GPT_LLM,
            {"reasoning_effort": "medium"},
            ("temperature", "max_tokens"),
            "max_completion_tokens",
            0.0001,
        ),
    ],
    ids=["custom_standard", "gpt6_reasoning"],
)
async def test_call_openai_model_parameter_selection(
    sample_transcription: str,
    mock_async_openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_params: dict[str, Any],
    absent_params: tuple[str, ...],
    expected_token_param: str,
    expected_cost: float | None,
) -> None:
    """
    GIVEN a target model identifier
    WHEN call_openai executes against the chat completions API
    THEN reasoning models receive reasoning effort while non-reasoning models receive temperature.

    :param sample_transcription: Sample transcript fixture.
    :param mock_async_openai_client: Reusable typed AsyncOpenAI mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param model: Target model identifier.
    :param expected_params: Sampling parameters expected in the request.
    :param absent_params: Sampling parameters that must not be sent.
    :param expected_token_param: Token limit parameter name expected ('max_tokens' or 'max_completion_tokens').
    :param expected_cost: Expected cost_usd (None when the model has no pricing entry).
    """
    mock_create = AsyncMock()

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(VALID_RESULT_DICT)
    mock_choice.message.tool_calls = None
    mock_resp.choices = [mock_choice]
    mock_resp.model = model
    mock_resp.usage.prompt_tokens = 250
    mock_resp.usage.completion_tokens = 150
    mock_resp.usage.total_tokens = 400
    mock_create.return_value = mock_resp
    mock_async_openai_client.chat.completions.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_openai_client",
        lambda: mock_async_openai_client,
    )

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

    assert isinstance(result["result"], EstimationResult)
    assert result["result"].total_cost_eur == 20000
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 400
    assert result.get("cost_usd") == expected_cost

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == model
    assert call_kwargs[expected_token_param] == 3000
    assert {k: call_kwargs.get(k) for k in expected_params} == expected_params
    assert all(k not in call_kwargs for k in absent_params)


@pytest.mark.anyio
async def test_call_openai_temperature_bad_request_stripping(
    sample_transcription: str,
    mock_async_openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an OpenAI call initially rejected due to an unsupported temperature parameter
    WHEN BadRequestError mentioning temperature is received
    THEN temperature is stripped and the call is retried successfully.

    :param sample_transcription: Sample transcript fixture.
    :param mock_async_openai_client: Reusable typed AsyncOpenAI mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_create = AsyncMock()

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(VALID_RESULT_DICT)
    mock_choice.message.tool_calls = None
    mock_resp.choices = [mock_choice]
    mock_resp.model = "custom-standard-model"
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50
    mock_resp.usage.total_tokens = 150

    err_response = MagicMock()
    err_response.status_code = 400
    temp_err = BadRequestError(
        message="Unsupported parameter: temperature is not supported by this model",
        response=err_response,
        body={"error": {"message": "temperature unsupported"}},
    )
    mock_create.side_effect = [temp_err, mock_resp]
    mock_async_openai_client.chat.completions.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_openai_client",
        lambda: mock_async_openai_client,
    )

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

    assert isinstance(result["result"], EstimationResult)
    assert mock_create.call_count == 2
    second_call_kwargs = mock_create.call_args_list[1].kwargs
    assert "temperature" not in second_call_kwargs


@pytest.mark.anyio
async def test_call_openai_converts_max_tokens_to_completion_tokens_on_bad_request(
    sample_transcription: str,
    mock_async_openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an OpenAI model returning a BadRequestError for max_tokens
    WHEN call_openai executes
    THEN max_tokens is converted to max_completion_tokens and retry succeeds.

    :param sample_transcription: Sample transcript fixture.
    :param mock_async_openai_client: Reusable typed AsyncOpenAI mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_create = AsyncMock()

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = json.dumps(VALID_RESULT_DICT)
    mock_choice.message.tool_calls = None
    mock_resp.choices = [mock_choice]
    mock_resp.model = "custom-standard-model"
    mock_resp.usage.prompt_tokens = 200
    mock_resp.usage.completion_tokens = 100
    mock_resp.usage.total_tokens = 300

    err_response = MagicMock()
    err_response.status_code = 400
    tokens_err = BadRequestError(
        message="Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
        response=err_response,
        body={"error": {"message": "Use 'max_completion_tokens' instead."}},
    )
    mock_create.side_effect = [tokens_err, mock_resp]
    mock_async_openai_client.chat.completions.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_openai_client",
        lambda: mock_async_openai_client,
    )

    opts = GenerationOptions(
        model="custom-standard-model",
        max_tokens=2500,
        temperature=0.3,
    )
    result = await call_openai(
        user_prompt=sample_transcription,
        system_prompt=None,
        opts=opts,
    )

    assert isinstance(result["result"], EstimationResult)
    assert mock_create.call_count == 2
    first_call_kwargs = mock_create.call_args_list[0].kwargs
    assert first_call_kwargs["max_tokens"] == 2500
    assert "max_completion_tokens" not in first_call_kwargs

    second_call_kwargs = mock_create.call_args_list[1].kwargs
    assert second_call_kwargs["max_completion_tokens"] == 2500
    assert "max_tokens" not in second_call_kwargs


@pytest.mark.anyio
async def test_call_openai_retry_exhaustion(
    sample_transcription: str,
    mock_async_openai_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN persistent transient rate limits from OpenAI
    WHEN retry attempts exceed the maximum configured count
    THEN LLMProviderError is raised with logged diagnostic context.

    :param sample_transcription: Sample transcript fixture.
    :param mock_async_openai_client: Reusable typed AsyncOpenAI mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_create = AsyncMock()

    err_response = MagicMock()
    err_response.status_code = 429
    rate_err = RateLimitError(
        message="Rate limit reached",
        response=err_response,
        body={"error": {"message": "Rate limit"}},
    )
    mock_create.side_effect = rate_err
    mock_async_openai_client.chat.completions.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_openai_client",
        lambda: mock_async_openai_client,
    )

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
    mock_async_anthropic_client: MagicMock,
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
    :param mock_async_anthropic_client: Reusable typed AsyncAnthropic mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param reasoning_effort: Explicit qualitative reasoning effort tier.
    :param expected_params: Sampling parameters expected in the request.
    :param absent_params: Sampling parameters that must not be sent.
    """
    mock_create = AsyncMock()

    mock_resp = MagicMock()
    mock_resp.model = DEFAULT_CLAUDE_LLM
    mock_resp.stop_reason = "end_turn"
    mock_resp.usage.input_tokens = 300
    mock_resp.usage.output_tokens = 200

    mock_tool = MagicMock()
    mock_tool.type = "tool_use"
    mock_tool.name = "EstimationResult"
    mock_tool.input = VALID_RESULT_DICT
    mock_resp.content = [mock_tool]
    mock_create.return_value = mock_resp
    mock_async_anthropic_client.messages.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_anthropic_client",
        lambda: mock_async_anthropic_client,
    )

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

    assert isinstance(result["result"], EstimationResult)
    assert result["result"].total_cost_eur == 20000
    assert result["finish_reason"] == "end_turn"
    assert result["usage"]["total_tokens"] == 500

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 4000
    assert call_kwargs["messages"] == [{"role": "user", "content": sample_transcription}]
    assert call_kwargs["system"] == "CAG instructions"
    assert {k: call_kwargs.get(k) for k in expected_params} == expected_params
    assert all(k not in call_kwargs for k in absent_params)


@pytest.mark.anyio
async def test_call_anthropic_error_unwrapping(
    sample_transcription: str,
    mock_async_anthropic_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an Anthropic call failing with an error wrapped in InstructorRetryException
    WHEN call_anthropic executes
    THEN LLMProviderError is raised containing the unwrapped root error message.

    :param sample_transcription: Sample transcript fixture.
    :param mock_async_anthropic_client: Reusable typed AsyncAnthropic mock fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_create = AsyncMock()
    inner_err = RuntimeError("Anthropic upstream network failure")
    retry_exc = InstructorRetryException(
        "Retry failed",
        last_completion=None,
        messages=[],
        n_attempts=1,
        total_usage=None,
    )
    retry_exc.__cause__ = inner_err
    mock_create.side_effect = retry_exc
    mock_async_anthropic_client.messages.create = mock_create

    monkeypatch.setattr(
        "aie_estimator.services.providers.native_providers.get_anthropic_client",
        lambda: mock_async_anthropic_client,
    )

    opts = GenerationOptions(model=DEFAULT_CLAUDE_LLM, max_tokens=1000)
    with pytest.raises(LLMProviderError, match=r"Anthropic upstream network failure"):
        await call_anthropic(
            user_prompt=sample_transcription,
            system_prompt=None,
            opts=opts,
        )
