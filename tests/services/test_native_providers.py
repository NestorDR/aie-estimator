# tests/services/test_native_providers.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from openai import BadRequestError, RateLimitError
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.schemas.models import DEFAULT_CLAUDE_LLM, DEFAULT_GPT_LLM
from aie_estimator.services import LLMServiceError
from aie_estimator.services.native_providers import (
    call_anthropic,
    call_openai,
    get_anthropic_client,
    get_openai_client,
)
from aie_estimator.settings import Environment, Settings
from tests.conftest import REASONING_EFFORT_TEST_CASES


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
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    is_reasoning: bool,
) -> None:
    """
    GIVEN a target model identifier
    WHEN call_openai executes against the Responses API
    THEN reasoning models receive reasoning effort while non-reasoning models receive temperature.

    :param sample_transcription: Sample transcript fixture.
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
        messages=sample_transcription,
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
    assert call_kwargs["input"] == sample_transcription
    assert call_kwargs["instructions"] == "You are a CAG estimator."
    assert call_kwargs["max_output_tokens"] == 3000

    if is_reasoning:
        assert call_kwargs.get("reasoning") == {"effort": "medium"}
        assert "temperature" not in call_kwargs
    else:
        assert call_kwargs.get("temperature") == 0.3
        assert "reasoning" not in call_kwargs


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

    monkeypatch.setattr("aie_estimator.services.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    result = await call_openai(
        messages=sample_transcription,
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
async def test_call_openai_retry_exhaustion(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN persistent transient rate limits from OpenAI
    WHEN retry attempts exceed the maximum configured count
    THEN LLMServiceError is raised with logged diagnostic context.

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

    monkeypatch.setattr("aie_estimator.services.native_providers.get_openai_client", lambda: mock_client)
    monkeypatch.setattr("asyncio.sleep", mock_sleep)

    with pytest.raises(LLMServiceError, match=r"OpenAI call failed"):
        await call_openai(
            messages=sample_transcription,
            system_prompt=None,
            model=DEFAULT_GPT_LLM,
            max_tokens=1000,
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    REASONING_EFFORT_TEST_CASES,
    ids=[
        "thinking_disabled",
        "effort_low",
        "effort_medium",
        "effort_high",
    ],
)
async def test_call_anthropic_thinking_and_temperature_exclusivity(
    sample_transcription: str,
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort: str | None,
    expected_effort: str | None,
) -> None:
    """
    GIVEN Anthropic configuration with or without qualitative reasoning effort
    WHEN call_anthropic executes
    THEN adaptive thinking and temperature are strictly mutually exclusive.

    :param sample_transcription: Sample transcript fixture.
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
        messages=sample_transcription,
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
    assert call_kwargs["messages"] == [{"role": "user", "content": sample_transcription}]
    assert call_kwargs["system"] == "CAG instructions"

    if expected_effort is not None:
        assert call_kwargs.get("thinking") == {"type": "adaptive"}
        assert call_kwargs.get("output_config") == {"effort": expected_effort}
        assert "temperature" not in call_kwargs
    else:
        assert call_kwargs.get("temperature") == 0.3
        assert "thinking" not in call_kwargs
        assert "output_config" not in call_kwargs
