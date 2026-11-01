# tests/services/test_llm_service.py

# --- Python Standard Library ---
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.schemas.models import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
)
from aie_estimator.services import LLMServiceError, llm_service
from aie_estimator.services.llm_service import GenerationOptions
from aie_estimator.settings import Environment, Settings


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


@pytest.mark.anyio
async def test_generate_estimation_two_phase_preprocessing(
    sample_transcription: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an estimation request configured with two_phase preprocessing
    WHEN generate_estimation executes
    THEN extract_requirements executes phase 1 before the estimation generation phase.

    :param sample_transcription: Sample transcript fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-key"),
        llm_model=DEFAULT_GPT_LLM,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_extract_resp = make_estimation_result(
        estimation="- Functional requirement: Webhooks\n- Non-functional requirement: High availability",
        usage={"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
    )
    mock_final_resp = make_estimation_result(
        estimation="## Project summary\nFinal estimation breakdown.",
        usage={"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
    )

    call_count = 0

    async def mock_call_openai(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return mock_extract_resp if call_count == 1 else mock_final_resp

    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_call_openai)

    opts = GenerationOptions(preprocessing="two_phase", engine="native")
    result = await llm_service.generate_estimation(sample_transcription, opts)

    assert result["preprocessing"] == "two_phase"
    assert "Webhooks" in str(result["extracted_requirements"])
    assert result["usage"]["preprocessing_input_tokens"] == 80
    assert result["usage"]["preprocessing_output_tokens"] == 40
    assert call_count == 2


@pytest.mark.anyio
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
async def test_generate_estimation_model_driven_provider_dispatch(
    sample_transcription: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    model_override: str | None,
    default_provider: str,
    expected_called_provider: str,
) -> None:
    """
    GIVEN a model override and a default environment provider
    WHEN generate_estimation is executed
    THEN the call is dispatched to the provider matching the model's catalog specification.

    :param sample_transcription: Sample transcript fixture.
    :param make_estimation_result: Estimation result factory fixture.
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
        return_value=make_estimation_result(
            estimation="OpenAI estimate",
            model=model_override or DEFAULT_GPT_LLM,
            provider="openai",
            usage={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        )
    )
    mock_anthropic = AsyncMock(
        return_value=make_estimation_result(
            estimation="Anthropic estimate",
            model=model_override or DEFAULT_GPT_LLM,
            provider="anthropic",
            finish_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
        )
    )

    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service.call_anthropic", mock_anthropic)

    opts = GenerationOptions(model=model_override, engine="native")
    result = await llm_service.generate_estimation(sample_transcription, opts)

    if expected_called_provider == "openai":
        assert mock_openai.call_count == 1
        assert mock_anthropic.call_count == 0
        assert result["provider"] == "openai"
    else:
        assert mock_anthropic.call_count == 1
        assert mock_openai.call_count == 0
        assert result["provider"] == "anthropic"


@pytest.mark.anyio
async def test_generate_estimation_cache_hit_bypasses_provider(
    sample_transcription: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a cached estimation entry in EstimationCache
    WHEN generate_estimation is executed with matching parameters
    THEN the cached result is returned with cache_hit=True without calling any provider.

    :param sample_transcription: Sample transcript fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        llm_model=DEFAULT_GPT_LLM,
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.get_settings", lambda: test_settings)

    mock_cache = AsyncMock()
    cached_payload: dict[str, Any] = make_estimation_result(
        estimation="Cached software estimation breakdown.",
        usage={"input_tokens": 50, "output_tokens": 50, "total_tokens": 100},
    )
    mock_cache.get.return_value = cached_payload
    monkeypatch.setattr("aie_estimator.services.llm_service.get_estimation_cache", lambda: mock_cache)

    mock_openai = AsyncMock()
    mock_anthropic = AsyncMock()
    monkeypatch.setattr("aie_estimator.services.llm_service.call_openai", mock_openai)
    monkeypatch.setattr("aie_estimator.services.llm_service.call_anthropic", mock_anthropic)

    result = await llm_service.generate_estimation(sample_transcription)

    assert result["cache_hit"] is True
    assert result["estimation"] == "Cached software estimation breakdown."
    mock_openai.assert_not_called()
    mock_anthropic.assert_not_called()


@pytest.mark.anyio
async def test_generate_estimation_litellm_engine_routes_to_router(
    sample_transcription: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN engine='litellm' in GenerationOptions
    WHEN generate_estimation is executed
    THEN the call dispatches through call_litellm_router and extracts cost_usd.

    :param sample_transcription: Sample transcript fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_router_call = AsyncMock(
        return_value=make_estimation_result(
            estimation="Router generated estimation breakdown.",
            model=DEFAULT_FAST_PROFILE,
            provider="litellm",
            usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            cost_usd=0.00045,
        )
    )
    monkeypatch.setattr("aie_estimator.services.llm_service.call_litellm_router", mock_router_call)

    opts = GenerationOptions(engine="litellm", model=DEFAULT_FAST_PROFILE)
    result = await llm_service.generate_estimation(sample_transcription, opts)

    mock_router_call.assert_called_once()
    assert result["estimation"] == "Router generated estimation breakdown."
    assert result["cost_usd"] == 0.00045
    assert result["provider"] == "litellm"


@pytest.mark.anyio
async def test_generate_estimation_native_engine_rejects_unsupported_provider(
    sample_transcription: str,
) -> None:
    """
    GIVEN engine='native' with a logical profile or unsupported model
    WHEN generate_estimation is executed
    THEN LLMServiceError is raised indicating the unsupported provider.

    :param sample_transcription: Sample transcript fixture.
    """
    opts = GenerationOptions(engine="native", model=DEFAULT_FAST_PROFILE)
    with pytest.raises(LLMServiceError, match=r"Unsupported provider"):
        await llm_service.generate_estimation(sample_transcription, opts)
