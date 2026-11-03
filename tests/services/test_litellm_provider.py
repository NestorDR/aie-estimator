# tests/services/test_litellm_provider.py

# --- Python Standard Library ---
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
import litellm
import pytest

# --- App modules ---
from aie_estimator.schemas.models import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM
from aie_estimator.services import LLMServiceError
from aie_estimator.services.litellm_provider import call_litellm_router, get_litellm_router
from tests.conftest import REASONING_EFFORT_TEST_CASES


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


@pytest.mark.anyio
async def test_call_router_cost_extraction_and_response_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = await call_litellm_router(
        messages="Analyze this project transcript",
        system_prompt="You are an estimator",
        model=DEFAULT_FAST_PROFILE,
        max_tokens=2000,
    )

    assert result["estimation"] == "Estimated Effort: 42 hours."
    assert result["model"] == f"openai/{DEFAULT_GPT_LLM}"
    assert result["provider"] == "litellm"
    assert result["finish_reason"] == "stop"
    assert result["cost_usd"] == 0.00035
    assert result["usage"]["input_tokens"] == 250
    assert result["usage"]["output_tokens"] == 75
    assert result["usage"]["total_tokens"] == 325


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    REASONING_EFFORT_TEST_CASES,
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


def test_get_litellm_router_reloads_on_cache_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an updated configuration file
    WHEN get_litellm_router.cache_clear() is called
    THEN a new Router instance is returned reflecting the updated configuration.

    :param tmp_path: Temporary directory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    config_file = tmp_path / "llm_profiles.yml"
    config_file.write_text(
        "general_settings:\n  disable_telemetry: true\nmodel_list:\n  - model_name: 'alpha'\n    litellm_params:\n      model: 'openai/gpt-6-luna'\n      api_key: 'os.environ/OPENAI_API_KEY'\n",
        encoding="utf-8",
    )

    with patch("aie_estimator.services.litellm_provider.get_profiles_config_path", return_value=config_file):
        get_litellm_router.cache_clear()
        router_v1 = get_litellm_router()
        models_v1 = [m["model_name"] for m in router_v1.model_list]
        assert "alpha" in models_v1
        assert "beta" not in models_v1

        config_file.write_text(
            "general_settings:\n  disable_telemetry: true\nmodel_list:\n  - model_name: 'alpha'\n    litellm_params:\n      model: 'openai/gpt-6-luna'\n      api_key: 'os.environ/OPENAI_API_KEY'\n  - model_name: 'beta'\n    litellm_params:\n      model: 'openai/gpt-6-luna'\n      api_key: 'os.environ/OPENAI_API_KEY'\n",
            encoding="utf-8",
        )

        get_litellm_router.cache_clear()
        router_v2 = get_litellm_router()
        models_v2 = [m["model_name"] for m in router_v2.model_list]
        assert "beta" in models_v2
    get_litellm_router.cache_clear()

