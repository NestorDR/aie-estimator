# tests/services/test_litellm_provider.py

# --- Python Standard Library ---
from collections.abc import Callable
import json
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from instructor.core import InstructorRetryException
import litellm
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResult
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.services.providers.litellm_provider import call_litellm_router, get_litellm_router
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM

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
    THEN LLMProviderError is raised indicating no valid credentials.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY", "SCALEWAY_API_KEY", "NEBIUS_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    get_litellm_router.cache_clear()
    with pytest.raises(LLMProviderError, match=r"No models in Router have valid credentials"):
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


@pytest.fixture
def mock_router_response(monkeypatch: pytest.MonkeyPatch) -> Callable[..., MagicMock]:
    """
    Provide a builder that patches get_litellm_router with a mock Router returning a canned completion.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Builder returning the mock router so tests can inspect acompletion calls.
    """

    def _build(
        content: str | None = None,
        hidden_params: dict[str, Any] | None = None,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> MagicMock:
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = content if content is not None else json.dumps(VALID_RESULT_DICT)
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens
        mock_response.usage.total_tokens = prompt_tokens + completion_tokens
        mock_response.model = f"openai/{DEFAULT_GPT_LLM}"
        mock_response._hidden_params = hidden_params or {}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_response)
        monkeypatch.setattr("aie_estimator.services.providers.litellm_provider.get_litellm_router", lambda: mock_router)
        return mock_router

    return _build


@pytest.mark.anyio
async def test_call_router_cost_extraction_and_response_structure(
    mock_router_response: Callable[..., MagicMock],
) -> None:
    """
    GIVEN a mocked LiteLLM Router completion response containing hidden cost params and structured JSON
    WHEN call_litellm_router is executed
    THEN structured EstimationResult, metadata, token counts, and cost_usd are extracted accurately.

    :param mock_router_response: Router response builder fixture.
    """
    mock_router_response(
        hidden_params={"response_cost": 0.00035, "custom_llm_provider": "openai"},
        prompt_tokens=250,
        completion_tokens=75,
    )

    opts = GenerationOptions(model=DEFAULT_FAST_PROFILE, max_tokens=2000)
    result = await call_litellm_router(
        user_prompt="Analyze this project transcript",
        system_prompt="You are an estimator",
        opts=opts,
    )

    assert isinstance(result["result"], EstimationResult)
    assert result["result"].total_cost_eur == 20000
    assert result["result"].confidence_pct == 85
    assert len(result["result"].phases) == 2
    assert result["model"] == f"openai/{DEFAULT_GPT_LLM}"
    assert result["provider"] == "litellm"
    assert result["finish_reason"] == "stop"
    assert result["cost_usd"] == 0.00035
    assert result["usage"]["input_tokens"] == 250
    assert result["usage"]["output_tokens"] == 75
    assert result["usage"]["total_tokens"] == 325


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reasoning_effort",
    [None, "medium"],
    ids=["router_reasoning_disabled", "router_medium"],
)
async def test_call_router_dispatches_reasoning_effort(
    mock_router_response: Callable[..., MagicMock],
    reasoning_effort: str | None,
) -> None:
    """
    GIVEN qualitative reasoning effort tier
    WHEN call_litellm_router executes
    THEN reasoning_effort is forwarded directly to router.acompletion.

    :param mock_router_response: Router response builder fixture.
    :param reasoning_effort: Explicit qualitative reasoning effort tier.
    """
    mock_router = mock_router_response(hidden_params={})

    opts = GenerationOptions(
        model=DEFAULT_FAST_PROFILE,
        max_tokens=2000,
        reasoning_effort=reasoning_effort,
    )
    result = await call_litellm_router(
        user_prompt="Analyze project",
        system_prompt="System prompt",
        opts=opts,
    )

    assert isinstance(result["result"], EstimationResult)
    call_kwargs = mock_router.acompletion.call_args.kwargs
    assert call_kwargs.get("reasoning_effort") == reasoning_effort


@pytest.mark.anyio
async def test_call_router_error_unwrapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a LiteLLM Router failure wrapped in InstructorRetryException
    WHEN call_litellm_router executes
    THEN LLMProviderError is raised containing the unwrapped root error message.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    inner_err = RuntimeError("LiteLLM all candidate models exhausted")
    retry_exc = InstructorRetryException(
        "Retry failed",
        last_completion=None,
        messages=[],
        n_attempts=1,
        total_usage=None,
    )
    retry_exc.__cause__ = inner_err

    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(side_effect=retry_exc)
    monkeypatch.setattr("aie_estimator.services.providers.litellm_provider.get_litellm_router", lambda: mock_router)

    opts = GenerationOptions(model=DEFAULT_FAST_PROFILE, max_tokens=1000)
    with pytest.raises(LLMProviderError, match=r"LiteLLM all candidate models exhausted"):
        await call_litellm_router(
            user_prompt="Analyze project",
            system_prompt=None,
            opts=opts,
        )


@pytest.mark.anyio
async def test_call_router_multi_turn_prepends_system_prompt(
    mock_router_response: Callable[..., MagicMock],
) -> None:
    """
    GIVEN a multi-turn conversation sequence and a separate system_prompt
    WHEN call_litellm_router is invoked with user_prompt=None and messages
    THEN router.acompletion receives the system message followed by all conversation turns.

    :param mock_router_response: Router response builder fixture.
    """
    mock_router = mock_router_response(hidden_params={})
    turns = [
        {"role": "user", "content": "Initial project scope"},
        {"role": "assistant", "content": '{"total_cost_eur": 20000}'},
        {"role": "user", "content": "Add OAuth2 SSO"},
    ]

    opts = GenerationOptions(model=DEFAULT_FAST_PROFILE, max_tokens=2000)
    result = await call_litellm_router(
        user_prompt=None,
        system_prompt="Conversational estimator system prompt",
        opts=opts,
        messages=turns,
    )

    assert isinstance(result["result"], EstimationResult)
    sent_messages = mock_router.acompletion.call_args.kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert sent_messages[0]["content"].startswith("Conversational estimator system prompt")
    assert sent_messages[1:] == turns

