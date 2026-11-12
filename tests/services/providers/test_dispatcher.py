# tests/services/test_dispatcher.py

# --- Python Standard Library ---
from typing import Any, Final
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResult
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers.dispatcher import call_llm
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.sessions.models import MetadataExtractionResult
from aie_estimator.settings import get_settings
from aie_estimator.utils.model_constants import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
)

DUMMY_RESULT: Final[dict[str, Any]] = {
    "summary": "Full-stack web application delivery breakdown.",
    "confidence_pct": 85,
    "phases": [
        {
            "name": "Design & Architecture",
            "duration_weeks": 2,
            "cost_eur": 5000,
            "summary": "Technical architecture specification and UX wireframes.",
        },
    ],
    "total_duration_weeks": 2,
    "total_cost_eur": 5000,
}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("engine", "model", "target"),
    [
        ("litellm", DEFAULT_FAST_PROFILE, "call_litellm_router"),
        ("native", DEFAULT_GPT_LLM, "call_openai"),
        ("native", DEFAULT_CLAUDE_LLM, "call_anthropic"),
    ],
    ids=["litellm_router", "native_openai", "native_anthropic"],
)
async def test_call_llm_routes_to_engine_target(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    model: str,
    target: str,
) -> None:
    """
    GIVEN an execution engine and a model mapped to a provider
    WHEN call_llm executes
    THEN the matching provider function is invoked once with the provided options.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param engine: Execution engine ('litellm' or 'native').
    :param model: Target model identifier.
    :param target: Name of the provider function expected to be called.
    """
    mock_call = AsyncMock(return_value={"result": EstimationResult(**DUMMY_RESULT), "model": model})
    monkeypatch.setattr(f"aie_estimator.services.providers.dispatcher.{target}", mock_call)

    opts = GenerationOptions(engine=engine, model=model)
    result = await call_llm(user_prompt="User input", system_prompt="System instructions", opts=opts)

    assert isinstance(result["result"], EstimationResult)
    assert result["result"].total_cost_eur == 5000
    mock_call.assert_awaited_once_with(
        user_prompt="User input",
        system_prompt="System instructions",
        opts=opts,
        messages=None,
        response_model=EstimationResult,
    )


@pytest.mark.anyio
async def test_call_llm_rejects_unsupported_native_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an execution engine set to native with an unknown or unsupported provider
    WHEN call_llm executes
    THEN an LLMProviderError is raised.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    opts = GenerationOptions(engine="native", model="unknown-unsupported-model")
    with pytest.raises(LLMProviderError, match=r"Unsupported provider"):
        await call_llm(user_prompt="User input", system_prompt="System instructions", opts=opts)


def test_generation_options_defaults() -> None:
    """
    GIVEN no explicit constructor arguments
    WHEN GenerationOptions is instantiated
    THEN model and reasoning effort are unset, engine is litellm, and effective_model resolves to settings.llm_model.
    """
    opts = GenerationOptions()
    assert opts.model is None
    assert opts.reasoning_effort is None
    assert opts.engine == "litellm"
    assert opts.effective_model == get_settings().llm_model


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("engine", "model", "target"),
    [
        ("litellm", DEFAULT_FAST_PROFILE, "call_litellm_router"),
        ("native", DEFAULT_GPT_LLM, "call_openai"),
        ("native", DEFAULT_CLAUDE_LLM, "call_anthropic"),
    ],
    ids=["litellm_router", "native_openai", "native_anthropic"],
)
async def test_call_llm_multi_turn_forwards_messages(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    model: str,
    target: str,
) -> None:
    """
    GIVEN multi-turn messages with user_prompt=None and a system_prompt
    WHEN call_llm executes
    THEN the matching provider function receives user_prompt=None, system_prompt, opts, and messages.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param engine: Execution engine ('litellm' or 'native').
    :param model: Target model identifier.
    :param target: Name of the provider function expected to be called.
    """
    mock_call = AsyncMock(return_value={"result": EstimationResult(**DUMMY_RESULT), "model": model})
    monkeypatch.setattr(f"aie_estimator.services.providers.dispatcher.{target}", mock_call)

    turns = [
        {"role": "user", "content": "Initial SaaS"},
        {"role": "assistant", "content": '{"total_cost_eur": 5000}'},
        {"role": "user", "content": "Add OAuth2"},
    ]
    opts = GenerationOptions(engine=engine, model=model)
    result = await call_llm(
        user_prompt=None,
        system_prompt="Conversational system",
        opts=opts,
        messages=turns,
    )

    assert isinstance(result["result"], EstimationResult)
    mock_call.assert_awaited_once_with(
        user_prompt=None,
        system_prompt="Conversational system",
        opts=opts,
        messages=turns,
        response_model=EstimationResult,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("engine", "model", "target"),
    [
        ("litellm", DEFAULT_FAST_PROFILE, "call_litellm_router"),
        ("native", DEFAULT_GPT_LLM, "call_openai"),
        ("native", DEFAULT_CLAUDE_LLM, "call_anthropic"),
    ],
    ids=["litellm_router", "native_openai", "native_anthropic"],
)
async def test_call_llm_forwards_custom_response_model(
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    model: str,
    target: str,
) -> None:
    """
    GIVEN a custom Pydantic response_model passed to call_llm
    WHEN call_llm dispatches across litellm or native runners
    THEN the matching provider runner receives response_model=MetadataExtractionResult.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param engine: Execution engine ('litellm' or 'native').
    :param model: Target model identifier.
    :param target: Name of the provider function expected to be called.
    """
    extracted = MetadataExtractionResult(project_name="Atlas")
    mock_call = AsyncMock(return_value={"result": extracted, "model": model})
    monkeypatch.setattr(f"aie_estimator.services.providers.dispatcher.{target}", mock_call)

    opts = GenerationOptions(engine=engine, model=model)
    result = await call_llm(
        user_prompt="Extract facts",
        system_prompt="Extractor system",
        opts=opts,
        response_model=MetadataExtractionResult,
    )

    assert isinstance(result["result"], MetadataExtractionResult)
    mock_call.assert_awaited_once_with(
        user_prompt="Extract facts",
        system_prompt="Extractor system",
        opts=opts,
        messages=None,
        response_model=MetadataExtractionResult,
    )
