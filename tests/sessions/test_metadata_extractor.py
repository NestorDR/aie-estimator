# tests/sessions/test_metadata_extractor.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResult, Phase
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers import GenerationOptions
from aie_estimator.sessions.metadata_extractor import (
    MetadataExtractionOutcome,
    resolve_extractor_options,
    sanitize_extraction_delta,
    update_metadata,
)
from aie_estimator.sessions.models import MetadataExtractionResult, ProjectMetadata
from aie_estimator.settings import Environment, Settings
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM


def _sample_estimation(*, confidence_pct: int = 85) -> EstimationResult:
    """Build a valid EstimationResult for metadata extractor unit tests."""
    if confidence_pct < 30:
        return EstimationResult(
            summary="Out of scope: Cannot size unrelated request.",
            confidence_pct=confidence_pct,
            phases=[Phase(name="Not estimated", duration_weeks=1, cost_eur=0, summary="Insufficient project scope.")],
            total_duration_weeks=1,
            total_cost_eur=0,
        )
    return EstimationResult(
        summary="MVP delivery estimate for Atlas portal.",
        confidence_pct=confidence_pct,
        phases=[Phase(name="Implementation", duration_weeks=4, cost_eur=10_000, summary="Core backend and UI.")],
        total_duration_weeks=4,
        total_cost_eur=10_000,
    )


@pytest.mark.anyio
async def test_update_metadata_initial_population_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an empty ProjectMetadata and a valid estimation turn
    WHEN update_metadata invokes the extractor LLM
    THEN initial fields, technologies, constraints, tokens, and cost are returned.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    delta = MetadataExtractionResult(
        project_name="Atlas",
        assumed_team_size=4,
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        agreed_scope="Phase 1 analytics dashboard",
        explicit_constraints=["SOC 2 Type II"],
    )
    mock_call = AsyncMock(
        return_value={
            "result": delta,
            "model": "gpt-6-luna",
            "usage": {"input_tokens": 220, "output_tokens": 55, "total_tokens": 275},
            "cost_usd": 0.00018,
        }
    )
    monkeypatch.setattr("aie_estimator.sessions.metadata_extractor.call_llm", mock_call)

    outcome = await update_metadata(
        previous=ProjectMetadata(),
        transcript="Build Atlas with 4 devs using FastAPI and PostgreSQL under SOC 2.",
        result=_sample_estimation(),
        opts=GenerationOptions(),
    )

    assert isinstance(outcome, MetadataExtractionOutcome)
    assert outcome.metadata == ProjectMetadata(
        project_name="Atlas",
        assumed_team_size=4,
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        agreed_scope="Phase 1 analytics dashboard",
        explicit_constraints=["SOC 2 Type II"],
    )
    assert outcome.input_tokens == 220
    assert outcome.output_tokens == 55
    assert outcome.cost_usd == 0.00018
    assert mock_call.await_args is not None
    assert mock_call.await_args.kwargs["response_model"] is MetadataExtractionResult


@pytest.mark.anyio
async def test_update_metadata_cumulative_and_explicit_policy1_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN existing ProjectMetadata and a turn retracting Rails in favor of Node
    WHEN update_metadata merges the returned delta
    THEN prior unaffected fields persist while Rails moves from technologies to rejected_options.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    previous = ProjectMetadata(
        project_name="Atlas",
        assumed_team_size=4,
        mentioned_technologies=["Rails", "PostgreSQL"],
        agreed_scope="Phase 1 analytics dashboard",
        explicit_constraints=["GDPR", "SOC 2 Type II"],
    )
    delta = MetadataExtractionResult(
        mentioned_technologies=["Node"],
        rejected_options=["Rails"],
        retracted_constraints=["gdpr"],
    )
    monkeypatch.setattr(
        "aie_estimator.sessions.metadata_extractor.call_llm",
        AsyncMock(
            return_value={
                "result": delta,
                "model": DEFAULT_FAST_PROFILE,
                "usage": {"input_tokens": 180, "output_tokens": 40},
                "cost_usd": None,
            }
        ),
    )

    outcome = await update_metadata(
        previous=previous,
        transcript="Drop Rails and GDPR; use Node instead.",
        result=_sample_estimation(),
        opts=GenerationOptions(),
    )

    assert outcome.metadata == ProjectMetadata(
        project_name="Atlas",
        assumed_team_size=4,
        mentioned_technologies=["PostgreSQL", "Node"],
        agreed_scope="Phase 1 analytics dashboard",
        explicit_constraints=["SOC 2 Type II"],
        rejected_options=["Rails"],
    )


def test_sanitize_extraction_delta_normalizes_nulls_blanks_and_duplicates() -> None:
    """
    GIVEN a raw MetadataExtractionResult with surrounding whitespace, null literals, and duplicate list items
    WHEN sanitize_extraction_delta runs
    THEN blank/null strings clear to None and lists deduplicate case-insensitively.
    """
    raw = MetadataExtractionResult(
        project_name="  Atlas  ",
        assumed_team_size=4,
        mentioned_technologies=["  FastAPI ", "", "null", "fastapi", "Redis"],
        agreed_scope="  none  ",
        explicit_constraints=[" GDPR ", "gdpr", "N/A"],
        rejected_options=["  "],
        retracted_constraints=[" SOC 2 ", "soc 2"],
        retracted_options=[],
    )

    sanitized = sanitize_extraction_delta(raw)

    assert sanitized.project_name == "Atlas"
    assert sanitized.assumed_team_size == 4
    assert sanitized.agreed_scope is None
    assert sanitized.mentioned_technologies == ["FastAPI", "Redis"]
    assert sanitized.explicit_constraints == ["GDPR"]
    assert sanitized.rejected_options == []
    assert sanitized.retracted_constraints == ["SOC 2"]


@pytest.mark.parametrize(
    ("env_enabled", "confidence_pct"),
    [
        ("false", 85),
        ("true", 20),
    ],
    ids=["disabled_by_setting", "out_of_scope_low_confidence"],
)
@pytest.mark.anyio
async def test_update_metadata_skips_when_disabled_or_out_of_scope(
    monkeypatch: pytest.MonkeyPatch,
    env_enabled: str,
    confidence_pct: int,
) -> None:
    """
    GIVEN metadata extraction disabled in settings or an out-of-scope estimation result
    WHEN update_metadata runs
    THEN call_llm is not invoked and previous metadata is returned with zero tokens.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param env_enabled: String flag for ENABLE_METADATA_EXTRACTION.
    :param confidence_pct: Estimation confidence percentage.
    """
    monkeypatch.setenv("ENABLE_METADATA_EXTRACTION", env_enabled)
    mock_call = AsyncMock()
    monkeypatch.setattr("aie_estimator.sessions.metadata_extractor.call_llm", mock_call)

    previous = ProjectMetadata(project_name="Atlas")
    outcome = await update_metadata(
        previous=previous,
        transcript="Tell me a joke.",
        result=_sample_estimation(confidence_pct=confidence_pct),
        opts=GenerationOptions(),
    )

    mock_call.assert_not_awaited()
    assert outcome == MetadataExtractionOutcome(metadata=previous)


@pytest.mark.parametrize(
    ("opts", "expected_model", "expected_engine", "expected_effort"),
    [
        (GenerationOptions(engine="litellm", model="reasoner-deep"), DEFAULT_FAST_PROFILE, "litellm", "low"),
        (GenerationOptions(engine="native", model=DEFAULT_GPT_LLM), DEFAULT_GPT_LLM, "native", "low"),
        (
            GenerationOptions(engine="native", model="mistral/mistral-small-3.2-24b-instruct"),
            "mistral/mistral-small-3.2-24b-instruct",
            "native",
            None,
        ),
    ],
    ids=["litellm_uses_fast_profile", "native_falls_back_to_effective_model", "non_reasoning_clears_effort"],
)
def test_resolve_extractor_options_engine_and_model_routing(
    opts: GenerationOptions,
    expected_model: str,
    expected_engine: str,
    expected_effort: str | None,
) -> None:
    """
    GIVEN primary turn GenerationOptions and application Settings
    WHEN resolve_extractor_options constructs extractor options
    THEN litellm uses reasoner-fast, native falls back to effective_model, and reasoning_effort matches model capability.

    :param opts: Primary turn GenerationOptions.
    :param expected_model: Expected resolved extractor model.
    :param expected_engine: Expected extractor execution engine.
    :param expected_effort: Expected reasoning effort override.
    """
    settings = Settings(app_env=Environment.TEST)
    resolved = resolve_extractor_options(opts, settings)

    assert resolved.model == expected_model
    assert resolved.engine == expected_engine
    assert resolved.reasoning_effort == expected_effort
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 1000


@pytest.mark.parametrize(
    "mock_side_effect_or_return",
    [
        LLMProviderError("Upstream timeout"),
        {"result": "invalid-string-payload"},
    ],
    ids=["provider_error", "unexpected_result_type"],
)
@pytest.mark.anyio
async def test_update_metadata_failure_returns_previous_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    mock_side_effect_or_return: Any,
) -> None:
    """
    GIVEN an extractor LLM error or malformed response payload
    WHEN update_metadata executes
    THEN prior ProjectMetadata is preserved unchanged with zero tokens and no exception raised.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param mock_side_effect_or_return: Exception instance or malformed response payload for call_llm.
    """
    if isinstance(mock_side_effect_or_return, Exception):
        mock_call = AsyncMock(side_effect=mock_side_effect_or_return)
    else:
        mock_call = AsyncMock(return_value=mock_side_effect_or_return)
    monkeypatch.setattr("aie_estimator.sessions.metadata_extractor.call_llm", mock_call)

    previous = ProjectMetadata(project_name="Atlas", mentioned_technologies=["PostgreSQL"])
    outcome = await update_metadata(
        previous=previous,
        transcript="Add Redis caching.",
        result=_sample_estimation(),
        opts=GenerationOptions(),
    )

    assert outcome == MetadataExtractionOutcome(metadata=previous)

