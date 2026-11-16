# tests/schemas/test_estimations.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
    TokenUsage,
)
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.settings import get_settings
from aie_estimator.utils.model_constants import (
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
)


@pytest.mark.parametrize(
    "invalid_model",
    [
        "string",
        "invalid_model",
        "llama-3",
        "gemini-2.5-flash",
        "gpt-5-nano",
    ],
    ids=[
        "swagger_placeholder_string",
        "arbitrary_invalid",
        "unsupported_llama",
        "unsupported_gemini",
        "unsupported_gpt5_nano",
    ],
)
def test_estimation_request_invalid_model_raises_validation_error(invalid_model: str) -> None:
    """
    GIVEN an invalid or unsupported model identifier
    WHEN EstimationRequest is instantiated
    THEN a pydantic ValidationError is raised matching the expected input keywords.

    :param invalid_model: Unsupported model identifier.
    """
    valid_params: dict[str, Any] = {
        "description": "Client requires a scalable notification service supporting email and SMS.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.PHASES_TABLE,
        "model": invalid_model,
    }
    with pytest.raises(ValidationError, match=r"Input should be"):
        EstimationRequest(**valid_params)


VALID_REQUEST_FIELDS: dict[str, Any] = {
    "description": "Client requires a scalable notification service supporting email and SMS.",
    "project_type": ProjectType.WEB_SAAS,
    "detail_level": DetailLevel.MEDIUM,
    "output_format": OutputFormat.PHASES_TABLE,
}


@pytest.mark.parametrize(
    ("overrides", "missing", "expected_error"),
    [
        ({"project_type": "invalid_type"}, None, "Input should be"),
        ({"detail_level": "invalid_level"}, None, "Input should be"),
        ({"output_format": "invalid_format"}, None, "Input should be"),
        ({"engine": "invalid_engine"}, None, "Input should be 'litellm' or 'native'"),
        ({}, "description", "Field required"),
        ({}, "project_type", "Field required"),
    ],
    ids=[
        "invalid_project_type",
        "invalid_detail_level",
        "invalid_output_format",
        "invalid_engine",
        "missing_description",
        "missing_project_type",
    ],
)
def test_estimation_request_invalid_or_missing_field_raises_validation_error(
    overrides: dict[str, Any], missing: str | None, expected_error: str
) -> None:
    """
    GIVEN an invalid enum value or a missing required field
    WHEN EstimationRequest is instantiated
    THEN a pydantic ValidationError is raised with the expected message.

    :param overrides: Field values replacing the valid ones.
    :param missing: Required field removed from the payload, if any.
    :param expected_error: Regex expected in the error message.
    """
    params = {**VALID_REQUEST_FIELDS, **overrides}
    if missing:
        params.pop(missing)
    with pytest.raises(ValidationError, match=expected_error):
        EstimationRequest(**params)


def test_estimation_request_valid_model_accepted() -> None:
    """
    GIVEN a valid native model registered in the catalog paired with engine='native'
    WHEN EstimationRequest is instantiated
    THEN the instance is created with the expected model value.
    """
    req = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        engine="native",
        model=DEFAULT_GPT_LLM,
    )
    assert req.model == DEFAULT_GPT_LLM


def test_estimation_request_native_omitted_model_defaults_to_first_native_model() -> None:
    """
    GIVEN an EstimationRequest with engine='native' and omitted model
    WHEN the instance is validated
    THEN model resolves to the first native model in the catalog.
    """
    req = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        engine="native",
    )
    assert req.model == DEFAULT_GPT_LLM


@pytest.mark.parametrize(
    ("engine", "model"),
    [
        ("litellm", DEFAULT_GPT_LLM),
        ("native", DEFAULT_FAST_PROFILE),
    ],
    ids=["litellm_with_physical_model", "native_with_logical_profile"],
)
def test_estimation_request_rejects_engine_model_mismatch(engine: str, model: str) -> None:
    """
    GIVEN an EstimationRequest pairing an incompatible engine and model
    WHEN the request is validated
    THEN a ValidationError is raised indicating the model is unsupported for that engine.

    :param engine: Target execution engine.
    :param model: Incompatible model or profile identifier.
    """
    with pytest.raises(ValidationError, match=r"is not supported for engine"):
        EstimationRequest(
            description="Client requires a scalable notification service supporting email and SMS.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            engine=engine,  # type: ignore[arg-type]
            model=model,
        )


def test_estimation_request_none_model_defaults_to_settings_llm_model() -> None:
    """
    GIVEN an EstimationRequest instantiated without a model or with model=None
    WHEN the instance is created
    THEN the model attribute defaults to settings.llm_model.
    """
    req_omitted = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    req_none = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        model=None,
    )
    expected_model = get_settings().llm_model
    assert req_omitted.model == expected_model
    assert req_none.model == expected_model


@pytest.mark.parametrize(
    "profile_model",
    [
        DEFAULT_FAST_PROFILE,
        DEFAULT_DEEP_PROFILE,
    ],
    ids=["req_reasoner_fast", "req_reasoner_deep"],
)
def test_estimation_request_accepts_logical_profiles(profile_model: str) -> None:
    """
    GIVEN an EstimationRequest with a logical profile as the model
    WHEN the request is validated
    THEN it is successfully instantiated with the requested profile.

    :param profile_model: Logical profile identifier.
    """
    req = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        model=profile_model,
    )
    assert req.model == profile_model


def test_estimation_request_description_length_boundaries() -> None:
    """
    GIVEN descriptions violating min (20) or max (2000) character constraints
    WHEN EstimationRequest is instantiated
    THEN a pydantic ValidationError is raised.
    """
    with pytest.raises(ValidationError, match=r"at least 20 characters"):
        EstimationRequest(
            description="Short desc",
            project_type=ProjectType.MOBILE_APP,
            detail_level=DetailLevel.SUMMARY,
            output_format=OutputFormat.LINE_ITEMS,
        )

    with pytest.raises(ValidationError, match=r"at most 2000 characters"):
        EstimationRequest(
            description="a" * 2001,
            project_type=ProjectType.MOBILE_APP,
            detail_level=DetailLevel.SUMMARY,
            output_format=OutputFormat.LINE_ITEMS,
        )


@pytest.mark.parametrize(
    ("phase_kwargs", "expected_match"),
    [
        ({"name": "", "duration_weeks": 2, "cost_eur": 1000, "summary": "Valid summary description."}, r"at least 1 character"),
        ({"name": "P1", "duration_weeks": 0, "cost_eur": 1000, "summary": "Valid summary description."}, r"greater than or equal to 1"),
        ({"name": "P1", "duration_weeks": 53, "cost_eur": 1000, "summary": "Valid summary description."}, r"less than or equal to 52"),
        ({"name": "P1", "duration_weeks": 2, "cost_eur": -1, "summary": "Valid summary description."}, r"greater than or equal to 0"),
        ({"name": "P1", "duration_weeks": 2, "cost_eur": 1000, "summary": "Short"}, r"at least 10 characters"),
    ],
    ids=[
        "empty_name",
        "zero_duration",
        "duration_exceeds_max",
        "negative_cost",
        "summary_too_short",
    ],
)
def test_phase_field_constraints_raise_validation_error(phase_kwargs: dict[str, Any], expected_match: str) -> None:
    """
    GIVEN invalid Phase attribute values violating schema constraints
    WHEN Phase is instantiated
    THEN a pydantic ValidationError is raised.

    :param phase_kwargs: Arguments for Phase instantiation.
    :param expected_match: Expected regex match in the validation error.
    """
    with pytest.raises(ValidationError, match=expected_match):
        Phase(**phase_kwargs)


def test_estimation_response_serialization() -> None:
    """
    GIVEN a valid structured PmEstimation, tier metadata, and telemetry
    WHEN EstimationResponse is instantiated
    THEN attributes are serialized accurately into dict and JSON.
    """
    result = PmEstimation(
        summary="Complete inventory SaaS delivery breakdown.",
        confidence_pct=85,
        phases=[
            Phase(
                name="Phase 1: Architecture",
                duration_weeks=2,
                cost_eur=5000,
                summary="System architecture design and environment setup.",
            ),
            Phase(
                name="Phase 2: Build",
                duration_weeks=6,
                cost_eur=15000,
                summary="Core API and UI implementation with automated tests.",
            ),
        ],
        team_composition=TeamComposition(developers=2, designers=1),
        total_duration_weeks=8,
        total_cost_eur=20000,
    )
    resp = EstimationResponse(
        result=result,
        tier=Tier.PM,
        tier_rule="default",
        prompt_version="v3",
        cached=False,
        model=DEFAULT_GPT_LLM,
        provider="openai",
        cost_usd=0.0012,
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    data = resp.model_dump()
    assert data["result"]["total_cost_eur"] == 20000
    assert data["result"]["phases"][0]["name"] == "Phase 1: Architecture"
    assert data["tier"] == "pm"
    assert data["tier_rule"] == "default"
    assert data["prompt_version"] == "v3"
    assert data["cached"] is False
    assert data["model"] == DEFAULT_GPT_LLM
    assert data["provider"] == "openai"
    assert data["cost_usd"] == 0.0012
    assert data["usage"]["total_tokens"] == 150

    from_service = EstimationResponse.from_service_result(
        {
            "result": result,
            "tier": Tier.PM,
            "tier_rule": "default",
            "cache_hit": True,
            "prompt_version": "v3",
            "model": DEFAULT_GPT_LLM,
            "provider": "openai",
            "cost_usd": 0.0012,
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
    )
    assert from_service.cached is True
    assert from_service.prompt_version == "v3"
    assert from_service.result.total_cost_eur == 20000

