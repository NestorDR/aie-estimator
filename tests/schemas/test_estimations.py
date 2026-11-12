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
    EstimationResult,
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    OutputFormat,
    Phase,
    ProjectType,
    TokenUsage,
)
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
    GIVEN a valid model registered in the catalog
    WHEN EstimationRequest is instantiated
    THEN the instance is created with the expected model value.
    """
    req = EstimationRequest(
        description="Client requires a scalable notification service supporting email and SMS.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        model=DEFAULT_GPT_LLM,
    )
    assert req.model == DEFAULT_GPT_LLM


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


def test_estimation_result_field_ordering() -> None:
    """
    GIVEN the EstimationResult Pydantic model definition
    WHEN model field keys are inspected in declaration order
    THEN phases appears before total_duration_weeks and total_cost_eur for autoregressive generation.
    """
    field_names = list(EstimationResult.model_fields.keys())
    assert field_names == [
        "summary",
        "confidence_pct",
        "phases",
        "total_duration_weeks",
        "total_cost_eur",
    ]


@pytest.mark.parametrize(
    ("phases_costs", "total_cost", "should_pass"),
    [
        ([5000, 15000], 20000, True),
        ([4000, 8000], 10000, False),
        ([1000], 1000, True),
        ([1000, 2000, 3000], 6500, False),
    ],
    ids=[
        "costs_match_two_phases",
        "costs_mismatch_two_phases",
        "costs_match_single_phase",
        "costs_mismatch_three_phases",
    ],
)
def test_estimation_result_phases_sum_matches_total_validator(phases_costs: list[int], total_cost: int, should_pass: bool) -> None:
    """
    GIVEN phase cost breakdown and total_cost_eur
    WHEN EstimationResult is instantiated
    THEN phases_sum_matches_total succeeds when values match, or raises ValidationError when they diverge.

    :param phases_costs: List of individual phase costs in EUR.
    :param total_cost: Stated total cost in EUR.
    :param should_pass: Whether validation should succeed.
    """
    phases = [
        Phase(
            name=f"Phase {i + 1}",
            duration_weeks=2,
            cost_eur=cost,
            summary=f"Deliverable summary for phase {i + 1}.",
        )
        for i, cost in enumerate(phases_costs)
    ]
    payload = {
        "summary": "Project effort breakdown covering delivery milestones.",
        "confidence_pct": 80,
        "phases": phases,
        "total_duration_weeks": len(phases) * 2,
        "total_cost_eur": total_cost,
    }
    if should_pass:
        result = EstimationResult(**payload)
        assert result.total_cost_eur == total_cost
    else:
        with pytest.raises(ValidationError, match=r"does not match"):
            EstimationResult(**payload)


@pytest.mark.parametrize(
    ("confidence_pct", "summary", "should_pass"),
    [
        (25, f"{OUT_OF_SCOPE_PREFIX} The project requirements are too ambiguous to estimate.", True),
        (10, "The project is completely undefined.", False),
        (LOW_CONFIDENCE_THRESHOLD, "Marginal confidence project scope description.", True),
        (90, "High confidence well-defined web application scope.", True),
    ],
    ids=[
        "low_confidence_with_prefix_passes",
        "low_confidence_without_prefix_fails",
        "threshold_confidence_without_prefix_passes",
        "high_confidence_passes",
    ],
)
def test_estimation_result_low_confidence_requires_out_of_scope_prefix(confidence_pct: int, summary: str, should_pass: bool) -> None:
    """
    GIVEN confidence_pct and project summary narrative
    WHEN EstimationResult is instantiated
    THEN low_confidence_requires_out_of_scope_prefix enforces prefix rules when confidence < 30.

    :param confidence_pct: Stated confidence percentage.
    :param summary: Summary string.
    :param should_pass: Whether validation is expected to succeed.
    """
    payload = {
        "summary": summary,
        "confidence_pct": confidence_pct,
        "phases": [
            Phase(
                name="Discovery",
                duration_weeks=2,
                cost_eur=5000,
                summary="Initial discovery and technical research phase.",
            )
        ],
        "total_duration_weeks": 2,
        "total_cost_eur": 5000,
    }
    if should_pass:
        result = EstimationResult(**payload)
        assert result.confidence_pct == confidence_pct
    else:
        with pytest.raises(ValidationError, match=r"requires summary"):
            EstimationResult(**payload)


def test_estimation_response_serialization() -> None:
    """
    GIVEN a valid structured EstimationResult and telemetry
    WHEN EstimationResponse is instantiated
    THEN attributes are serialized accurately into dict and JSON.
    """
    result = EstimationResult(
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
        total_duration_weeks=8,
        total_cost_eur=20000,
    )
    resp = EstimationResponse(
        result=result,
        prompt_version="v1",
        cached=False,
        model=DEFAULT_GPT_LLM,
        provider="openai",
        cost_usd=0.0012,
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )
    data = resp.model_dump()
    assert data["result"]["total_cost_eur"] == 20000
    assert data["result"]["phases"][0]["name"] == "Phase 1: Architecture"
    assert data["prompt_version"] == "v1"
    assert data["cached"] is False
    assert data["model"] == DEFAULT_GPT_LLM
    assert data["provider"] == "openai"
    assert data["cost_usd"] == 0.0012
    assert data["usage"]["total_tokens"] == 150

    from_service = EstimationResponse.from_service_result(
        {
            "result": result,
            "cache_hit": True,
            "prompt_version": "v2",
            "model": DEFAULT_GPT_LLM,
            "provider": "openai",
            "cost_usd": 0.0012,
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
    )
    assert from_service.cached is True
    assert from_service.prompt_version == "v2"
    assert from_service.result.total_cost_eur == 20000

