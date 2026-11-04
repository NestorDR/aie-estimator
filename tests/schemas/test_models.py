# tests/schemas/test_models.py

# --- Python Standard Library ---
from dataclasses import FrozenInstanceError
from typing import Any

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
    PromptEstimationRequest,
)
from aie_estimator.schemas.models import (
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
    ModelSpec,
)
import aie_estimator.schemas.models as models_module
from aie_estimator.settings import get_settings


def test_model_spec_dataclass_attributes_and_defaults() -> None:
    """
    GIVEN the ModelSpec dataclass
    WHEN instantiated with default or explicit parameters
    THEN fields reflect the expected values and default boolean flags are False.
    """
    spec_default = ModelSpec(provider="openai")
    assert spec_default.provider == "openai"
    assert spec_default.is_reasoning is False
    assert spec_default.is_profile is False

    spec_custom = ModelSpec(provider="litellm", is_reasoning=True, is_profile=True)
    assert spec_custom.provider == "litellm"
    assert spec_custom.is_reasoning is True
    assert spec_custom.is_profile is True


def test_model_spec_is_frozen_immutable() -> None:
    """
    GIVEN an instantiated ModelSpec
    WHEN attempting to modify an attribute
    THEN a FrozenInstanceError is raised.
    """
    spec = ModelSpec(provider="openai")
    with pytest.raises(FrozenInstanceError):
        spec.provider = "anthropic"  # type: ignore[misc]


def test_schemas_models_contains_no_io_or_loader_methods() -> None:
    """
    GIVEN the schemas.models module
    WHEN checking its exported attributes
    THEN no file I/O or catalog loading functions are present in the module namespace.
    """
    assert not hasattr(models_module, "get_model_catalog")
    assert not hasattr(models_module, "resolve_provider")
    assert not hasattr(models_module, "get_profiles_config_path")
    assert not hasattr(models_module, "get_supported_models")


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
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    with pytest.raises(ValidationError, match=r"Input should be"):
        EstimationRequest(transcription=valid_transcript, model=invalid_model)  # type: ignore[arg-type]


def test_estimation_request_valid_model_accepted() -> None:
    """
    GIVEN a valid model registered in the catalog
    WHEN EstimationRequest is instantiated
    THEN the instance is created with the expected model value.
    """
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    req = EstimationRequest(transcription=valid_transcript, model=DEFAULT_GPT_LLM)
    assert req.model == DEFAULT_GPT_LLM


def test_estimation_request_none_model_defaults_to_settings_llm_model() -> None:
    """
    GIVEN an EstimationRequest instantiated without a model or with model=None
    WHEN the instance is created
    THEN the model attribute defaults to settings.llm_model.
    """
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    req_omitted = EstimationRequest(transcription=valid_transcript)
    req_none = EstimationRequest(transcription=valid_transcript, model=None)
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
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    req = EstimationRequest(transcription=valid_transcript, model=profile_model)
    assert req.model == profile_model


@pytest.mark.parametrize(
    ("engine_input", "expected_engine"),
    [
        (None, "litellm"),
        ("litellm", "litellm"),
        ("native", "native"),
    ],
    ids=["default_litellm", "explicit_litellm", "explicit_native"],
)
def test_estimation_request_engine_selection(engine_input: str | None, expected_engine: str) -> None:
    """
    GIVEN an EstimationRequest instantiated with default or explicit engine
    WHEN the instance is created
    THEN engine matches the expected execution engine.

    :param engine_input: Engine parameter passed, or None to test default.
    :param expected_engine: Expected engine string.
    """
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    kwargs: dict[str, Any] = {"transcription": valid_transcript}
    if engine_input is not None:
        kwargs["engine"] = engine_input

    req = EstimationRequest(**kwargs)
    assert req.engine == expected_engine


def test_estimation_request_invalid_engine_raises_validation_error() -> None:
    """
    GIVEN an invalid engine identifier
    WHEN EstimationRequest is instantiated
    THEN a pydantic ValidationError is raised.
    """
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    with pytest.raises(ValidationError, match=r"Input should be"):
        EstimationRequest(transcription=valid_transcript, engine="unknown_engine")  # type: ignore[arg-type]


def test_prompt_estimation_request_valid() -> None:
    """
    GIVEN valid prompt estimation parameters
    WHEN PromptEstimationRequest is instantiated
    THEN attributes match the inputs.
    """
    req = PromptEstimationRequest(
        description="Mobile application for iOS and Android with authentication and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )
    assert req.project_type == ProjectType.MOBILE_APP
    assert req.detail_level == DetailLevel.DETAILED
    assert req.output_format == OutputFormat.PHASES_TABLE
    assert "Mobile application" in req.description


def test_prompt_estimation_request_short_description_raises_validation_error() -> None:
    """
    GIVEN a description shorter than 20 characters
    WHEN PromptEstimationRequest is instantiated
    THEN a pydantic ValidationError is raised.
    """
    with pytest.raises(ValidationError, match=r"at least 20 characters"):
        PromptEstimationRequest(
            description="Too short",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.SUMMARY,
            output_format=OutputFormat.NARRATIVE,
        )

