# tests/test_models_catalog.py

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationRequest
from aie_estimator.schemas.models import (
    MODEL_CATALOG,
    ModelSpec,
    SUPPORTED_MODELS,
)
from aie_estimator.settings import get_settings


def test_model_catalog_keys_match_supported_models_tuple() -> None:
    """
    GIVEN the SUPPORTED_MODELS tuple and MODEL_CATALOG
    WHEN comparing their member keys
    THEN all tuple elements match the keys in the catalog exactly.
    """
    assert set(SUPPORTED_MODELS) == set(MODEL_CATALOG.keys())


@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_reasoning"),
    [
        ("gpt-4o-mini", "openai", False),
        ("gpt-6-luna", "openai", True),
        ("claude-haiku-4-5-20251001", "anthropic", True),
    ],
    ids=[
        "gpt4o_mini_openai_standard",
        "gpt6_luna_openai_reasoning",
        "claude_haiku_anthropic_reasoning",
    ],
)
def test_model_catalog_specs(
    model_name: str,
    expected_provider: str,
    expected_reasoning: bool,
) -> None:
    """
    GIVEN a model registered in MODEL_CATALOG
    WHEN its ModelSpec attributes are inspected
    THEN the provider and reasoning capability flags match expected specifications.

    :param model_name: Target model identifier.
    :param expected_provider: Expected provider identifier ('openai' or 'anthropic').
    :param expected_reasoning: Expected boolean indicating reasoning/thinking support.
    """
    spec: ModelSpec = MODEL_CATALOG[model_name]
    assert spec.provider == expected_provider
    assert spec.is_reasoning is expected_reasoning


@pytest.mark.parametrize(
    "invalid_model",
    [
        "string",
        "invalid_model",
        "llama-3",
        "gemini-2.5-flash",
    ],
    ids=["swagger_placeholder_string", "arbitrary_invalid", "unsupported_llama", "unsupported_gemini"],
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
    GIVEN a valid model registered in MODEL_CATALOG
    WHEN EstimationRequest is instantiated
    THEN the instance is created with the expected model value.
    """
    valid_transcript = "Client: We need a scalable notification service supporting email and SMS. Team: We can implement webhooks."
    req = EstimationRequest(transcription=valid_transcript, model="gpt-4o-mini")
    assert req.model == "gpt-4o-mini"


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

