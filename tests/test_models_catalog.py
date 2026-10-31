# tests/test_models_catalog.py

# --- Python Standard Library ---
from pathlib import Path
from typing import Any
from unittest.mock import patch

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationRequest
from aie_estimator.schemas.models import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
    PROVIDER_PREFIXES,
    UNKNOWN_PROVIDER,
    ModelSpec,
    get_model_catalog,
    get_supported_models,
    resolve_provider,
)
from aie_estimator.settings import get_settings


def test_model_catalog_keys_match_supported_models_tuple() -> None:
    """
    GIVEN the get_supported_models() tuple and get_model_catalog()
    WHEN comparing their member keys
    THEN all tuple elements match the keys in the catalog exactly.
    """
    assert set(get_supported_models()) == set(get_model_catalog().keys())


@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_reasoning"),
    [
        (DEFAULT_GPT_LLM, "openai", True),
        (DEFAULT_CLAUDE_LLM, "anthropic", True),
    ],
    ids=[
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
    GIVEN a model registered in get_model_catalog()
    WHEN its ModelSpec attributes are inspected
    THEN the provider and reasoning capability flags match expected specifications.

    :param model_name: Target model identifier.
    :param expected_provider: Expected provider identifier ('openai' or 'anthropic').
    :param expected_reasoning: Expected boolean indicating reasoning/thinking support.
    """
    spec: ModelSpec = get_model_catalog()[model_name]
    assert spec.provider == expected_provider
    assert spec.is_reasoning is expected_reasoning


@pytest.mark.parametrize(
    ("litellm_params", "expected_provider"),
    [
        ({"api_key": "os.environ/SCALEWAY_API_KEY", "model": "openai/deepseek-ai/DeepSeek-V4-Flash"}, "scaleway"),
        ({"api_key": "os.environ/NEBIUS_API_KEY", "model": "openai/deepseek-ai/DeepSeek-R2"}, "nebius"),
        ({"model": "gemini/gemini-3.5-flash-lite"}, "gemini"),
        ({"model": "mistral/mistral-small-3.2"}, "mistral"),
        ({"model": DEFAULT_CLAUDE_LLM}, "anthropic"),
        ({"model": "unknown-custom-model"}, UNKNOWN_PROVIDER),
    ],
    ids=["scaleway_by_env", "nebius_by_env", "gemini_by_prefix", "mistral_by_prefix", "anthropic_by_prefix", "unknown_provider"],
)
def test_resolve_provider_via_provider_prefixes(litellm_params: dict[str, Any], expected_provider: str) -> None:
    """
    GIVEN LiteLLM model parameters
    WHEN resolve_provider is called
    THEN it resolves the provider from PROVIDER_PREFIXES via api_key or model prefix, or returns UNKNOWN_PROVIDER.

    :param litellm_params: Model parameter dictionary.
    :param expected_provider: Expected resolved provider string.
    """
    assert expected_provider in PROVIDER_PREFIXES or expected_provider == UNKNOWN_PROVIDER
    assert resolve_provider(litellm_params) == expected_provider


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
    "profile_name",
    [
        DEFAULT_FAST_PROFILE,
        DEFAULT_DEEP_PROFILE,
    ],
    ids=["profile_reasoner_fast", "profile_reasoner_deep"],
)
def test_logical_profiles_registered_in_catalog(profile_name: str) -> None:
    """
    GIVEN a defined logical profile name
    WHEN the catalog is inspected
    THEN the profile is present, marked as is_profile=True, and has provider='litellm'.

    :param profile_name: Logical profile identifier.
    """
    catalog = get_model_catalog()
    assert profile_name in catalog
    spec = catalog[profile_name]
    assert spec.is_profile is True
    assert spec.provider == "litellm"
    assert spec.is_reasoning is True


def test_dynamic_model_catalog_fallback_on_missing_file(tmp_path: Path) -> None:
    """
    GIVEN a non-existent configuration file path
    WHEN get_model_catalog is called
    THEN the fallback default catalog is returned safely without raising exceptions, excluding removed models.

    :param tmp_path: Pytest temporary directory fixture.
    """
    non_existent = tmp_path / "missing_profiles.yml"
    with patch("aie_estimator.schemas.models.get_profiles_config_path", return_value=non_existent):
        get_model_catalog.cache_clear()
        catalog = get_model_catalog()
        assert DEFAULT_GPT_LLM in catalog
        assert DEFAULT_FAST_PROFILE in catalog
        assert "gpt-4o-mini" not in catalog
        assert "gpt-5-nano" not in catalog
    get_model_catalog.cache_clear()


def test_dynamic_model_catalog_fallback_on_corrupted_yaml_and_empty_list(tmp_path: Path) -> None:
    """
    GIVEN a corrupted YAML file or a YAML file with an empty model_list
    WHEN get_model_catalog is called
    THEN the exception or empty list is logged and the fallback default catalog is returned.

    :param tmp_path: Pytest temporary directory fixture.
    """
    corrupted_yaml = tmp_path / "corrupted_profiles.yml"
    corrupted_yaml.write_text("::: invalid_yaml: [unclosed", encoding="utf-8")
    with patch("aie_estimator.schemas.models.get_profiles_config_path", return_value=corrupted_yaml):
        get_model_catalog.cache_clear()
        catalog = get_model_catalog()
        assert DEFAULT_GPT_LLM in catalog
        assert DEFAULT_CLAUDE_LLM in catalog

    empty_yaml = tmp_path / "empty_profiles.yml"
    empty_yaml.write_text("model_list: []\n", encoding="utf-8")
    with patch("aie_estimator.schemas.models.get_profiles_config_path", return_value=empty_yaml):
        get_model_catalog.cache_clear()
        catalog = get_model_catalog()
        assert DEFAULT_GPT_LLM in catalog
        assert DEFAULT_CLAUDE_LLM in catalog

    unknown_yaml = tmp_path / "unknown_provider_profiles.yml"
    unknown_yaml.write_text(
        "model_list:\n  - model_name: 'custom-unrecognized'\n    litellm_params:\n      model: 'unrecognized/model-1'\n",
        encoding="utf-8",
    )
    with patch("aie_estimator.schemas.models.get_profiles_config_path", return_value=unknown_yaml):
        get_model_catalog.cache_clear()
        catalog = get_model_catalog()
        assert "custom-unrecognized" not in catalog
    get_model_catalog.cache_clear()


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
