# tests/foundation/llm/test_model_catalog.py

# --- Python Standard Library ---
from pathlib import Path
from typing import Any

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.model_catalog import (
    get_model_catalog,
    get_model_provider,
    get_models_for_engine,
    get_supported_models,
    resolve_engine_model,
    resolve_provider,
)
from aie_estimator.utils.constants import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
    UNKNOWN_PROVIDER,
)


@pytest.mark.parametrize(
    ("model_name", "expected_provider", "expected_reasoning"),
    [
        (DEFAULT_GPT_LLM, "openai", True),
        (DEFAULT_CLAUDE_LLM, "anthropic", True),
        (DEFAULT_FAST_PROFILE, "litellm", True),
        (DEFAULT_DEEP_PROFILE, "litellm", True),
        ("non_existent_model_id", UNKNOWN_PROVIDER, None),
    ],
    ids=[
        "gpt6_luna_openai",
        "claude_haiku_anthropic",
        "reasoner_fast_litellm",
        "reasoner_deep_litellm",
        "unknown_model",
    ],
)
def test_model_catalog_specs_and_provider(
    model_name: str,
    expected_provider: str,
    expected_reasoning: bool | None,
) -> None:
    """
    GIVEN a known or unknown model identifier
    WHEN its ModelSpec is looked up and get_model_provider is queried
    THEN provider and reasoning flag match, and unknown models return UNKNOWN_PROVIDER.

    :param model_name: Target model identifier.
    :param expected_provider: Expected provider identifier string.
    :param expected_reasoning: Expected reasoning flag, or None if the model is not in the catalog.
    """
    assert get_model_provider(model_name) == expected_provider
    catalog = get_model_catalog()
    if expected_reasoning is None:
        assert model_name not in catalog
    else:
        assert catalog[model_name].is_reasoning is expected_reasoning


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
    ids=[
        "scaleway_by_env",
        "nebius_by_env",
        "gemini_by_prefix",
        "mistral_by_prefix",
        "anthropic_by_prefix",
        "unknown_provider",
    ],
)
def test_resolve_provider_via_prefixes(litellm_params: dict[str, Any], expected_provider: str) -> None:
    """
    GIVEN LiteLLM model parameters
    WHEN resolve_provider is called
    THEN it resolves the provider from PROVIDER_PREFIXES or returns UNKNOWN_PROVIDER.

    :param litellm_params: Model parameter dictionary.
    :param expected_provider: Expected provider string identifier.
    """
    assert resolve_provider(litellm_params) == expected_provider


VALID_PROFILES_YAML = (
    "model_list:\n"
    "  - model_name: 'test-model-beta'\n"
    "    litellm_params:\n"
    "      model: 'mistral/test-model-beta'\n"
    "  - litellm_params:\n"
    "      model: 'openai/unnamed'\n"
    "  - model_name: 'test-model-unknown'\n"
    "    litellm_params:\n"
    "      model: 'unknown-custom-model'\n"
)


@pytest.mark.parametrize(
    ("file_content", "expect_loaded_model"),
    [
        (None, False),
        ("::: unclosed [yaml invalid", False),
        (VALID_PROFILES_YAML, True),
    ],
    ids=["missing_file", "corrupted_yaml", "valid_file"],
)
def test_model_catalog_loads_config_or_falls_back(tmp_path: Path, file_content: str | None, expect_loaded_model: bool) -> None:
    """
    GIVEN a missing, corrupted, or valid profiles configuration file
    WHEN get_model_catalog is called with direct profiles_path parameter injection
    THEN defaults are always present; valid entries are added and unnamed or unknown-provider ones skipped.

    :param tmp_path: Pytest temporary directory fixture.
    :param file_content: Content to write to the config file, or None to leave it missing.
    :param expect_loaded_model: Whether the configured model must appear in the catalog.
    """
    config_file = tmp_path / "profiles.yml"
    if file_content is not None:
        config_file.write_text(file_content, encoding="utf-8")
    get_model_catalog.cache_clear()
    catalog = get_model_catalog(profiles_path=config_file)
    supported_models = get_supported_models(profiles_path=config_file)
    get_model_catalog.cache_clear()

    assert DEFAULT_GPT_LLM in catalog
    assert DEFAULT_FAST_PROFILE in catalog
    assert ("test-model-beta" in catalog) is expect_loaded_model
    assert ("test-model-beta" in supported_models) is expect_loaded_model
    assert "test-model-unknown" not in catalog
    assert "test-model-unknown" not in supported_models
    if expect_loaded_model:
        assert catalog["test-model-beta"].provider == "mistral"


@pytest.mark.parametrize(
    ("engine", "expected_models", "excluded_models"),
    [
        (
            "litellm",
            (DEFAULT_FAST_PROFILE, DEFAULT_DEEP_PROFILE),
            (DEFAULT_GPT_LLM, DEFAULT_CLAUDE_LLM, "test-model-beta"),
        ),
        (
            "native",
            (DEFAULT_GPT_LLM, DEFAULT_CLAUDE_LLM, "test-model-beta", "test-openai-extra"),
            (DEFAULT_FAST_PROFILE, DEFAULT_DEEP_PROFILE),
        ),
    ],
    ids=["litellm_profiles_only", "native_physical_models_only"],
)
def test_get_models_for_engine_filters_catalog(
    tmp_path: Path,
    engine: str,
    expected_models: tuple[str, ...],
    excluded_models: tuple[str, ...],
) -> None:
    """
    GIVEN a model catalog containing logical profiles and physical models
    WHEN get_models_for_engine is called for 'litellm' or 'native'
    THEN 'litellm' returns only profiles and 'native' returns only non-profile physical models.

    :param tmp_path: Pytest temporary directory fixture.
    :param engine: Target execution engine ('litellm' or 'native').
    :param expected_models: Models expected in the returned tuple.
    :param excluded_models: Models that must be excluded from the returned tuple.
    """
    profiles_file = tmp_path / "profiles.yml"
    profiles_file.write_text(
        VALID_PROFILES_YAML
        + "  - model_name: 'test-openai-extra'\n"
        + "    litellm_params:\n"
        + "      model: 'openai/gpt-6-mini'\n",
        encoding="utf-8",
    )
    get_model_catalog.cache_clear()
    models = get_models_for_engine(engine, profiles_path=profiles_file)  # type: ignore[arg-type]
    get_model_catalog.cache_clear()

    for expected in expected_models:
        assert expected in models
    for excluded in excluded_models:
        assert excluded not in models


def test_resolve_engine_model_raises_clear_error_when_catalog_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN get_models_for_engine returning no models for the requested engine
    WHEN resolve_engine_model is called
    THEN ValueError is raised with a clear configuration message instead of IndexError.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr("aie_estimator.model_catalog.get_models_for_engine", lambda _engine: ())
    with pytest.raises(ValueError, match=r"No models are configured for engine 'native'"):
        resolve_engine_model("native", None)
