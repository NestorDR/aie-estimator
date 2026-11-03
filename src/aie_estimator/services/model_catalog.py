# src/aie_estimator/services/model_catalog.py

# --- Python Standard Library ---
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

# --- Third Party Libraries ---
import structlog
import yaml

# --- App modules ---
from aie_estimator.schemas.models import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
    UNKNOWN_PROVIDER,
    ModelSpec,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

# Logical names to group model categories, decoupling the app from specific vendors
_LOGICAL_PROFILE_NAMES: Final[set[str]] = {DEFAULT_FAST_PROFILE, DEFAULT_DEEP_PROFILE}

# Supported LLM providers and their model prefixes, used to identify vendors and map API key env vars
PROVIDER_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "openai": ("openai/", "gpt-"),
    "anthropic": ("anthropic/", "claude-"),
    "gemini": ("gemini/",),
    "mistral": ("mistral/",),
    "scaleway": ("scaleway/",),
    "nebius": ("nebius/",),
}

_DEFAULT_CATALOG: Final[dict[str, ModelSpec]] = {
    DEFAULT_FAST_PROFILE: ModelSpec(provider="litellm", is_reasoning=True, is_profile=True),
    DEFAULT_DEEP_PROFILE: ModelSpec(provider="litellm", is_reasoning=True, is_profile=True),
    DEFAULT_GPT_LLM: ModelSpec(provider="openai", is_reasoning=True),
    DEFAULT_CLAUDE_LLM: ModelSpec(provider="anthropic", is_reasoning=True),
}


def resolve_provider(litellm_params: dict[str, Any]) -> str:
    """
    Identify the LLM provider from a model's configuration parameters.

    :param litellm_params: Dictionary of LiteLLM parameters for a model entry.

    :return: Resolved provider name, or UNKNOWN_PROVIDER if unidentified.
    """
    api_key_ref = str(litellm_params.get("api_key", ""))
    for provider in PROVIDER_PREFIXES:
        if api_key_ref == f"os.environ/{provider.upper()}_API_KEY":
            return provider

    model_target = str(litellm_params.get("model", ""))
    for provider, prefixes in PROVIDER_PREFIXES.items():
        if model_target.startswith(prefixes):
            return provider

    return UNKNOWN_PROVIDER


def get_profiles_config_path() -> Path:
    """
    Resolve the filesystem path to the llm_profiles.yml configuration file.

    :return: Absolute Path to the package-relative llm_profiles.yml file.
    """
    return Path(__file__).resolve().parent.parent / "llm_profiles.yml"


@lru_cache(maxsize=1)
def get_model_catalog() -> dict[str, ModelSpec]:
    """
    Parse and assemble the model catalog from llm_profiles.yml.

    :return: Mapping of model or profile identifiers to ModelSpec metadata instances.
    """
    config_path = get_profiles_config_path()
    if not config_path.is_file():
        log.warning("llm_profiles_file_not_found", config_path=str(config_path))
        return dict(_DEFAULT_CATALOG)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception as e:
        log.exception("llm_profiles_error_reading_file", config_path=str(config_path), error=repr(e))
        return dict(_DEFAULT_CATALOG)

    model_list: list[dict[str, Any]] = data.get("model_list", [])
    if not model_list:
        log.warning("llm_profiles_empty_model_list", config_path=str(config_path))
        return dict(_DEFAULT_CATALOG)

    catalog: dict[str, ModelSpec] = dict(_DEFAULT_CATALOG)
    for entry in model_list:
        model_name: str | None = entry.get("model_name")
        if not model_name:
            log.warning("llm_profiles_unnamed_model", config_path=str(config_path))
            continue

        if model_name in _LOGICAL_PROFILE_NAMES:
            catalog[model_name] = ModelSpec(provider="litellm", is_reasoning=True, is_profile=True)
            continue

        litellm_params: dict[str, Any] = entry.get("litellm_params", {})
        provider = resolve_provider(litellm_params)
        if provider == UNKNOWN_PROVIDER:
            log.warning("llm_profiles_unknown_provider", config_path=str(config_path), litellm_params=litellm_params)
            continue

        is_reasoning = bool(
            "reasoning_effort" in litellm_params
            or "thinking" in litellm_params
            or "output_config" in litellm_params
            or model_name in {DEFAULT_GPT_LLM, DEFAULT_CLAUDE_LLM}
        )
        catalog[model_name] = ModelSpec(provider=provider, is_reasoning=is_reasoning, is_profile=False)

    log.info(
        "llm_profiles_catalog_loaded",
        models={name: spec.provider for name, spec in catalog.items()},
    )

    return catalog


def get_model_provider(model: str) -> str:
    """
    Resolve the upstream provider identifier for a given model or profile name.

    :param model: The model or logical profile identifier.
    :return: Provider string identifier, or UNKNOWN_PROVIDER if not registered.
    """
    spec = get_model_catalog().get(model)
    return spec.provider if spec is not None else UNKNOWN_PROVIDER


@lru_cache(maxsize=1)
def get_supported_models() -> tuple[str, ...]:
    """
    Return all supported model identifiers and logical profile names from the catalog.

    :return: Tuple containing supported model and profile names.
    """
    return tuple(get_model_catalog().keys())
