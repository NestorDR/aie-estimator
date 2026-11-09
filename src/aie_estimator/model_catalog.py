# src/aie_estimator/model_catalog.py

# --- Python Standard Library ---
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

# --- Third Party Libraries ---
import structlog
import yaml

# --- App modules ---
from aie_estimator.settings import get_settings
from aie_estimator.utils.model_constants import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_DEEP_PROFILE,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
    LOGICAL_PROFILE_NAMES,
    PROVIDER_PREFIXES,
    UNKNOWN_PROVIDER,
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Metadata specification for supported large language models.

    :param provider: Upstream LLM provider identifier ('openai', 'anthropic', 'litellm', etc.).
    :param is_reasoning: Whether the model supports chain-of-thought reasoning capabilities.
    :param is_profile: Whether this entry represents a logical routing profile rather than a physical model.
    """

    provider: str
    is_reasoning: bool = False
    is_profile: bool = False


log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

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


@lru_cache(maxsize=4)
def get_model_catalog(profiles_path: Path | str | None = None) -> dict[str, ModelSpec]:
    """
    Parse and assemble the model catalog from an LLM profiles configuration YAML file.

    :param profiles_path: Explicit filesystem path override, or None to use Settings.llm_profiles_path.

    :return: Mapping of model or profile identifiers to ModelSpec metadata instances.
    """
    resolved_path = Path(profiles_path).resolve() if profiles_path is not None else get_settings().llm_profiles_path

    if not resolved_path.is_file():
        log.warning("model_catalog_file_not_found", profiles_path=str(resolved_path))
        return dict(_DEFAULT_CATALOG)

    try:
        with resolved_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception as e:
        log.exception("model_catalog_read_failed", profiles_path=str(resolved_path), error=repr(e))
        return dict(_DEFAULT_CATALOG)

    model_list: list[dict[str, Any]] = data.get("model_list", [])
    if not model_list:
        log.warning("model_catalog_empty_list", profiles_path=str(resolved_path))
        return dict(_DEFAULT_CATALOG)

    catalog: dict[str, ModelSpec] = dict(_DEFAULT_CATALOG)
    for entry in model_list:
        model_name: str | None = entry.get("model_name")
        if not model_name:
            log.warning("model_catalog_model_unnamed", profiles_path=str(resolved_path))
            continue

        if model_name in LOGICAL_PROFILE_NAMES:
            catalog[model_name] = ModelSpec(provider="litellm", is_reasoning=True, is_profile=True)
            continue

        litellm_params: dict[str, Any] = entry.get("litellm_params", {})
        provider = resolve_provider(litellm_params)
        if provider == UNKNOWN_PROVIDER:
            log.warning("model_catalog_unknown_provider", profiles_path=str(resolved_path), litellm_params=litellm_params)
            continue

        is_reasoning = bool(
            "reasoning_effort" in litellm_params
            or "thinking" in litellm_params
            or "output_config" in litellm_params
            or model_name in {DEFAULT_GPT_LLM, DEFAULT_CLAUDE_LLM}
        )
        catalog[model_name] = ModelSpec(provider=provider, is_reasoning=is_reasoning, is_profile=False)

    log.info(
        "model_catalog_loaded",
        profiles_path=str(resolved_path),
        models={name: spec.provider for name, spec in catalog.items()},
    )

    return catalog


def get_model_provider(model: str, profiles_path: Path | str | None = None) -> str:
    """
    Resolve the upstream provider identifier for a given model or profile name.

    :param model: The model or logical profile identifier.
    :param profiles_path: Explicit filesystem path override, or None to use Settings.llm_profiles_path.

    :return: Provider string identifier, or UNKNOWN_PROVIDER if not registered.
    """
    spec = get_model_catalog(profiles_path=profiles_path).get(model)
    return spec.provider if spec is not None else UNKNOWN_PROVIDER


def get_supported_models(profiles_path: Path | str | None = None) -> tuple[str, ...]:
    """
    Return all supported model identifiers and logical profile names from the catalog.

    :param profiles_path: Explicit filesystem path override, or None to use Settings.llm_profiles_path.

    :return: Tuple containing supported model and profile names.
    """
    return tuple(get_model_catalog(profiles_path=profiles_path).keys())
