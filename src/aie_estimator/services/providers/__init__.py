# src/aie_estimator/services/providers/__init__.py

# --- Python Standard Library ---
from typing import Any, Final

# --- Third Party Libraries ---
import structlog


# --- App modules ---
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers.generation_options import (
    DEFAULT_MAX_RETRIES as DEFAULT_MAX_RETRIES,
    DEFAULT_TEMPERATURE as DEFAULT_TEMPERATURE,
    GenerationOptions as GenerationOptions,
)
from aie_estimator.services.providers.litellm_provider import (
    call_litellm_router as call_litellm_router,
    get_litellm_router as get_litellm_router,
)
from aie_estimator.services.providers.model_catalog import (
    get_model_catalog as get_model_catalog,
    get_model_provider as get_model_provider,
    get_supported_models as get_supported_models,
)
from aie_estimator.services.providers.native_providers import (
    call_anthropic as call_anthropic,
    call_openai as call_openai,
    get_anthropic_client as get_anthropic_client,
    get_openai_client as get_openai_client,
)
from aie_estimator.services.providers.pricing import MODEL_COSTS as MODEL_COSTS, estimate_cost as estimate_cost


log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


async def call_llm(
    user_prompt: str,
    system_prompt: str,
    opts: GenerationOptions,
) -> dict[str, Any]:
    """
    Dispatch an LLM completion request to the configured execution engine and provider.

    :param user_prompt: Pre-rendered user prompt string.
    :param system_prompt: Pre-rendered system prompt string.
    :param opts: Generation options controlling model, engine, max_tokens, temperature, and reasoning.

    :return: Standardized provider response dictionary.
    :raises LLMProviderError: If the provider is unsupported or the upstream call fails.
    """
    model = opts.effective_model
    provider = opts.provider

    if opts.engine == "litellm":
        return await call_litellm_router(user_prompt=user_prompt, system_prompt=system_prompt, opts=opts)
    if provider == "openai":
        return await call_openai(user_prompt=user_prompt, system_prompt=system_prompt, opts=opts)
    if provider == "anthropic":
        return await call_anthropic(user_prompt=user_prompt, system_prompt=system_prompt, opts=opts)

    log.error("unsupported_provider_for_native_engine", provider=provider, model=model)
    raise LLMProviderError(f"Unsupported provider '{provider}' for legacy native engine.")
