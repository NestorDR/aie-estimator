# src/aie_estimator/services/providers/dispatcher.py

# --- Python Standard Library ---
from typing import Any, Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.services.providers.litellm_provider import call_litellm_router
from aie_estimator.services.providers.native_providers import call_anthropic, call_openai

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


async def call_llm(
    user_prompt: str | None,
    system_prompt: str | None,
    opts: GenerationOptions,
    *,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Dispatch an LLM completion request to the configured execution engine and provider.

    :param user_prompt: Pre-rendered user prompt string for single-turn calls, or ``None`` when ``messages`` is passed.
    :param system_prompt: Optional pre-rendered system prompt string.
    :param opts: Generation options controlling model, engine, max_tokens, temperature, and reasoning.
    :param messages: Optional multi-turn list of non-system ``user``/``assistant`` message dictionaries.

    :return: Standardized provider response dictionary.
    :raises LLMProviderError: If the provider is unsupported or the upstream call fails.
    """
    model = opts.effective_model
    provider = opts.provider

    if opts.engine == "litellm":
        return await call_litellm_router(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            opts=opts,
            messages=messages,
        )
    if provider == "openai":
        return await call_openai(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            opts=opts,
            messages=messages,
        )
    if provider == "anthropic":
        return await call_anthropic(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            opts=opts,
            messages=messages,
        )

    log.error("provider_dispatch_unsupported_provider", provider=provider, model=model)
    raise LLMProviderError(f"Unsupported provider '{provider}' for legacy native engine.")
