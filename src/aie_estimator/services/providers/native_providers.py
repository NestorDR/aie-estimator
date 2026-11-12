# src/aie_estimator/services/providers/native_providers.py

# --- Python Standard Library ---
from functools import lru_cache
from typing import Any, Final

# --- Third Party Libraries ---
from anthropic import AsyncAnthropic
import instructor
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel
import structlog

# --- App modules ---
from aie_estimator.model_catalog import get_model_catalog
from aie_estimator.schemas.estimations import EstimationResult
from aie_estimator.services.exceptions import LLMProviderError, unwrap_exception
from aie_estimator.services.providers.formatting import (
    build_chat_messages,
    format_provider_response,
    normalize_turn_messages,
)
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """
    Retrieve or initialize the cached asynchronous OpenAI API client instance.

    :return: Authenticated AsyncOpenAI client.
    :raises LLMProviderError: If the OpenAI API key is missing.
    """
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not api_key:
        log.error("provider_api_key_missing", provider="openai")
        raise LLMProviderError("OpenAI API key is not configured in the application environment.")
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_anthropic_client() -> AsyncAnthropic:
    """
    Retrieve or initialize the cached asynchronous Anthropic API client instance.

    :return: Authenticated AsyncAnthropic client with configured retries.
    :raises LLMProviderError: If the Anthropic API key is missing.
    """
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if not api_key:
        log.error("provider_api_key_missing", provider="anthropic")
        raise LLMProviderError("Anthropic API key is not configured in the application environment.")
    return AsyncAnthropic(api_key=api_key, max_retries=2)


async def call_openai(
    user_prompt: str | None,
    system_prompt: str | None,
    opts: GenerationOptions,
    *,
    messages: list[dict[str, str]] | None = None,
    response_model: type[BaseModel] = EstimationResult,
) -> dict[str, Any]:
    """
    Send a structured-output request to the OpenAI API with parameter recovery.

    :param user_prompt: Pre-rendered user prompt string for single-turn calls, or ``None`` when ``messages`` is passed.
    :param system_prompt: Optional system instructions and reference context.
    :param opts: Generation options specifying model, max_tokens, temperature, reasoning_effort, and max_retries.
    :param messages: Optional multi-turn list of non-system ``user``/``assistant`` message dictionaries.
    :param response_model: Target Pydantic model class for structured output extraction.

    :return: Standardized response dictionary containing result, model, provider, finish_reason, and usage.
    :raises LLMProviderError: If retries are exhausted or an unrecoverable API error occurs.
    :raises ValueError: If ``user_prompt`` and ``messages`` violate turn normalization rules.
    """
    chat_messages = build_chat_messages(user_prompt, system_prompt, messages)
    raw_client = get_openai_client()
    model = opts.effective_model
    spec = get_model_catalog().get(model)
    is_reasoning_model = spec.is_reasoning if spec is not None else False

    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    token_key = "max_completion_tokens" if is_reasoning_model else "max_tokens"
    request_params: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        token_key: opts.max_tokens,
    }

    if is_reasoning_model:
        request_params["reasoning_effort"] = opts.reasoning_effort or "medium"
    elif opts.temperature is not None:
        request_params["temperature"] = opts.temperature

    structured_result = None
    response = None

    for _ in range(2):
        try:
            structured_result, response = await client.chat.completions.create_with_completion(
                response_model=response_model,
                max_retries=opts.max_retries,
                **request_params,
            )
            break
        except Exception as e:
            err = unwrap_exception(e)
            if isinstance(err, BadRequestError):
                if "temperature" in str(err) and "temperature" in request_params:
                    request_params.pop("temperature", None)
                    log.warning("provider_temperature_stripped", provider="openai", model=model, error=repr(err))
                    continue
                if "max_tokens" in str(err) and "max_tokens" in request_params:
                    request_params["max_completion_tokens"] = request_params.pop("max_tokens")
                    log.warning("provider_tokens_remapped", provider="openai", model=model, error=repr(err))
                    continue

            log.exception(
                "provider_call_failed",
                provider="openai",
                model=model,
                error=repr(err),
            )
            raise LLMProviderError(f"OpenAI call failed: {err}") from e
    else:
        log.error("provider_retries_exhausted", provider="openai", model=model)
        raise LLMProviderError("OpenAI call failed: request exhausted after retry attempts.")

    if structured_result is None or response is None:
        log.error("provider_retries_exhausted", provider="openai", model=model)
        raise LLMProviderError("OpenAI call failed: request exhausted after retry attempts.")

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    log.info(
        "provider_response_received",
        provider="openai",
        model=getattr(response, "model", model),
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return format_provider_response(
        result=structured_result,
        model=getattr(response, "model", model),
        provider="openai",
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def call_anthropic(
    user_prompt: str | None,
    system_prompt: str | None,
    opts: GenerationOptions,
    *,
    messages: list[dict[str, str]] | None = None,
    response_model: type[BaseModel] = EstimationResult,
) -> dict[str, Any]:
    """
    Send a structured-output request to the Anthropic Messages API with structured logging and async capability.

    :param user_prompt: Pre-rendered user prompt string for single-turn calls, or ``None`` when ``messages`` is passed.
    :param system_prompt: Optional system instructions and reference context passed as Anthropic's top-level ``system``.
    :param opts: Generation options specifying model, max_tokens, temperature, and reasoning_effort.
    :param messages: Optional multi-turn list of non-system ``user``/``assistant`` message dictionaries.
    :param response_model: Target Pydantic model class for structured output extraction.

    :return: Standardized response dictionary containing result, model, provider, finish_reason, and usage.
    :raises LLMProviderError: If the API call fails or encounters an unhandled provider error.
    :raises ValueError: If ``user_prompt`` and ``messages`` violate turn normalization rules.
    """
    formatted_messages = normalize_turn_messages(user_prompt, messages)
    raw_client = get_anthropic_client()
    model = opts.effective_model

    client = instructor.from_anthropic(raw_client)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": opts.max_tokens,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if opts.reasoning_effort is not None:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": opts.reasoning_effort}
    elif opts.temperature is not None:
        kwargs["temperature"] = opts.temperature

    try:
        structured_result, response = await client.messages.create_with_completion(
            response_model=response_model,
            max_retries=opts.max_retries,
            **kwargs,
        )
    except Exception as e:
        err = unwrap_exception(e)
        log.exception(
            "provider_call_failed",
            provider="anthropic",
            model=model,
            error=repr(err),
        )
        raise LLMProviderError(f"Anthropic call failed: {err}") from e

    finish_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
    if finish_reason == "max_tokens":
        log.warning("provider_response_truncated", provider="anthropic", model=model, stop_reason=finish_reason)

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    log.info(
        "provider_response_received",
        provider="anthropic",
        model=getattr(response, "model", model),
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return format_provider_response(
        result=structured_result,
        model=getattr(response, "model", model),
        provider="anthropic",
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
