# src/aie_estimator/services/native_providers.py

# --- Python Standard Library ---
import asyncio
from functools import lru_cache
import random
from typing import Any, Final

# --- Third Party Libraries ---
from anthropic import AsyncAnthropic
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
import structlog

# --- App modules ---
from aie_estimator.schemas.models import DEFAULT_GPT_LLM
from aie_estimator.services import DEFAULT_MAX_TOKENS, LLMServiceError, format_provider_response
from aie_estimator.services.model_catalog import get_model_catalog
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI:
    """
    Retrieve or initialize the cached asynchronous OpenAI API client instance.

    :return: Authenticated AsyncOpenAI client.
    :raises LLMServiceError: If the OpenAI API key is missing.
    """
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not api_key:
        log.error("openai_api_key_missing")
        raise LLMServiceError("OpenAI API key is not configured in the application environment.")
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_anthropic_client() -> AsyncAnthropic:
    """
    Retrieve or initialize the cached asynchronous Anthropic API client instance.

    :return: Authenticated AsyncAnthropic client with configured retries.
    :raises LLMServiceError: If the Anthropic API key is missing.
    """
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if not api_key:
        log.error("anthropic_api_key_missing")
        raise LLMServiceError("Anthropic API key is not configured in the application environment.")
    return AsyncAnthropic(api_key=api_key, max_retries=2)


async def call_openai(
    messages: str | list[dict[str, Any]],
    system_prompt: str | None,
    model: str = DEFAULT_GPT_LLM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
    effort: str = "medium",
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Send an estimation request to the OpenAI Responses API with retries and exponential backoff.

    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional system instructions and reference context.
    :param model: Target model identifier.
    :param max_tokens: Maximum output token budget.
    :param temperature: Sampling temperature for non-reasoning models.
    :param effort: Reasoning effort for reasoning models ('low', 'medium', 'high').
    :param max_retries: Maximum number of retry attempts for transient failures.

    :return: Standardized response dictionary containing estimation, model, provider, finish_reason, and usage.
    :raises LLMServiceError: If retries are exhausted or an unrecoverable API error occurs.
    """
    client = get_openai_client()
    spec = get_model_catalog().get(model)
    is_reasoning_model = spec.is_reasoning if spec is not None else False

    request_params: dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
        "store": False,
    }

    if system_prompt:
        request_params["instructions"] = system_prompt

    if is_reasoning_model:
        request_params["reasoning"] = {"effort": effort}
    elif temperature is not None:
        request_params["temperature"] = temperature

    response = None
    secure_rng = random.SystemRandom()

    for attempt in range(max_retries):
        try:
            response = await client.responses.create(**request_params)
            break
        except (BadRequestError, RateLimitError, InternalServerError, APIConnectionError) as e:
            if isinstance(e, BadRequestError) and "temperature" in str(e) and "temperature" in request_params:
                request_params.pop("temperature", None)
                log.warning("openai_temperature_stripped_on_bad_request", model=model, error=repr(e))

            if attempt >= max_retries - 1:
                log.exception("openai_provider_call_failed", model=model, attempt=attempt + 1, error=repr(e))
                raise LLMServiceError(f"OpenAI call failed: {e}") from e

            jitter: float = secure_rng.uniform(0, 1)
            wait: float = (2**attempt) + jitter
            log.warning(
                "openai_transient_error_retrying",
                model=model,
                attempt=attempt + 1,
                max_retries=max_retries,
                wait_seconds=round(wait, 2),
                error=repr(e),
            )
            await asyncio.sleep(wait)
        except Exception as e:
            log.exception("openai_unexpected_error", model=model, error=repr(e))
            raise LLMServiceError(f"OpenAI call failed: {e}") from e

    if response is None:
        log.error("llm_provider_call_exhausted", provider="openai", model=model)
        raise LLMServiceError("OpenAI call failed: request exhausted after retry attempts.")

    status = getattr(response, "status", "completed")
    if status != "completed":
        if status == "incomplete":
            detail = f"Incomplete response from OpenAI: {getattr(response, 'incomplete_details', None)}"
        elif status == "failed":
            detail = f"OpenAI request failed: {getattr(response, 'error', None)}"
        else:
            detail = f"OpenAI response status not completed: {status}"
        log.error("llm_response_not_completed", provider="openai", model=model, status=status, detail=detail)
        raise LLMServiceError(detail)

    finish_reason = "stop"
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    log.info(
        "llm_response_received",
        provider="openai",
        model=getattr(response, "model", model),
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return format_provider_response(
        estimation=getattr(response, "output_text", ""),
        model=getattr(response, "model", model),
        provider="openai",
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def call_anthropic(
    messages: str | list[dict[str, Any]],
    system_prompt: str | None,
    model: str,
    max_tokens: int,
    temperature: float = 0.3,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Send an estimation request to the Anthropic Messages API with structured logging and asynchronous capability.

    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional system instructions and reference context.
    :param model: Target model identifier.
    :param max_tokens: Maximum output token budget.
    :param temperature: Sampling temperature (ignored and unset when thinking/reasoning is configured).
    :param reasoning_effort: Qualitative reasoning effort tier ('low', 'medium', 'high').

    :return: Standardized response dictionary containing estimation, model, provider, finish_reason, and usage.
    :raises LLMServiceError: If the API call fails or encounters an unhandled provider error.
    """
    client = get_anthropic_client()

    formatted_messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if reasoning_effort is not None:
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": reasoning_effort}
    elif temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = await client.messages.create(**kwargs)
    except Exception as e:
        log.exception(
            "llm_provider_call_failed",
            provider="anthropic",
            model=model,
            error_type=type(e).__name__,
            error=repr(e),
        )
        raise LLMServiceError(f"Anthropic call failed: {e}") from e

    finish_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
    if finish_reason == "max_tokens":
        log.warning("llm_response_truncated", provider="anthropic", model=model, stop_reason=finish_reason)

    text_block = next(
        (block for block in response.content if getattr(block, "type", None) == "text"),
        None,
    )
    estimation_text = getattr(text_block, "text", "") if text_block else ""

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    log.info(
        "llm_response_received",
        provider="anthropic",
        model=getattr(response, "model", model),
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return format_provider_response(
        estimation=estimation_text,
        model=getattr(response, "model", model),
        provider="anthropic",
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
