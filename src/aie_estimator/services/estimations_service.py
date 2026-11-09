# src/aie_estimator/services/estimations_service.py

# --- Python Standard Library ---
import time
from typing import Any, Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.guardrails.input import check_input, get_moderation_client
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.schemas.estimations import EstimationResult
from aie_estimator.services.cache import EstimationCache, get_estimation_cache
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers import GenerationOptions, call_llm

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


async def generate_estimation(
    system_prompt: str,
    user_prompt: str,
    opts: GenerationOptions | None = None,
) -> dict[str, Any]:
    """
    Coordinate the end-to-end software project estimation workflow using multirole prompts.

    :param system_prompt: Pre-rendered system prompt string containing instructions and calibration examples.
    :param user_prompt: Pre-rendered user prompt string containing the project details.
    :param opts: Optional generation options controlling model selection, engine, and token limits.

    :return: Response dictionary containing validated EstimationResult, model metadata, token usage, latency, and cache hit indicator.
    :raises InputGuardrailViolation: If input guardrails reject the prompt.
    :raises LLMProviderError: If any LLM provider invocation encounters an error.
    :raises ValueError: If user_prompt is not provided.
    """
    if not user_prompt:
        raise ValueError("user_prompt must be provided.")

    # Input guardrails — raises InputGuardrailViolation on rejection.
    await check_input(user_prompt, openai_client=get_moderation_client())

    # Options resolution and latency tracking setup.
    if opts is None:
        opts = GenerationOptions()

    t0 = time.perf_counter()

    model = opts.effective_model
    provider = opts.provider

    # Exact-match cache lookup.
    cache = get_estimation_cache()
    cache_key = EstimationCache.make_key(
        system_prompt=system_prompt,
        user_message=user_prompt,
        model=model,
        max_tokens=opts.max_tokens,
        reasoning_effort=opts.reasoning_effort,
        engine=opts.engine,
    )

    cached_result = await cache.get(cache_key)
    if cached_result is not None:
        cached_result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        cached_result["cache_hit"] = True
        if isinstance(cached_result.get("result"), dict):
            cached_result["result"] = EstimationResult.model_validate(cached_result["result"])
        log.info("estimation_cache_hit", model=model, latency_ms=cached_result["latency_ms"])
        return cached_result

    # LLM provider dispatch with structured output extraction.
    log.info(
        "estimation_generation_started",
        engine=opts.engine,
        provider=provider,
        model=model,
        max_tokens=opts.max_tokens,
        reasoning_effort=opts.reasoning_effort,
    )

    try:
        result = await call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            opts=opts,
        )

    except LLMProviderError:
        raise

    except Exception as e:
        log.exception(
            "estimation_generation_failed",
            provider=provider,
            model=model,
            error=repr(e),
            engine=opts.engine,
        )
        raise LLMProviderError(f"LLM call failed: {e}") from e

    # Response parsing and validation.
    raw_result = result.get("result")
    if isinstance(raw_result, dict):
        structured_result = EstimationResult.model_validate(raw_result)
    elif isinstance(raw_result, EstimationResult):
        structured_result = raw_result
    else:
        raise LLMProviderError(f"Unexpected result type from provider: {type(raw_result)}")

    # Output guardrail (filter): normalises low-confidence answers.
    structured_result = enforce_scope_response(structured_result)
    result["result"] = structured_result

    # Response telemetry, usage normalization, and latency calculation.
    result.setdefault("cost_usd", None)
    result["usage"].setdefault("preprocessing_input_tokens", 0)
    result["usage"].setdefault("preprocessing_output_tokens", 0)
    result["extracted_requirements"] = None
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["cache_hit"] = False

    log.info(
        "estimation_generation_completed",
        engine=opts.engine,
        provider=provider,
        model=model,
        latency_ms=result["latency_ms"],
        cost_usd=result.get("cost_usd"),
        usage=result.get("usage"),
        confidence_pct=structured_result.confidence_pct,
        total_cost_eur=structured_result.total_cost_eur,
        phases=len(structured_result.phases),
    )

    # Cache the validated payload only (never persist failed validations).
    cache_payload = {
        **result,
        "result": structured_result.model_dump(mode="json"),
    }
    await cache.set(cache_key, cache_payload)

    # Return estimation response payload.
    return result
