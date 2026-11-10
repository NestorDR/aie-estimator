# src/aie_estimator/services/estimations_service.py

# --- Python Standard Library ---
import time
from typing import Any, Final

# --- Third Party Libraries ---
from openai import AsyncOpenAI
import structlog

# --- App modules ---
from aie_estimator.guardrails.input import check_input
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.prompts.loader import render_estimation_prompt, resolve_prompt_version
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResult
from aie_estimator.services.cache import EstimationExactCache, EstimationSemanticCache
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers import GenerationOptions, call_llm

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


class EstimationService:
    """
    Coordinate the end-to-end software project estimation workflow using multirole prompts.

    Evaluates Tier 1 exact-match hash cache and Tier 2 semantic vector cache before
    rendering prompt templates and dispatching to LLM providers.
    """

    def __init__(
        self,
        exact_cache: EstimationExactCache | None = None,
        semantic_cache: EstimationSemanticCache | None = None,
        moderation_client: AsyncOpenAI | None = None,
    ) -> None:
        """
        Initialize the EstimationService with caching and content moderation collaborators.

        :param exact_cache: Optional Tier 1 exact-match hash cache instance.
        :param semantic_cache: Optional Tier 2 semantic vector cache instance.
        :param moderation_client: Optional AsyncOpenAI client used for moderation guardrails.
        """
        self.exact_cache: Final[EstimationExactCache | None] = exact_cache
        self.semantic_cache: Final[EstimationSemanticCache | None] = semantic_cache
        self.moderation_client: Final[AsyncOpenAI | None] = moderation_client

    async def generate_estimation(
        self,
        request: EstimationRequest,
        opts: GenerationOptions | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate the end-to-end software project estimation workflow using multirole prompts.

        Evaluates Tier 1 exact-match hash cache and Tier 2 semantic vector cache before
        rendering prompt templates and dispatching to LLM providers.

        :param request: Validated EstimationRequest containing project description and preferences.
        :param opts: Optional generation options controlling model selection, engine, and token limits.
        :param prompt_version: Optional prompt version identifier; defaults to configured active version.

        :return: Response dictionary containing validated EstimationResult, model metadata, token usage, latency, prompt_version, and cache hit indicator.
        :raises InputGuardrailViolation: If input guardrails reject the prompt.
        :raises LLMProviderError: If any LLM provider invocation encounters an error.
        :raises ValueError: If request or request.description is empty.
        """
        if not request or not request.description or not request.description.strip():
            raise ValueError("request.description must be provided.")

        # Input guardrails — raises InputGuardrailViolation on rejection.
        await check_input(request.description, openai_client=self.moderation_client)

        # Options and prompt version resolution.
        if opts is None:
            opts = GenerationOptions(model=request.model, engine=request.engine)

        version = prompt_version or resolve_prompt_version()
        t0 = time.perf_counter()

        model = opts.effective_model
        provider = opts.provider

        # Tier 1: Exact-match cache lookup.
        exact_cache_key = None
        if self.exact_cache is not None:
            exact_cache_key = EstimationExactCache.make_key(
                request=request,
                prompt_version=version,
                opts=opts,
            )

            cached_result = await self.exact_cache.get(exact_cache_key)
            if cached_result is not None:
                cached_result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                cached_result["cache_hit"] = True
                cached_result["prompt_version"] = version
                if isinstance(cached_result.get("result"), dict):
                    cached_result["result"] = EstimationResult.model_validate(cached_result["result"])
                log.info("estimation_cache_hit", kind="exact", model=model, latency_ms=cached_result["latency_ms"])
                return cached_result

        # Tier 2: Semantic vector cache lookup (evaluated on exact-match miss).
        if self.semantic_cache is not None:
            semantic_hit = await self.semantic_cache.lookup(request, version)
            if semantic_hit is not None:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log.info("estimation_cache_hit", kind="semantic", model=model, latency_ms=latency_ms)
                return {
                    "result": semantic_hit,
                    "cache_hit": True,
                    "prompt_version": version,
                    "model": model,
                    "provider": provider,
                    "cost_usd": 0.0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "preprocessing_input_tokens": 0,
                        "preprocessing_output_tokens": 0,
                    },
                    "latency_ms": latency_ms,
                    "extracted_requirements": None,
                }

        # Lazy prompt template rendering (executed only on cache miss before LLM call).
        system_prompt, user_prompt = render_estimation_prompt(request, version=version)

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
        result["prompt_version"] = version

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

        # Dual-write: cache validated payload in both Tier 1 exact-match and Tier 2 semantic caches.
        cache_payload = {
            **result,
            "result": structured_result.model_dump(mode="json"),
        }
        if self.exact_cache is not None and exact_cache_key is not None:
            await self.exact_cache.set(exact_cache_key, cache_payload)

        if self.semantic_cache is not None:
            await self.semantic_cache.store(request, structured_result, version)

        # Return estimation response payload.
        return result
