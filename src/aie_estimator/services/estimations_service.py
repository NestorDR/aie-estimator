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
from aie_estimator.prompts.loader import (
    render_conversational_prompt,
    render_estimation_prompt,
    resolve_prompt_version,
)
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    ProjectType,
)
from aie_estimator.services.cache import EstimationExactCache, EstimationSemanticCache
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers import GenerationOptions, call_llm
from aie_estimator.sessions.metadata_extractor import update_metadata
from aie_estimator.sessions.models import Session
from aie_estimator.settings import get_settings

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

        result, structured_result = await self._dispatch_and_normalize(
            system_prompt=system_prompt,
            opts=opts,
            version=version,
            t0=t0,
            user_prompt=user_prompt,
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

    async def generate_session_estimation(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        opts: GenerationOptions | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute a multi-turn conversational estimation bypassing Tier 1 and Tier 2 caches.

        Executes input guardrails on ``transcript``, renders conversational templates with
        ``session.metadata``, dispatches prior ``session.history`` turns plus the new
        user message, enforces output scope filtering, appends the completed turn
        to ``session.history``, and refreshes ``session.metadata`` via non-fatal
        post-turn extraction while recording extractor tokens and cost.

        :param session: Active conversational ``Session`` holding history and distilled metadata.
        :param transcript: Current turn transcript (optionally enriched with attachment text).
        :param project_type: Target project domain category.
        :param detail_level: Desired breakdown depth.
        :param output_format: Desired presentation layout.
        :param opts: Optional generation options controlling model, engine, and token limits.
        :param prompt_version: Optional prompt version override; defaults to ``conversational_prompt_version``.

        :return: Response dictionary matching :meth:`generate_estimation` with ``cache_hit=False``.
        :raises InputGuardrailViolation: If input guardrails reject ``transcript``.
        :raises LLMProviderError: If any LLM provider invocation fails.
        :raises ValueError: If ``transcript`` is blank or ``prompt_version`` is invalid.
        """
        if not transcript or not transcript.strip():
            raise ValueError("transcript must be provided.")

        await check_input(transcript, openai_client=self.moderation_client)

        opts = opts or GenerationOptions()
        version = resolve_prompt_version(prompt_version or get_settings().conversational_prompt_version)
        t0 = time.perf_counter()

        system_prompt, user_prompt = render_conversational_prompt(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            metadata=session.metadata,
            version=version,
        )

        turns = [*session.history.to_messages(), {"role": "user", "content": user_prompt}]
        log.info(
            "estimation_session_request",
            session_id=session.session_id,
            history_messages=len(session.history.messages),
            metadata_is_empty=session.metadata.is_empty(),
            transcript_chars=len(transcript),
        )

        result, structured_result = await self._dispatch_and_normalize(
            system_prompt=system_prompt,
            opts=opts,
            version=version,
            t0=t0,
            user_prompt=None,
            messages=turns,
        )

        session.history.append(user=transcript, assistant=structured_result.model_dump_json())

        metadata_outcome = await update_metadata(
            previous=session.metadata,
            transcript=transcript,
            result=structured_result,
            opts=opts,
        )
        session.metadata = metadata_outcome.metadata
        result["usage"]["preprocessing_input_tokens"] = metadata_outcome.input_tokens
        result["usage"]["preprocessing_output_tokens"] = metadata_outcome.output_tokens
        if result.get("cost_usd") is not None and metadata_outcome.cost_usd is not None:
            result["cost_usd"] = round(float(result["cost_usd"]) + metadata_outcome.cost_usd, 6)
        result["latency_ms"] = int((time.perf_counter() - t0) * 1000)

        return result

    async def _dispatch_and_normalize(
        self,
        *,
        system_prompt: str,
        opts: GenerationOptions,
        version: str,
        t0: float,
        user_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[dict[str, Any], EstimationResult]:
        """
        Dispatch single-turn or multi-turn prompt payload to ``call_llm`` and normalize response telemetry.

        :param system_prompt: Rendered system instructions string.
        :param opts: Generation options for provider execution.
        :param version: Resolved prompt version identifier.
        :param t0: Request start timestamp from ``time.perf_counter()``.
        :param user_prompt: Single-turn user prompt string, or ``None`` when ``messages`` is provided.
        :param messages: Multi-turn sequence of non-system turn dictionaries.

        :return: Tuple of normalized response dictionary and scope-filtered ``EstimationResult``.
        :raises LLMProviderError: If provider execution fails or returns an invalid result type.
        """
        model = opts.effective_model
        provider = opts.provider

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
                messages=messages,
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

        return result, structured_result
