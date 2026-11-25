# src/aie_estimator/services/estimations_service.py

# --- Python Standard Library ---
from dataclasses import dataclass
import time
from typing import Any, Final

# --- Third Party Libraries ---
from openai import AsyncOpenAI
from pydantic import ValidationError
import structlog

# --- App modules ---
from aie_estimator.attachments.extractor import (
    enrich_transcript,
    extract_text,
    validate_attachment_count,
)
from aie_estimator.guardrails.input import sanitize_attachment_input, sanitize_input
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.prompts.loader import (
    render_tier_estimation_prompt,
    resolve_prompt_version,
)
from aie_estimator.prompts.tier_config import TIER_CONFIG
from aie_estimator.schemas.acb import CriticFeedback
from aie_estimator.schemas.estimations import (
    BaseTierEstimation,
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from aie_estimator.services.boss_service import Boss
from aie_estimator.services.cache import EstimationExactCache, EstimationSemanticCache
from aie_estimator.services.critic_service import Critic, CriticOutcome
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.services.providers import GenerationOptions, call_llm
from aie_estimator.sessions.compression.policy import CompressionPolicy
from aie_estimator.sessions.metadata_extractor import update_metadata
from aie_estimator.sessions.models import Session, TurnObservation
from aie_estimator.sessions.tier_resolver import Tier, resolve_tier
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@dataclass(frozen=True)
class _PreparedSessionTurn:
    """Validated and enriched session turn context shared across estimation pipelines."""

    enriched_transcript: str
    attachment_count: int
    attachments_total_chars: int
    attachments_truncated_chars: int
    opts: GenerationOptions
    version: str
    tier: Tier
    tier_rule: str
    response_model: type[BaseTierEstimation]
    t0: float


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
        compression_policy: CompressionPolicy | None = None,
    ) -> None:
        """
        Initialize the EstimationService with caching, moderation, and compression collaborators.

        :param exact_cache: Optional Tier 1 exact-match hash cache instance.
        :param semantic_cache: Optional Tier 2 semantic vector cache instance.
        :param moderation_client: Optional AsyncOpenAI client used for moderation guardrails.
        :param compression_policy: Optional history compression policy for conversational turns.
        """
        self.exact_cache: Final[EstimationExactCache | None] = exact_cache
        self.semantic_cache: Final[EstimationSemanticCache | None] = semantic_cache
        self.moderation_client: Final[AsyncOpenAI | None] = moderation_client
        self.compression_policy: Final[CompressionPolicy] = (
            compression_policy or CompressionPolicy.from_settings(get_settings())
        )

    async def generate_estimation(
        self,
        request: EstimationRequest,
        opts: GenerationOptions | None = None,
        caller_tier: Tier | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate the end-to-end software project estimation workflow using tier-adaptive prompts.

        Evaluates Tier 1 exact-match hash cache and Tier 2 semantic vector cache before
        rendering prompt templates and dispatching to LLM providers.

        :param request: Validated EstimationRequest containing project description and preferences.
        :param opts: Optional generation options controlling model selection, engine, and token limits.
        :param caller_tier: Optional trusted stakeholder tier propagated from HTTP headers.

        :return: Response dictionary containing validated tier result, tier metadata, model metadata, token usage, latency, prompt_version, and cache hit indicator.
        :raises InputGuardrailViolation: If input guardrails reject the prompt.
        :raises LLMProviderError: If any LLM provider invocation encounters an error.
        :raises ValueError: If request or request.description is empty.
        """
        if not request or not request.description or not request.description.strip():
            raise ValueError("request.description must be provided.")

        # Input guardrails — raises InputGuardrailViolation on rejection.
        await sanitize_input(request.description, openai_client=self.moderation_client)

        # Options, prompt version, and stakeholder tier resolution.
        if opts is None:
            opts = GenerationOptions(model=request.model, engine=request.engine)

        version = resolve_prompt_version(get_settings().active_prompt_version)
        tier, tier_rule = resolve_tier(
            transcript=request.description,
            caller_tier=caller_tier,
            override=request.tier,
        )
        tier_config = TIER_CONFIG[tier]
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
                tier=tier,
            )

            cached_response = await self.exact_cache.get(exact_cache_key)
            if cached_response is not None:
                cached_response["latency_ms"] = int((time.perf_counter() - t0) * 1000)
                cached_response["cached"] = True
                cached_response["prompt_version"] = version
                cached_response["tier"] = tier
                cached_response["tier_rule"] = tier_rule
                if isinstance(cached_response.get("result"), dict):
                    cached_response["result"] = tier_config.schema.model_validate(cached_response["result"])
                log.info("estimation_cache_hit", kind="exact", tier=tier.value, model=model, latency_ms=cached_response["latency_ms"])
                return cached_response

        # Tier 2: Semantic vector cache lookup (evaluated on exact-match miss).
        if self.semantic_cache is not None:
            semantic_hit = await self.semantic_cache.lookup(
                request,
                version,
                tier=tier,
                response_model=tier_config.schema,
            )
            if semantic_hit is not None:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                log.info("estimation_cache_hit", kind="semantic", tier=tier.value, model=model, latency_ms=latency_ms)
                return {
                    "result": semantic_hit,
                    "tier": tier,
                    "tier_rule": tier_rule,
                    "cached": True,
                    "prompt_version": version,
                    "model": model,
                    "provider": provider,
                    "cost_usd": 0.0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                    "latency_ms": latency_ms,
                }

        # Lazy prompt template rendering (executed only on cache miss before LLM call).
        system_prompt, user_prompt = render_tier_estimation_prompt(
            description=request.description,
            project_type=request.project_type,
            detail_level=request.detail_level,
            output_format=request.output_format,
            tier=tier,
            version=version,
        )

        response = await self._dispatch_and_normalize(
            system_prompt=system_prompt,
            opts=opts,
            version=version,
            tier=tier,
            tier_rule=tier_rule,
            response_model=tier_config.schema,
            user_prompt=user_prompt,
        )

        # Dual-write: cache validated payload in both Tier 1 exact-match and Tier 2 semantic caches.
        structured_result: BaseTierEstimation = response["result"]
        cache_payload = {
            **response,
            "tier": tier.value,
            "result": structured_result.model_dump(mode="json"),
        }
        if self.exact_cache is not None and exact_cache_key is not None:
            await self.exact_cache.set(exact_cache_key, cache_payload)

        if self.semantic_cache is not None:
            await self.semantic_cache.store(request, structured_result, version, tier=tier)

        # Return estimation response payload.
        return response

    async def generate_session_estimation(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        attachments: list[tuple[str, bytes]] | None = None,
        opts: GenerationOptions | None = None,
        caller_tier: Tier | None = None,
        tier_override: Tier | None = None,
    ) -> dict[str, Any]:
        """
        Execute a multi-turn conversational estimation bypassing Tier 1 and Tier 2 caches.

        Validates attachment count, executes input guardrails on ``transcript``, extracts and
        sanitizes attachment text via ``extract_text`` and ``sanitize_attachment_input`` (with PII
        redaction), enriches the transcript inside ``<attachments>`` XML delimiters, renders
        conversational templates with ``session.metadata``, dispatches prior ``session.history``
        turns plus the new user message, enforces output scope filtering, appends the completed
        turn to ``session.history``, and refreshes ``session.metadata`` via non-fatal post-turn
        extraction while recording extractor tokens and cost.

        :param session: Active conversational ``Session`` holding history and distilled metadata.
        :param transcript: Current raw user turn transcript.
        :param project_type: Target project domain category.
        :param detail_level: Desired breakdown depth.
        :param output_format: Desired presentation layout.
        :param attachments: Optional list of ``(filename, content)`` raw attachment pairs.
        :param opts: Optional generation options controlling model, engine, and token limits.
        :param caller_tier: Optional trusted stakeholder tier from request headers.
        :param tier_override: Optional explicit stakeholder tier override.

        :return: Response dictionary matching :meth:`generate_estimation` with ``cached=False``.
        :raises AttachmentExtractionError: If attachment count, extension, byte size, or parsing fails.
        :raises InputGuardrailViolation: If input guardrails reject ``transcript`` or attachment content.
        :raises LLMProviderError: If any LLM provider invocation fails.
        :raises ValueError: If ``transcript`` is blank.
        """
        prepared = await self._prepare_session_turn(
            session=session,
            transcript=transcript,
            attachments=attachments,
            opts=opts,
            caller_tier=caller_tier,
            tier_override=tier_override,
        )

        log.info(
            "estimation_session_started",
            session_id=session.session_id,
            tier=prepared.tier.value,
            tier_rule=prepared.tier_rule,
            history_messages=len(session.history.messages),
            metadata_is_empty=session.metadata.is_empty(),
            transcript_chars=len(transcript),
            enriched_transcript_chars=len(prepared.enriched_transcript),
            attachment_count=prepared.attachment_count,
        )

        response = await self._dispatch_session_actor_turn(
            session=session,
            prepared=prepared,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )

        await self._finalize_session_turn(
            session=session,
            prepared=prepared,
            response=response,
        )
        log.info(
            "estimation_session_completed",
            session_id=session.session_id,
            tier=prepared.tier.value,
        )
        return response

    async def generate_session_estimation_acb(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        attachments: list[tuple[str, bytes]] | None = None,
        opts: GenerationOptions | None = None,
        caller_tier: Tier | None = None,
        tier_override: Tier | None = None,
        critic: Critic | None = None,
        boss: Boss | None = None,
    ) -> dict[str, Any]:
        """
        Execute a bounded Actor-Critic-Boss conversational estimation turn.

        Reuses turn sanitization, attachment enrichment, and tier resolution, then delegates
        iterative Actor generation and Critic auditing to ``Boss.run(...)`` before appending
        only the final accepted or synthesized result to ``session.history`` and refreshing
        ``session.metadata`` once.

        :param session: Active conversational ``Session`` holding history and distilled metadata.
        :param transcript: Current raw user turn transcript.
        :param project_type: Target project domain category.
        :param detail_level: Desired breakdown depth.
        :param output_format: Desired presentation layout.
        :param attachments: Optional list of ``(filename, content)`` raw attachment pairs.
        :param opts: Optional generation options controlling model, engine, and token limits.
        :param caller_tier: Optional trusted stakeholder tier from request headers.
        :param tier_override: Optional explicit stakeholder tier override.
        :param critic: Optional injected ``Critic`` service (defaults from settings).
        :param boss: Optional injected ``Boss`` orchestrator (defaults from settings).

        :return: Normalized response dictionary including ``boss_trace`` and aggregated telemetry.
        :raises AttachmentExtractionError: If attachment count, extension, byte size, or parsing fails.
        :raises InputGuardrailViolation: If input guardrails reject ``transcript`` or attachment content.
        :raises LLMProviderError: If any Actor LLM provider invocation fails.
        :raises ValueError: If ``transcript`` is blank.
        """
        prepared = await self._prepare_session_turn(
            session=session,
            transcript=transcript,
            attachments=attachments,
            opts=opts,
            caller_tier=caller_tier,
            tier_override=tier_override,
        )
        settings = get_settings()
        effective_critic = critic or Critic(model=settings.acb_critic_model)
        effective_boss = boss or Boss(max_iterations=settings.acb_max_iterations)

        log.info(
            "estimation_acb_started",
            session_id=session.session_id,
            tier=prepared.tier.value,
            tier_rule=prepared.tier_rule,
            max_iterations=effective_boss.max_iterations,
            critic_model=effective_critic.model,
            history_messages=len(session.history.messages),
            attachment_count=prepared.attachment_count,
        )

        actor_responses: list[dict[str, Any]] = []
        critic_outcomes: list[CriticOutcome] = []

        async def _actor(feedback: CriticFeedback | None) -> BaseTierEstimation:
            """
            Execute one Actor generation pass with optional Critic revision feedback.

            :param feedback: Optional structured ``CriticFeedback`` from the previous round.

            :return: Validated and scope-filtered tier estimation candidate.
            """
            actor_response = await self._dispatch_session_actor_turn(
                session=session,
                prepared=prepared,
                project_type=project_type,
                detail_level=detail_level,
                output_format=output_format,
                critic_feedback=feedback,
            )
            actor_responses.append(actor_response)
            return actor_response["result"]

        async def _critic(candidate: BaseTierEstimation) -> CriticFeedback:
            """
            Audit one Actor candidate estimation and record Critic telemetry.

            :param candidate: Structured tier estimation candidate to audit.

            :return: Structured ``CriticFeedback`` verdict for Boss governance.
            """
            outcome = await effective_critic.review(
                transcript=prepared.enriched_transcript,
                metadata=session.metadata,
                tier=prepared.tier,
                result=candidate,
                opts=prepared.opts,
            )
            critic_outcomes.append(outcome)
            return outcome.feedback

        final_structured, boss_trace = await effective_boss.run(actor=_actor, critic=_critic)

        aggregated_usage, aggregated_cost = self._aggregate_acb_telemetry(
            actor_results=actor_responses,
            critic_outcomes=critic_outcomes,
        )
        final_response = dict(actor_responses[-1])
        final_response["result"] = final_structured
        final_response["usage"] = aggregated_usage
        final_response["cost_usd"] = aggregated_cost
        final_response["boss_trace"] = boss_trace

        await self._finalize_session_turn(
            session=session,
            prepared=prepared,
            response=final_response,
        )
        log.info(
            "estimation_acb_completed",
            session_id=session.session_id,
            tier=prepared.tier.value,
            final_decision=boss_trace.final_decision,
            iterations_run=boss_trace.iterations_run,
        )
        return final_response

    async def _dispatch_session_actor_turn(
        self,
        *,
        session: Session,
        prepared: _PreparedSessionTurn,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        critic_feedback: CriticFeedback | None = None,
    ) -> dict[str, Any]:
        """
        Render tier-adaptive conversational prompts and dispatch one Actor turn.

        :param session: Active conversational ``Session`` holding history and metadata.
        :param prepared: Prepared turn context with enriched transcript and resolved tier.
        :param project_type: Target project domain category.
        :param detail_level: Desired breakdown depth.
        :param output_format: Desired presentation layout.
        :param critic_feedback: Optional ``CriticFeedback`` for Actor revision passes.

        :return: Normalized response dictionary containing scope-filtered tier estimate.
        :raises LLMProviderError: If provider execution fails or returns an unexpected type.
        """
        system_prompt, user_prompt = render_tier_estimation_prompt(
            description=prepared.enriched_transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            tier=prepared.tier,
            metadata=session.metadata,
            conversation_summary=session.history.summary,
            anchored_turns=session.history.anchors,
            version=prepared.version,
            critic_feedback=critic_feedback,
        )
        turns = [*session.history.to_messages(), {"role": "user", "content": user_prompt}]
        return await self._dispatch_and_normalize(
            system_prompt=system_prompt,
            opts=prepared.opts,
            version=prepared.version,
            tier=prepared.tier,
            tier_rule=prepared.tier_rule,
            response_model=prepared.response_model,
            user_prompt=None,
            messages=turns,
        )

    async def _prepare_session_turn(
        self,
        *,
        session: Session,
        transcript: str,
        attachments: list[tuple[str, bytes]] | None,
        opts: GenerationOptions | None,
        caller_tier: Tier | None,
        tier_override: Tier | None,
    ) -> _PreparedSessionTurn:
        """
        Validate inputs, sanitize attachments, enrich transcript, and resolve tier.

        :param session: Active conversational ``Session`` holding history and metadata.
        :param transcript: Current raw user turn transcript.
        :param attachments: Optional list of ``(filename, content)`` raw attachment pairs.
        :param opts: Optional generation options controlling model, engine, and token limits.
        :param caller_tier: Optional trusted stakeholder tier from request headers.
        :param tier_override: Optional explicit stakeholder tier override.

        :return: Prepared turn context ready for prompt rendering and LLM dispatch.
        :raises ValueError: If ``transcript`` is blank.
        :raises AttachmentExtractionError: If attachment count, extension, byte size, or parsing fails.
        :raises InputGuardrailViolation: If input guardrails reject ``transcript`` or attachment content.
        """
        if not transcript or not transcript.strip():
            raise ValueError("transcript must be provided.")

        raw_attachments = attachments or []
        validate_attachment_count(len(raw_attachments))
        await sanitize_input(transcript, openai_client=self.moderation_client)

        extracted_pairs: list[tuple[str, str]] = []
        extract_truncated_chars = 0
        for filename, content in raw_attachments:
            extracted_text, file_truncated_chars = extract_text(filename=filename, content=content)
            extract_truncated_chars += file_truncated_chars
            sanitized_text = await sanitize_attachment_input(
                extracted_text,
                openai_client=self.moderation_client,
            )
            if sanitized_text:
                extracted_pairs.append((filename, sanitized_text))

        enriched_transcript, attachments_total_chars, turn_truncated_chars = enrich_transcript(
            transcript=transcript,
            attachments=extracted_pairs,
        )

        effective_opts = opts or GenerationOptions()
        version = resolve_prompt_version(get_settings().conversational_prompt_version)
        tier, tier_rule = resolve_tier(
            transcript=enriched_transcript,
            metadata=session.metadata,
            caller_tier=caller_tier,
            override=tier_override,
        )
        return _PreparedSessionTurn(
            enriched_transcript=enriched_transcript,
            attachment_count=len(extracted_pairs),
            attachments_total_chars=attachments_total_chars,
            attachments_truncated_chars=extract_truncated_chars + turn_truncated_chars,
            opts=effective_opts,
            version=version,
            tier=tier,
            tier_rule=tier_rule,
            response_model=TIER_CONFIG[tier].schema,
            t0=time.perf_counter(),
        )

    async def _finalize_session_turn(
        self,
        *,
        session: Session,
        prepared: _PreparedSessionTurn,
        response: dict[str, Any],
    ) -> None:
        """
        Persist turn in session history, run post-turn memory updates, and emit turn telemetry.

        :param session: Active conversational ``Session`` to update.
        :param prepared: Prepared turn context with enriched transcript and timing baseline.
        :param response: Read-only normalized generation response dictionary.
        """
        structured_result: BaseTierEstimation = response["result"]
        session.turn_index += 1
        session.last_tier = prepared.tier.value
        session.last_tier_rule = prepared.tier_rule
        session.history.append(user=prepared.enriched_transcript, assistant=structured_result.model_dump_json())

        compression_outcome = await self.compression_policy.apply(session.history)
        metadata_outcome = await update_metadata(
            previous=session.metadata,
            transcript=prepared.enriched_transcript,
            result=structured_result,
            opts=prepared.opts,
        )
        session.metadata = metadata_outcome.metadata
        
        preprocessing_input_tokens = metadata_outcome.input_tokens + compression_outcome.input_tokens
        preprocessing_output_tokens = metadata_outcome.output_tokens + compression_outcome.output_tokens

        base_cost = response.get("cost_usd")
        extra_costs = [cost for cost in (metadata_outcome.cost_usd, compression_outcome.cost_usd) if cost is not None]
        turn_cost_usd = round(base_cost + sum(extra_costs), 6) if base_cost is not None else None

        usage = response["usage"]
        observation = TurnObservation(
            turn_index=session.turn_index,
            session_id=session.session_id,
            enriched_transcript_chars=len(prepared.enriched_transcript),
            attachments_total_chars=prepared.attachments_total_chars,
            attachments_truncated_chars=prepared.attachments_truncated_chars,
            messages_in_window=len(session.history.messages),
            anchors_count=len(session.history.anchors),
            summary_chars=len(session.history.summary) if session.history.summary else 0,
            tokens_in=usage.get("input_tokens", 0) + preprocessing_input_tokens,
            tokens_out=usage.get("output_tokens", 0) + preprocessing_output_tokens,
            cost_usd=turn_cost_usd,
            latency_ms=int((time.perf_counter() - prepared.t0) * 1000),
            cache_hit_kind="none",
            last_resolved_tier=prepared.tier.value,
        )
        session.last_turn_observed = observation
        log.info("session_turn_observed", **observation.model_dump())

    @staticmethod
    def _aggregate_acb_telemetry(
        *,
        actor_results: list[dict[str, Any]],
        critic_outcomes: list[CriticOutcome],
    ) -> tuple[dict[str, int], float | None]:
        """
        Sum token usage and USD costs across Actor and Critic calls, preserving ``None`` if any Actor call is unpriced.

        :param actor_results: Response dictionaries from every Actor invocation in the loop.
        :param critic_outcomes: ``CriticOutcome`` records from every Critic invocation in the loop.

        :return: Tuple of aggregated ``usage`` dictionary and total ``cost_usd`` (or ``None``).
        """
        input_tokens = sum(res["usage"].get("input_tokens", 0) for res in actor_results)
        output_tokens = sum(res["usage"].get("output_tokens", 0) for res in actor_results)
        input_tokens += sum(outcome.input_tokens for outcome in critic_outcomes)
        output_tokens += sum(outcome.output_tokens for outcome in critic_outcomes)

        if any(res.get("cost_usd") is None for res in actor_results):
            total_cost = None
        else:
            costs = [res["cost_usd"] for res in actor_results]
            costs.extend(outcome.cost_usd for outcome in critic_outcomes if outcome.cost_usd is not None)
            total_cost = round(sum(costs), 6) if costs else None

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return usage, total_cost

    async def _dispatch_and_normalize(
        self,
        *,
        system_prompt: str,
        opts: GenerationOptions,
        version: str,
        tier: Tier,
        tier_rule: str,
        response_model: type[BaseTierEstimation],
        user_prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Dispatch single-turn or multi-turn prompt payload to ``call_llm`` and normalize response telemetry.

        :param system_prompt: Rendered system instructions string.
        :param opts: Generation options for provider execution.
        :param version: Resolved prompt version identifier.
        :param tier: Resolved stakeholder tier.
        :param tier_rule: Tier resolution rule identifier.
        :param response_model: Target Pydantic schema class for structured output.
        :param user_prompt: Single-turn user prompt string, or ``None`` when ``messages`` is passed.
        :param messages: Multi-turn sequence of non-system turn dictionaries.

        :return: Normalized response dictionary containing scope-filtered tier estimate.
        :raises LLMProviderError: If provider execution fails or returns an invalid result type.
        """
        model = opts.effective_model
        provider = opts.provider
        call_t0 = time.perf_counter()

        log.info(
            "estimation_generation_started",
            engine=opts.engine,
            provider=provider,
            model=model,
            tier=tier.value,
            tier_rule=tier_rule,
            max_tokens=opts.max_tokens,
            reasoning_effort=opts.reasoning_effort,
        )

        try:
            response = await call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                opts=opts,
                messages=messages,
                response_model=response_model,
            )

        except LLMProviderError:
            raise

        except Exception as e:
            log.exception(
                "estimation_generation_failed",
                provider=provider,
                model=model,
                tier=tier.value,
                error=repr(e),
                engine=opts.engine,
            )
            raise LLMProviderError(f"LLM call failed: {e}") from e

        raw_result = response.get("result")
        if isinstance(raw_result, response_model):
            structured_result = raw_result
        else:
            try:
                structured_result = response_model.model_validate(raw_result)
            except ValidationError as e:
                raise LLMProviderError(f"Unexpected result type from provider: {type(raw_result)}") from e

        # Output guardrail (filter): normalises low-confidence answers.
        structured_result = enforce_scope_response(structured_result)
        response["result"] = structured_result
        response["tier"] = tier
        response["tier_rule"] = tier_rule
        response["prompt_version"] = version
        response["latency_ms"] = int((time.perf_counter() - call_t0) * 1000)
        response["cached"] = False

        log.info(
            "estimation_generation_completed",
            engine=opts.engine,
            provider=provider,
            model=model,
            tier=tier.value,
            tier_rule=tier_rule,
            schema=response_model.__name__,
            latency_ms=response["latency_ms"],
            cost_usd=response.get("cost_usd"),
            usage=response.get("usage"),
            confidence_pct=structured_result.confidence_pct,
        )

        return response
