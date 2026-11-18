# src/aie_estimator/services/critic_service.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.prompts.loader import render_critic_prompt
from aie_estimator.schemas.acb import CriticFeedback
from aie_estimator.schemas.estimations import BaseTierEstimation
from aie_estimator.services.providers import GenerationOptions, call_llm
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.sessions.tier_resolver import Tier

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_CRITIC_MAX_TOKENS: Final[int] = 1500


@dataclass(frozen=True)
class CriticOutcome:
    """Result of a Critic review including structured feedback and LLM telemetry."""

    feedback: CriticFeedback
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class Critic:
    """Stateless asynchronous auditor that evaluates candidate estimations."""

    def __init__(self, *, model: str) -> None:
        """
        Initialize the Critic with its configured model or routing profile.

        :param model: Model identifier or logical profile name used for Critic calls.
        """
        self.model = model

    async def review(
        self,
        *,
        transcript: str,
        metadata: ProjectMetadata,
        tier: Tier,
        result: BaseTierEstimation,
        opts: GenerationOptions | None = None,
    ) -> CriticOutcome:
        """
        Audit a candidate estimation against transcript, metadata, and tier criteria.

        Degrades gracefully to a synthetic ``accept`` verdict with ``confidence_in_review=0``
        if the provider invocation or structured validation fails.

        :param transcript: Enriched user turn transcript under evaluation.
        :param metadata: Accumulated session ``ProjectMetadata``.
        :param tier: Resolved stakeholder ``Tier``.
        :param result: Candidate tier estimation produced by the Actor.
        :param opts: Optional generation options used to inherit provider engine mode.

        :return: ``CriticOutcome`` containing structured feedback and token/cost telemetry.
        """
        system_prompt, user_prompt = render_critic_prompt(
            transcript=transcript,
            metadata=metadata,
            tier=tier,
            result=result,
        )
        critic_opts = GenerationOptions(
            model=self.model,
            max_tokens=_CRITIC_MAX_TOKENS,
            engine=opts.engine if opts else "litellm",
        )

        try:
            response = await call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                opts=critic_opts,
                response_model=CriticFeedback,
            )
            feedback = CriticFeedback.model_validate(response.get("result"))
        except Exception as e:  # noqa: BLE001
            log.warning(
                "critic_review_failed",
                fallback="accept",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            return CriticOutcome(
                feedback=CriticFeedback(verdict="accept", issues=[], confidence_in_review=0),
            )

        usage = response.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost_usd = response.get("cost_usd")

        log.info(
            "critic_review_completed",
            verdict=feedback.verdict,
            issue_count=len(feedback.issues),
            confidence_in_review=feedback.confidence_in_review,
            model=response.get("model"),
            cost_usd=cost_usd,
        )
        return CriticOutcome(
            feedback=feedback,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
