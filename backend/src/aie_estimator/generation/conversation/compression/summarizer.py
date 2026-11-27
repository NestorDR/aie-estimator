# src/aie_estimator/generation/conversation/compression/summarizer.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final

# --- Third Party Libraries ---
from pydantic import BaseModel, Field
import structlog

# --- App modules ---
from aie_estimator.foundation.prompts.loader import render_conversation_summary_prompt
from aie_estimator.foundation.llm import GenerationOptions, call_llm
from aie_estimator.generation.conversation.models import Message
from aie_estimator.utils.constants import DEFAULT_FAST_PROFILE

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_SUMMARY_MAX_CHARS: Final[int] = 4000
_SUMMARY_MAX_TOKENS: Final[int] = 800


class _SummaryEnvelope(BaseModel):
    """Structured wrapper holding the updated rolling conversation summary."""

    summary: str = Field(min_length=1, max_length=_SUMMARY_MAX_CHARS)


@dataclass(frozen=True, slots=True)
class SummarizerOutcome:
    """Result of one cumulative summarization pass with token and cost telemetry."""

    summary: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class CumulativeSummarizer:
    """Asynchronous rolling summarizer for evicted non-anchor conversation turns."""

    def __init__(self, *, model: str = DEFAULT_FAST_PROFILE) -> None:
        """
        Initialize the cumulative summarizer with the target model or profile.

        :param model: Logical profile or model identifier used for summarization calls.
        """
        self.model: Final[str] = model

    async def summarize(
        self,
        *,
        previous_summary: str | None,
        evicted: list[Message],
    ) -> SummarizerOutcome:
        """
        Fold ``evicted`` messages and ``previous_summary`` into an updated rolling summary.

        On any prompt, provider, or validation error, logs ``history_summary_failed`` and
        returns ``previous_summary`` unchanged so the conversational turn never fails.

        :param previous_summary: Existing rolling summary text, if any.
        :param evicted: Chronologically ordered messages peeled from the sliding window.

        :return: ``SummarizerOutcome`` with the resulting summary and LLM usage telemetry.
        """
        if not evicted:
            return SummarizerOutcome(summary=previous_summary)

        try:
            system_prompt, user_prompt = render_conversation_summary_prompt(
                previous_summary=previous_summary,
                evicted=evicted,
            )
            response = await call_llm(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                opts=GenerationOptions(model=self.model, max_tokens=_SUMMARY_MAX_TOKENS, reasoning_effort="low"),
                response_model=_SummaryEnvelope,
            )
            envelope = _SummaryEnvelope.model_validate(response.get("result"))
        except Exception as e:
            log.warning(
                "history_summary_failed",
                evicted_count=len(evicted),
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            return SummarizerOutcome(summary=previous_summary)

        usage = response.get("usage") or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost_usd = response.get("cost_usd")
        cleaned_summary = envelope.summary.strip()

        log.info(
            "history_summary_updated",
            evicted_count=len(evicted),
            summary_chars=len(cleaned_summary),
            model=response.get("model"),
            cost_usd=cost_usd,
        )
        return SummarizerOutcome(
            summary=cleaned_summary,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
