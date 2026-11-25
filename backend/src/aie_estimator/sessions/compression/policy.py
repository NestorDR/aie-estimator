# src/aie_estimator/sessions/compression/policy.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.sessions.compression.anchors import AnchorDetector
from aie_estimator.sessions.compression.summarizer import CumulativeSummarizer
from aie_estimator.sessions.models import ConversationHistory, Message

if TYPE_CHECKING:
    from aie_estimator.settings import Settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class CompressionOutcome:
    """Token and cost telemetry from one post-turn compression pass."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class CompressionPolicy:
    """
    Hybrid history compression policy combining a sliding window, bounded anchors, and rolling summary.

    ``CompressionPolicy`` is the sole owner of overflow eviction on ``ConversationHistory``.
    """

    def __init__(
        self,
        *,
        anchor_detector: AnchorDetector,
        summarizer: CumulativeSummarizer,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the compression policy with its anchor detector and cumulative summarizer.

        :param anchor_detector: Detector classifying overflow turns for verbatim anchor retention.
        :param summarizer: Asynchronous summarizer folding non-anchor overflow into ``history.summary``.
        :param enabled: When ``False``, falls back to plain sliding-window trimming without anchors or LLM calls.
        """
        self.anchor_detector: Final[AnchorDetector] = anchor_detector
        self.summarizer: Final[CumulativeSummarizer] = summarizer
        self.enabled: Final[bool] = enabled

    @classmethod
    def from_settings(cls, settings: "Settings") -> "CompressionPolicy":
        """
        Construct a ``CompressionPolicy`` instance from application ``Settings``.

        :param settings: Application configuration settings.

        :return: Configured ``CompressionPolicy`` instance.
        """
        return cls(
            anchor_detector=AnchorDetector(
                strategy=settings.anchor_strategy,
                model=settings.compression_model,
            ),
            summarizer=CumulativeSummarizer(model=settings.compression_model),
            enabled=settings.enable_history_compression,
        )

    async def apply(self, history: ConversationHistory) -> CompressionOutcome:
        """
        Mutate ``history`` in place when over capacity: promote anchors and summarize remaining overflow.

        Idempotent when ``history`` is already within ``max_turns``.

        :param history: Mutable ``ConversationHistory`` belonging to an active session.

        :return: ``CompressionOutcome`` carrying aggregated LLM token and cost telemetry.
        """
        max_messages = history.max_turns * 2
        if len(history.messages) <= max_messages:
            log.debug("history_compression_skipped", reason="within_capacity", message_count=len(history.messages))
            return CompressionOutcome()

        overflow_count = len(history.messages) - max_messages
        if not self.enabled:
            del history.messages[:overflow_count]
            log.debug("history_compression_skipped", reason="disabled", trimmed_messages=overflow_count)
            return CompressionOutcome()

        evicted_for_summary: list[Message] = []
        evicted_pairs = 0
        promoted_anchors = 0
        overflow_anchors = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost: float | None = None

        while len(history.messages) > max_messages and len(history.messages) >= 2:
            user_msg = history.messages.pop(0)
            assistant_msg = history.messages.pop(0)
            evicted_pairs += 1

            match = await self.anchor_detector.detect(user_msg)
            total_input_tokens += match.input_tokens
            total_output_tokens += match.output_tokens
            if match.cost_usd is not None:
                total_cost = (total_cost or 0.0) + match.cost_usd

            if match.is_anchor and history.max_anchor_pairs > 0:
                # The configuration allows anchors, so promote this pair of anchors
                history.anchors.extend((user_msg, assistant_msg))
                promoted_anchors += 1
                log.info(
                    "history_anchor_promoted",
                    matched_rules=match.matched_rules,
                    anchors_count=len(history.anchors),
                )
                # If the max anchor pairs threshold is exceeded, discard the oldest ones and send them to the summary
                while len(history.anchors) > history.max_anchor_pairs * 2 and len(history.anchors) >= 2:
                    evicted_for_summary.append(history.anchors.pop(0))
                    evicted_for_summary.append(history.anchors.pop(0))
                    overflow_anchors += 1
            else:
                evicted_for_summary.extend((user_msg, assistant_msg))

        if evicted_for_summary:
            # Summarize all the overflow turns that are not anchors
            summary_outcome = await self.summarizer.summarize(
                previous_summary=history.summary,
                evicted=evicted_for_summary,
            )
            history.summary = summary_outcome.summary
            total_input_tokens += summary_outcome.input_tokens
            total_output_tokens += summary_outcome.output_tokens
            if summary_outcome.cost_usd is not None:
                total_cost = (total_cost or 0.0) + summary_outcome.cost_usd

        log.info(
            "history_compression_completed",
            evicted_pairs=evicted_pairs,
            promoted_anchors=promoted_anchors,
            overflow_anchors_summarized=overflow_anchors,
            has_summary=bool(history.summary),
            anchors_count=len(history.anchors),
        )
        return CompressionOutcome(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=total_cost,
        )
