# src/aie_estimator/services/boss_service.py

# --- Python Standard Library ---
from collections.abc import Awaitable, Callable
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.schemas.acb import ACBIteration, BossDecision, BossTrace, CriticFeedback
from aie_estimator.schemas.estimations import LOW_CONFIDENCE_THRESHOLD, BaseTierEstimation

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_MAX_ISSUE_SUMMARY_ITEMS: Final[int] = 5
_MAX_SUMMARY_CHARS: Final[int] = 1200

ActorCallable = Callable[[CriticFeedback | None], Awaitable[BaseTierEstimation]]
CriticCallable = Callable[[BaseTierEstimation], Awaitable[CriticFeedback]]


class Boss:
    """Deterministic orchestrator coordinating Actor generation and Critic auditing."""

    def __init__(self, *, max_iterations: int = 2) -> None:
        """
        Initialize the Boss orchestrator with a bounded iteration budget.

        :param max_iterations: Maximum number of Actor-Critic rounds (must be >= 1).
        :raises ValueError: If ``max_iterations`` is less than 1.
        """
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        self.max_iterations = max_iterations

    async def run(
        self,
        *,
        actor: ActorCallable,
        critic: CriticCallable,
    ) -> tuple[BaseTierEstimation, BossTrace]:
        """
        Execute the bounded Actor-Critic loop and return the final estimation with its audit trace.

        :param actor: Async callable producing a candidate estimation given optional prior ``CriticFeedback``.
        :param critic: Async callable auditing a candidate estimation and returning ``CriticFeedback``.

        :return: Tuple of ``(final_estimation, BossTrace)``.
        """
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None
        current: BaseTierEstimation | None = None

        for iteration in range(self.max_iterations):
            current = await actor(feedback)
            review = await critic(current)

            iterations_left = self.max_iterations - iteration - 1
            decision = self._decide(review, iterations_left=iterations_left)
            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[
                        f"[{issue.severity}] {issue.category} @ {issue.field_path}"
                        for issue in review.issues[:_MAX_ISSUE_SUMMARY_ITEMS]
                    ],
                )
            )
            trace.iterations_run = iteration + 1

            log.info(
                "boss_iteration_completed",
                iteration=iteration,
                decision=decision,
                verdict=review.verdict,
                issue_count=len(review.issues),
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return current, trace
            if decision == "synthesize":
                if review.verdict == "needs_iteration":
                    log.info("boss_iteration_budget_exhausted", max_iterations=self.max_iterations)
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(current, review), trace

            feedback = review

        raise AssertionError("Unreachable: Boss loop must terminate within max_iterations.")

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        """
        Map a Critic verdict and remaining budget to a deterministic Boss decision.

        :param review: Structured ``CriticFeedback`` from the latest audit.
        :param iterations_left: Remaining iteration budget after the current round.

        :return: Next ``BossDecision`` (``"accept"``, ``"iterate"``, or ``"synthesize"``).
        """
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "reject":
            return "synthesize"
        if iterations_left > 0:
            return "iterate"
        return "synthesize"

    @staticmethod
    def _synthesize_fallback(
        last_result: BaseTierEstimation,
        last_feedback: CriticFeedback,
    ) -> BaseTierEstimation:
        """
        Annotate the Actor's last draft with open Critic caveats and floor confidence at ``LOW_CONFIDENCE_THRESHOLD``.

        :param last_result: Most recent candidate estimation from the Actor.
        :param last_feedback: Unresolved ``CriticFeedback`` explaining open defects.

        :return: Copy of ``last_result`` with caveats prepended to ``summary`` and reduced ``confidence_pct``.
        """
        issue_lines = [
            f"- [{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
            for issue in last_feedback.issues
        ]
        caveats_block = (
            "Open caveats from independent review (loop did not fully converge):\n"
            + "\n".join(issue_lines)
            + "\n\n"
        )
        new_summary = (caveats_block + last_result.summary)[:_MAX_SUMMARY_CHARS]
        # Floor at LOW_CONFIDENCE_THRESHOLD so valid-with-caveats drafts do not trigger Out-of-scope prefix validators.
        reduced_confidence = max(LOW_CONFIDENCE_THRESHOLD, last_result.confidence_pct // 2)
        return last_result.model_copy(
            update={
                "summary": new_summary,
                "confidence_pct": reduced_confidence,
            }
        )
