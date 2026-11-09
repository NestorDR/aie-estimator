# src/aie_estimator/guardrails/output.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    EstimationResult,
    Phase,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_NOT_ESTIMATED_PHASE: Final[Phase] = Phase(
    name="Not estimated",
    duration_weeks=1,
    cost_eur=0,
    summary="Cannot be sized without more information about scope, integrations and team.",
)


def enforce_scope_response(result: EstimationResult) -> EstimationResult:
    """
    Enforce filter policy on low-confidence estimation results.

    If confidence is below the threshold and summary does not begin with the out-of-scope prefix,
    rewrites the result to a standardized out-of-scope response.

    :param result: The validated EstimationResult to inspect.

    :return: Conforming EstimationResult with standardized out-of-scope indicators if confidence is low.
    """
    is_low_confidence = result.confidence_pct < LOW_CONFIDENCE_THRESHOLD
    already_marked = result.summary.startswith(OUT_OF_SCOPE_PREFIX)

    if not is_low_confidence or already_marked:
        return result

    log.info(
        "output_scope_filtered",
        confidence_pct=result.confidence_pct,
        original_summary_chars=len(result.summary),
    )
    new_summary = (
        f"{OUT_OF_SCOPE_PREFIX} not enough information to estimate confidently. "
        f"Original model rationale: {result.summary[:400]}"
    )
    return EstimationResult(
        summary=new_summary[:1200],
        confidence_pct=result.confidence_pct,
        phases=[_NOT_ESTIMATED_PHASE],
        total_duration_weeks=1,
        total_cost_eur=0,
    )
