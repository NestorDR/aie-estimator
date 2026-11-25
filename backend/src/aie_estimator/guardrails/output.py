# src/aie_estimator/guardrails/output.py

# --- Python Standard Library ---
from typing import Final, TypeVar

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import (
    LOW_CONFIDENCE_THRESHOLD,
    OUT_OF_SCOPE_PREFIX,
    BaseTierEstimation,
    ComponentEstimation,
    CostRange,
    DeveloperEstimation,
    DurationRange,
    ExecutiveEstimation,
    HoursRange,
    Phase,
    PmEstimation,
    TeamComposition,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

TResult = TypeVar("TResult", bound=BaseTierEstimation)

_NOT_ESTIMATED_PHASE: Final[Phase] = Phase(
    name="Not estimated",
    duration_weeks=1,
    cost_eur=0,
    summary="Cannot be sized without more information about scope, integrations and team.",
)
_NOT_ESTIMATED_COMPONENT: Final[ComponentEstimation] = ComponentEstimation(
    name="Not estimated",
    estimated_hours=0,
    technical_risks=["Insufficient specification"],
    stack_assumptions=["Cannot determine stack without additional context"],
)


def _build_out_of_scope_developer(summary: str, confidence_pct: int) -> DeveloperEstimation:
    return DeveloperEstimation(
        summary=summary,
        confidence_pct=confidence_pct,
        components=[_NOT_ESTIMATED_COMPONENT],
        technical_risks=["Insufficient scope definition"],
        stack_assumptions=[],
        uncertainty_drivers=["Core requirements are unspecified"],
        total_hours_range=HoursRange(min_hours=0, max_hours=0),
    )


def _build_out_of_scope_pm(summary: str, confidence_pct: int) -> PmEstimation:
    return PmEstimation(
        summary=summary,
        confidence_pct=confidence_pct,
        phases=[_NOT_ESTIMATED_PHASE],
        milestones=[],
        team_composition=TeamComposition(developers=0, designers=0, qa_engineers=0, tech_lead=False),
        blockers=["Clarify project scope and requirements"],
        total_duration_weeks=1,
        total_cost_eur=0,
    )


def _build_out_of_scope_executive(summary: str, confidence_pct: int) -> ExecutiveEstimation:
    return ExecutiveEstimation(
        summary=summary,
        confidence_pct=confidence_pct,
        headline_cost_range=CostRange(min_eur=0, max_eur=0),
        headline_duration_range=DurationRange(min_weeks=1, max_weeks=1),
        confidence_level="low",
        top_three_risks=["Insufficient scope to produce a defensible commercial estimate"],
        go_no_go_recommendation="Hold: gather missing scope details before committing budget or timeline.",
    )


def enforce_scope_response(result: TResult) -> TResult:
    """
    Enforce filter policy on low-confidence estimation results across all tier schemas.

    If confidence is below the threshold and summary does not begin with the out-of-scope prefix,
    rewrites the result into a standardized out-of-scope instance of the exact same schema.

    :param result: The validated estimation result to inspect.

    :return: Conforming result with standardized out-of-scope indicators if confidence is low.
    """
    is_low_confidence = result.confidence_pct < LOW_CONFIDENCE_THRESHOLD
    already_marked = result.summary.startswith(OUT_OF_SCOPE_PREFIX)

    if not is_low_confidence or already_marked:
        return result

    log.info(
        "output_scope_filtered",
        schema=type(result).__name__,
        confidence_pct=result.confidence_pct,
        original_summary_chars=len(result.summary),
    )
    new_summary = (
        f"{OUT_OF_SCOPE_PREFIX} not enough information to estimate confidently. "
        f"Original model rationale: {result.summary[:400]}"
    )[:1200]

    if isinstance(result, DeveloperEstimation):
        return _build_out_of_scope_developer(new_summary, result.confidence_pct)
    if isinstance(result, PmEstimation):
        return _build_out_of_scope_pm(new_summary, result.confidence_pct)
    if isinstance(result, ExecutiveEstimation):
        return _build_out_of_scope_executive(new_summary, result.confidence_pct)
    raise TypeError(f"Unsupported estimation schema: {type(result).__name__}")
