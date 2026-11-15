# evals/metrics.py

# --- Python Standard Library ---
from dataclasses import dataclass
import statistics
from typing import Final

# --- App modules ---
from aie_estimator.schemas.estimations import (
    LOW_CONFIDENCE_THRESHOLD,
    DeveloperEstimation,
    EstimationResponse,
    ExecutiveEstimation,
    PmEstimation,
    TierEstimationResult,
)
from evals.dataset import GoldenCase

DEFAULT_COMPONENT_RECALL_THRESHOLD: Final[float] = 0.5
DEFAULT_CV_THRESHOLD: Final[float] = 0.25
_MAX_EXECUTIVE_RISKS: Final[int] = 3


@dataclass(frozen=True)
class MetricResult:
    """Structured outcome of a single evaluation metric."""

    name: str
    score: float
    passed: bool
    details: str


@dataclass(frozen=True)
class _TierMetricsView:
    """Normalized scalar projections extracted from any tier estimation result."""

    scalar_anchor: float
    cost_eur: float | None
    duration_weeks: float | None
    is_zero_envelope: bool


def _tier_metrics_view(result: TierEstimationResult) -> _TierMetricsView:
    """
    Project tier-specific estimation fields into a uniform view for Family 1 and Family 2 checks.

    :param result: Tier-specific estimation payload.

    :return: Normalized ``_TierMetricsView`` instance.
    """
    if isinstance(result, DeveloperEstimation):
        mid_hours = (result.total_hours_range.min_hours + result.total_hours_range.max_hours) / 2.0
        return _TierMetricsView(mid_hours, None, None, result.total_hours_range.max_hours == 0)
    if isinstance(result, PmEstimation):
        return _TierMetricsView(float(result.total_cost_eur), float(result.total_cost_eur), float(result.total_duration_weeks), result.total_cost_eur == 0)
    cost_mid = (result.headline_cost_range.min_eur + result.headline_cost_range.max_eur) / 2.0
    weeks_mid = (result.headline_duration_range.min_weeks + result.headline_duration_range.max_weeks) / 2.0
    return _TierMetricsView(cost_mid, cost_mid, weeks_mid, result.headline_cost_range.max_eur == 0)


def evaluate_schema_and_tier(case: GoldenCase, response: EstimationResponse) -> MetricResult:
    """
    Verify Family 1 tier resolution accuracy, out-of-scope confidence, and eval-specific shape constraints.

    :param case: Golden Dataset scenario.
    :param response: API estimation response.

    :return: Deterministic ``MetricResult`` for ``schema_and_tier_adherence``.
    """
    problems: list[str] = []
    result = response.result

    if case.expected_tier is not None and response.tier != case.expected_tier:
        problems.append(f"expected tier '{case.expected_tier.value}', got '{response.tier.value}'")

    if case.expected_out_of_scope and result.confidence_pct >= LOW_CONFIDENCE_THRESHOLD:
        problems.append(f"expected out-of-scope confidence < {LOW_CONFIDENCE_THRESHOLD}, got {result.confidence_pct}")

    if isinstance(result, PmEstimation) and case.expected_phase_count_range is not None:
        lo, hi = case.expected_phase_count_range
        if not (lo <= len(result.phases) <= hi):
            problems.append(f"phase count {len(result.phases)} outside [{lo}, {hi}]")
    elif isinstance(result, ExecutiveEstimation) and not case.expected_out_of_scope:
        if not (1 <= len(result.top_three_risks) <= _MAX_EXECUTIVE_RISKS):
            problems.append(f"expected 1-{_MAX_EXECUTIVE_RISKS} top_three_risks, got {len(result.top_three_risks)}")

    return MetricResult("schema_and_tier_adherence", 1.0 if not problems else 0.0, not problems, "; ".join(problems) or "schema and tier invariants ok")


def evaluate_numeric_bounds(case: GoldenCase, response: EstimationResponse) -> MetricResult:
    """
    Verify Family 1 effort, cost, duration, and confidence bounds against ``GoldenCase`` expectations.

    :param case: Golden Dataset scenario.
    :param response: API estimation response.

    :return: Deterministic ``MetricResult`` for ``numeric_bounds``.
    """
    problems: list[str] = []
    result = response.result
    view = _tier_metrics_view(result)

    if case.expected_out_of_scope:
        if not view.is_zero_envelope:
            problems.append("out-of-scope response has non-zero cost or effort")
    else:
        if case.min_confidence_pct is not None and result.confidence_pct < case.min_confidence_pct:
            problems.append(f"confidence_pct {result.confidence_pct} < min {case.min_confidence_pct}")
        if case.expected_hours_range is not None and isinstance(result, DeveloperEstimation):
            lo, hi = case.expected_hours_range
            if not (lo <= view.scalar_anchor <= hi):
                problems.append(f"hours midpoint {view.scalar_anchor:.1f} outside [{lo}, {hi}]")
        if case.expected_cost_range_eur is not None and view.cost_eur is not None:
            lo, hi = case.expected_cost_range_eur
            if not (lo <= view.cost_eur <= hi):
                problems.append(f"cost {view.cost_eur:.0f} EUR outside [{lo}, {hi}]")
        if case.expected_duration_weeks_range is not None and view.duration_weeks is not None:
            lo, hi = case.expected_duration_weeks_range
            if not (lo <= view.duration_weeks <= hi):
                problems.append(f"duration {view.duration_weeks:.1f} weeks outside [{lo}, {hi}]")

    return MetricResult("numeric_bounds", 1.0 if not problems else 0.0, not problems, "; ".join(problems) or "numeric bounds ok")


def evaluate_component_recall(
    case: GoldenCase,
    response: EstimationResponse,
    *,
    min_recall: float = DEFAULT_COMPONENT_RECALL_THRESHOLD,
) -> MetricResult:
    """
    Compute Family 1 case-insensitive component recall and technology hit across the serialized payload.

    :param case: Golden Dataset scenario.
    :param response: API estimation response.
    :param min_recall: Minimum fractional recall required to pass.

    :return: ``MetricResult`` with fractional recall score in ``[0.0, 1.0]``.
    """
    haystack = response.result.model_dump_json().lower()
    required = case.required_components_any_of
    recall = 1.0 if not required else sum(1 for term in required if term.lower() in haystack) / len(required)
    missing = [term for term in required if term.lower() not in haystack]
    tech_ok = not case.expected_technologies_any_of or any(tech.lower() in haystack for tech in case.expected_technologies_any_of)

    passed = recall >= min_recall and tech_ok
    notes: list[str] = [f"recall={len(required) - len(missing)}/{len(required)}"] if required else []
    if missing:
        notes.append(f"missing={missing}")
    if not tech_ok:
        notes.append(f"no tech hit from {case.expected_technologies_any_of}")

    return MetricResult("component_recall", round(recall if tech_ok else recall * 0.5, 4), passed, "; ".join(notes) or "content ok")


def run_family1_metrics(case: GoldenCase, response: EstimationResponse) -> list[MetricResult]:
    """
    Execute all Family 1 deterministic invariant metrics for a single response.

    :param case: Golden Dataset scenario.
    :param response: API estimation response.

    :return: List of Family 1 ``MetricResult`` instances.
    """
    return [
        evaluate_schema_and_tier(case, response),
        evaluate_numeric_bounds(case, response),
        evaluate_component_recall(case, response),
    ]


def evaluate_consistency_cv(
    responses: list[EstimationResponse],
    *,
    max_cv: float = DEFAULT_CV_THRESHOLD,
) -> MetricResult:
    """
    Evaluate Family 2 statistical consistency across multiple runs of the same scenario.

    :param responses: Repeated ``EstimationResponse`` outputs for one ``GoldenCase``.
    :param max_cv: Maximum allowed Coefficient of Variation (``stdev / mean``).

    :return: ``MetricResult`` for ``consistency_cv``.
    """
    if len(responses) < 2:
        return MetricResult("consistency_cv", 1.0, True, "single run; variance check skipped")

    tiers = sorted({response.tier.value for response in responses})
    if len(tiers) > 1:
        return MetricResult("consistency_cv", 0.0, False, f"tier drift across runs: {tiers}")

    anchors = [_tier_metrics_view(response.result).scalar_anchor for response in responses]
    mean = statistics.fmean(anchors)
    cv = 0.0 if mean == 0.0 else statistics.stdev(anchors) / mean
    return MetricResult(
        "consistency_cv",
        round(max(0.0, min(1.0, 1.0 - cv)), 4),
        cv <= max_cv,
        f"cv={cv:.4f} (max={max_cv:.2f}, n={len(responses)}, mean={mean:.1f})",
    )
