# tests/evals/test_metrics.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import (
    ComponentEstimation,
    CostRange,
    DeveloperEstimation,
    DurationRange,
    EstimationResponse,
    ExecutiveEstimation,
    HoursRange,
    Milestone,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
    TierEstimationResult,
    TokenUsage,
)
from aie_estimator.sessions.tier_resolver import Tier
from evals.dataset import GoldenCase
from evals.metrics import (
    evaluate_component_recall,
    evaluate_consistency_cv,
    evaluate_numeric_bounds,
    evaluate_schema_and_tier,
    run_family1_metrics,
)


def _make_response(result: TierEstimationResult, tier: Tier) -> EstimationResponse:
    """Build a minimal valid ``EstimationResponse`` wrapper for metric unit tests."""
    return EstimationResponse(
        result=result,
        tier=tier,
        tier_rule="explicit_override",
        prompt_version="v3",
        cached=False,
        model="reasoner-fast",
        provider="openai",
        cost_usd=0.001,
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    )


def _make_pm_result(*, total_cost_eur: int = 30000, total_weeks: int = 8, confidence_pct: int = 80, summary: str = "CRM with Stripe billing and RBAC.") -> PmEstimation:
    """Build a valid ``PmEstimation`` fixture."""
    discovery_cost = min(10000, total_cost_eur)
    return PmEstimation(
        summary=summary,
        confidence_pct=confidence_pct,
        phases=[
            Phase(name="Discovery", duration_weeks=2, cost_eur=discovery_cost, summary="Scope and architecture baseline."),
            Phase(name="Delivery", duration_weeks=total_weeks - 2, cost_eur=total_cost_eur - discovery_cost, summary="Core CRM and Stripe build."),
        ],
        milestones=[Milestone(name="MVP", target_week=total_weeks, deliverable="Deployed CRM")],
        team_composition=TeamComposition(developers=2, designers=1, qa_engineers=1, tech_lead=True),
        blockers=[],
        total_duration_weeks=total_weeks,
        total_cost_eur=total_cost_eur,
    )


def _make_dev_result(*, min_hours: int = 400, max_hours: int = 600, confidence_pct: int = 75) -> DeveloperEstimation:
    """Build a valid ``DeveloperEstimation`` fixture."""
    return DeveloperEstimation(
        summary="Kubernetes microservices with Kafka and PostgreSQL.",
        confidence_pct=confidence_pct,
        components=[
            ComponentEstimation(
                name="Kafka Event Bus",
                estimated_hours=200,
                technical_risks=["Backpressure"],
                stack_assumptions=["Kafka 3.x"],
            ),
        ],
        technical_risks=["Schema evolution"],
        stack_assumptions=["Kubernetes", "PostgreSQL"],
        uncertainty_drivers=["Throughput SLA"],
        total_hours_range=HoursRange(min_hours=min_hours, max_hours=max_hours),
    )


def _make_exec_result(*, min_eur: int = 80000, max_eur: int = 120000, confidence_pct: int = 70) -> ExecutiveEstimation:
    """Build a valid ``ExecutiveEstimation`` fixture."""
    return ExecutiveEstimation(
        summary="HIPAA telehealth portal with audit logging.",
        confidence_pct=confidence_pct,
        headline_cost_range=CostRange(min_eur=min_eur, max_eur=max_eur),
        headline_duration_range=DurationRange(min_weeks=12, max_weeks=18),
        confidence_level="medium",
        top_three_risks=["HIPAA audit scope", "EHR vendor API", "Timeline pressure"],
        go_no_go_recommendation="Proceed with phased discovery.",
    )


def test_family1_metrics_pass_for_valid_pm_response() -> None:
    """
    GIVEN a ``GoldenCase`` and a matching ``PmEstimation`` response within bounds
    WHEN ``run_family1_metrics`` is called
    THEN all three Family 1 metrics pass with full or passing scores.
    """
    case = GoldenCase(
        id="pm-001",
        description="Build a B2B SaaS CRM with Stripe billing and role-based access control.",
        project_type=ProjectType.WEB_SAAS,
        expected_tier=Tier.PM,
        expected_cost_range_eur=(20000, 50000),
        expected_duration_weeks_range=(4, 12),
        required_components_any_of=["crm", "stripe", "rbac"],
        min_confidence_pct=50,
    )
    response = _make_response(_make_pm_result(), Tier.PM)

    results = run_family1_metrics(case, response)

    assert len(results) == 3
    assert all(metric.passed for metric in results)


@pytest.mark.parametrize(
    ("tier", "result_factory", "case_kwargs"),
    [
        (
            Tier.DEVELOPER,
            lambda: _make_dev_result(min_hours=450, max_hours=550),
            {"expected_tier": Tier.DEVELOPER, "expected_hours_range": (400, 600)},
        ),
        (
            Tier.EXECUTIVE,
            lambda: _make_exec_result(min_eur=80000, max_eur=120000),
            {"expected_tier": Tier.EXECUTIVE, "expected_cost_range_eur": (70000, 150000), "expected_duration_weeks_range": (10, 20)},
        ),
    ],
    ids=["developer_hours_bounds", "executive_cost_and_duration_bounds"],
)
def test_numeric_bounds_supports_developer_and_executive_tiers(
    tier: Tier,
    result_factory: object,
    case_kwargs: dict[str, object],
) -> None:
    """
    GIVEN a ``DeveloperEstimation`` or ``ExecutiveEstimation`` response within expected bounds
    WHEN ``evaluate_schema_and_tier`` and ``evaluate_numeric_bounds`` are called
    THEN both metrics pass.

    :param tier: Target stakeholder tier.
    :param result_factory: Callable producing the tier estimation fixture.
    :param case_kwargs: Keyword overrides for the ``GoldenCase``.
    """
    case = GoldenCase(
        id=f"{tier.value}-001",
        description="Enterprise software platform with clear architectural scope.",
        project_type=ProjectType.WEB_SAAS,
        **case_kwargs,  # type: ignore[arg-type]
    )
    response = _make_response(result_factory(), tier)  # type: ignore[operator]

    assert evaluate_schema_and_tier(case, response).passed is True
    assert evaluate_numeric_bounds(case, response).passed is True


def test_out_of_scope_envelope_passes_and_non_zero_fails() -> None:
    """
    GIVEN an ``expected_out_of_scope=True`` case
    WHEN evaluated against a zero-cost low-confidence response vs. a priced response
    THEN the zero-cost response passes and the priced response fails.
    """
    case = GoldenCase(
        id="oos-001",
        description="Make something cool with AI that disrupts everything quickly.",
        project_type=ProjectType.WEB_SAAS,
        expected_tier=Tier.PM,
        expected_out_of_scope=True,
    )
    valid_oos = _make_response(
        PmEstimation(
            summary="Out of scope: request lacks actionable functional requirements.",
            confidence_pct=10,
            phases=[Phase(name="n/a", duration_weeks=1, cost_eur=0, summary="Insufficient scope to estimate.")],
            milestones=[],
            team_composition=TeamComposition(developers=0, designers=0, qa_engineers=0, tech_lead=False),
            blockers=["Missing requirements"],
            total_duration_weeks=1,
            total_cost_eur=0,
        ),
        Tier.PM,
    )
    invalid_oos = _make_response(_make_pm_result(total_cost_eur=25000, confidence_pct=80), Tier.PM)

    assert evaluate_schema_and_tier(case, valid_oos).passed is True
    assert evaluate_numeric_bounds(case, valid_oos).passed is True
    assert evaluate_schema_and_tier(case, invalid_oos).passed is False
    assert evaluate_numeric_bounds(case, invalid_oos).passed is False


def test_component_recall_reports_partial_fraction() -> None:
    """
    GIVEN a case requiring 4 components where only 1 appears in the serialized result
    WHEN ``evaluate_component_recall`` runs with default threshold 0.5
    THEN it reports ``score=0.25`` and ``passed=False``.
    """
    case = GoldenCase(
        id="recall-001",
        description="Build an analytics pipeline with Airflow, Spark, Kafka, and Terraform.",
        project_type=ProjectType.DATA_PIPELINE,
        required_components_any_of=["airflow", "spark", "kafka", "terraform"],
    )
    response = _make_response(_make_dev_result(), Tier.DEVELOPER)

    metric = evaluate_component_recall(case, response)

    assert metric.score == pytest.approx(0.25)
    assert metric.passed is False
    assert "airflow" in metric.details


def test_phase_count_and_technology_recall_checks() -> None:
    """
    GIVEN a ``PmEstimation`` with 2 phases and a case expecting 3-6 phases and a technology hit
    WHEN ``evaluate_schema_and_tier`` and ``evaluate_component_recall`` run
    THEN phase count violation and missing technology both fail with diagnostic details.
    """
    case = GoldenCase(
        id="pm-checks-001",
        description="Build a B2B SaaS CRM with Stripe billing and role-based access control.",
        project_type=ProjectType.WEB_SAAS,
        expected_tier=Tier.PM,
        expected_phase_count_range=(3, 6),
        required_components_any_of=["crm"],
        expected_technologies_any_of=["Kubernetes", "Kafka"],
    )
    response = _make_response(_make_pm_result(), Tier.PM)

    schema_metric = evaluate_schema_and_tier(case, response)
    recall_metric = evaluate_component_recall(case, response)

    assert schema_metric.passed is False
    assert "phase count 2 outside [3, 6]" in schema_metric.details
    assert recall_metric.passed is False
    assert "no tech hit" in recall_metric.details


@pytest.mark.parametrize(
    ("costs", "tiers", "expected_passed"),
    [
        ([100000, 105000, 95000], [Tier.PM, Tier.PM, Tier.PM], True),
        ([50000, 150000, 100000], [Tier.PM, Tier.PM, Tier.PM], False),
        ([0, 0, 0], [Tier.PM, Tier.PM, Tier.PM], True),
        ([100000, 100000], [Tier.PM, Tier.DEVELOPER], False),
    ],
    ids=["low_variance_passes", "high_variance_fails", "zero_mean_passes", "tier_drift_fails"],
)
def test_consistency_cv_evaluates_variance_and_tier_stability(
    costs: list[int],
    tiers: list[Tier],
    expected_passed: bool,
) -> None:
    """
    GIVEN multiple ``EstimationResponse`` runs with varying costs or tiers
    WHEN ``evaluate_consistency_cv`` is called
    THEN it passes only when tier is stable and CV <= threshold.

    :param costs: Total EUR costs across repeated runs.
    :param tiers: Resolved tiers across repeated runs.
    :param expected_passed: Expected pass/fail boolean.
    """
    responses = [
        _make_response(
            _make_pm_result(total_cost_eur=cost) if tier == Tier.PM else _make_dev_result(),
            tier,
        )
        for cost, tier in zip(costs, tiers, strict=True)
    ]

    metric = evaluate_consistency_cv(responses)

    assert metric.passed is expected_passed
