# tests/domain/schemas/test_estimations_tiers.py

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.domain.schemas.estimations import (
    ComponentEstimation,
    CostRange,
    DeveloperEstimation,
    DurationRange,
    EstimationRequest,
    EstimationResponse,
    ExecutiveEstimation,
    HoursRange,
    Milestone,
    Phase,
    PmEstimation,
    TeamComposition,
    TokenUsage,
)
from aie_estimator.generation.conversation.tier_resolver import Tier


def test_developer_estimate_valid() -> None:
    """
    GIVEN valid developer-tier fields and consistent hours range
    WHEN DeveloperEstimation is validated
    THEN schema validation succeeds.
    """
    estimate = DeveloperEstimation(
        summary="Backend API and PostgreSQL schema for billing.",
        confidence_pct=82,
        components=[
            ComponentEstimation(
                name="Backend API",
                estimated_hours=80,
                technical_risks=["Webhook idempotency"],
                stack_assumptions=["FastAPI + PostgreSQL 18"],
            )
        ],
        technical_risks=["Rate limiting"],
        stack_assumptions=["Python 3.14"],
        uncertainty_drivers=["Third-party ERP sync"],
        total_hours_range=HoursRange(min_hours=70, max_hours=110),
    )

    assert estimate.components[0].estimated_hours == 80
    assert estimate.total_hours_range.max_hours == 110


def test_pm_estimate_valid() -> None:
    """
    GIVEN valid PM-tier fields with matching phase sums
    WHEN PmEstimation is validated
    THEN schema validation succeeds.
    """
    estimate = PmEstimation(
        summary="Phased delivery plan for customer portal.",
        confidence_pct=78,
        phases=[
            Phase(name="Discovery", duration_weeks=1, cost_eur=2000, summary="Scope alignment and architecture."),
            Phase(name="Implementation", duration_weeks=3, cost_eur=6000, summary="Core backend and UI build."),
        ],
        milestones=[Milestone(name="MVP", target_week=4, deliverable="Staging release")],
        team_composition=TeamComposition(developers=2, designers=1, qa_engineers=1, tech_lead=True),
        blockers=["SSO credentials pending"],
        total_duration_weeks=4,
        total_cost_eur=8000,
    )

    assert estimate.total_cost_eur == 8000
    assert estimate.total_duration_weeks == 4


def test_pm_estimate_rejects_phase_mismatch() -> None:
    """
    GIVEN a PmEstimation whose phase costs do not equal total_cost_eur
    WHEN PmEstimation is validated
    THEN ValidationError is raised.
    """
    with pytest.raises(ValidationError, match=r"does not match the sum"):
        PmEstimation(
            summary="Phased delivery plan for customer portal.",
            confidence_pct=78,
            phases=[Phase(name="Build", duration_weeks=2, cost_eur=4000, summary="Core build phase work.")],
            team_composition=TeamComposition(developers=1, designers=0),
            total_duration_weeks=2,
            total_cost_eur=5000,
        )


def test_executive_estimate_valid() -> None:
    """
    GIVEN valid executive-tier fields
    WHEN ExecutiveEstimation is validated
    THEN schema validation succeeds.
    """
    estimate = ExecutiveEstimation(
        summary="Executive summary for Q3 commercial proposal.",
        confidence_pct=75,
        headline_cost_range=CostRange(min_eur=20000, max_eur=28000),
        headline_duration_range=DurationRange(min_weeks=4, max_weeks=6),
        confidence_level="medium",
        top_three_risks=["Vendor API SLA", "Compliance sign-off"],
        go_no_go_recommendation="Go with a 15% contingency buffer.",
    )

    assert estimate.confidence_level == "medium"
    assert len(estimate.top_three_risks) == 2


@pytest.mark.parametrize(
    ("range_cls", "kwargs"),
    [
        (HoursRange, {"min_hours": 100, "max_hours": 50}),
        (CostRange, {"min_eur": 5000, "max_eur": 3000}),
        (DurationRange, {"min_weeks": 6, "max_weeks": 2}),
    ],
)
def test_range_models_reject_inverted_bounds(range_cls: type, kwargs: dict[str, int]) -> None:
    """
    GIVEN a range model where min exceeds max
    WHEN validated
    THEN ValidationError is raised.
    """
    with pytest.raises(ValidationError, match=r"cannot exceed"):
        range_cls(**kwargs)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "cls": DeveloperEstimation,
            "data": {
                "summary": "Missing out of scope prefix.",
                "confidence_pct": 15,
                "components": [ComponentEstimation(name="API", estimated_hours=10)],
                "total_hours_range": HoursRange(min_hours=10, max_hours=15),
            },
        },
        {
            "cls": ExecutiveEstimation,
            "data": {
                "summary": "Missing out of scope prefix.",
                "confidence_pct": 20,
                "headline_cost_range": CostRange(min_eur=0, max_eur=0),
                "headline_duration_range": DurationRange(min_weeks=1, max_weeks=1),
                "confidence_level": "low",
                "top_three_risks": ["Insufficient scope"],
                "go_no_go_recommendation": "Hold.",
            },
        },
    ],
)
def test_low_confidence_requires_prefix_across_tiers(payload: dict) -> None:
    """
    GIVEN a tier estimate with confidence_pct < 30 and no 'Out of scope:' prefix
    WHEN validated
    THEN ValidationError is raised.
    """
    with pytest.raises(ValidationError, match=r"Out of scope:"):
        payload["cls"](**payload["data"])


def test_estimation_request_and_response_tier_contract() -> None:
    """
    GIVEN an EstimationRequest with explicit tier and an ExecutiveEstimation service result
    WHEN EstimationResponse.model_validate is called
    THEN tier, tier_rule, and ExecutiveEstimation result are preserved.
    """
    req = EstimationRequest(
        description="Build an enterprise analytics platform for 500 users.",
        project_type="web_saas",
        detail_level="summary",
        output_format="narrative",
        tier=Tier.EXECUTIVE,
    )
    assert req.tier is Tier.EXECUTIVE

    exec_result = ExecutiveEstimation(
        summary="Executive summary.",
        confidence_pct=80,
        headline_cost_range=CostRange(min_eur=15000, max_eur=22000),
        headline_duration_range=DurationRange(min_weeks=3, max_weeks=5),
        confidence_level="high",
        top_three_risks=["Data migration"],
        go_no_go_recommendation="Proceed.",
    )
    response = EstimationResponse.model_validate(
        {
            "result": exec_result,
            "tier": Tier.EXECUTIVE,
            "tier_rule": "explicit_override",
            "cached": False,
            "prompt_version": "v3",
            "model": "reasoner-fast",
            "provider": "openai",
            "cost_usd": 0.01,
            "usage": TokenUsage(input_tokens=10, output_tokens=20, total_tokens=30),
        }
    )

    assert response.tier is Tier.EXECUTIVE
    assert response.tier_rule == "explicit_override"
    assert isinstance(response.result, ExecutiveEstimation)

