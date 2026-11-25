# tests/schemas/test_acb.py

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.schemas.acb import (
    ACBIteration,
    ACBResponse,
    BossTrace,
    CriticFeedback,
    CriticIssue,
)
from aie_estimator.schemas.estimations import CostRange, DurationRange, ExecutiveEstimation, TokenUsage
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.settings import Environment, Settings
from aie_estimator.utils.constants import DEFAULT_FAST_PROFILE


def test_critic_feedback_accepts_minor_issues() -> None:
    """
    GIVEN a CriticFeedback with verdict='accept' and a minor issue
    WHEN validated
    THEN the model is constructed without raising a validation error.
    """
    feedback = CriticFeedback(
        verdict="accept",
        issues=[
            CriticIssue(
                category="unrealistic_estimate",
                severity="minor",
                field_path="summary",
                description="Summary phrasing could be slightly tighter.",
            )
        ],
        confidence_in_review=88,
    )

    assert feedback.verdict == "accept"
    assert len(feedback.issues) == 1


@pytest.mark.parametrize(
    ("verdict", "issues", "expected_match"),
    [
        (
            "needs_iteration",
            [
                CriticIssue(
                    category="scope_omission",
                    severity="minor",
                    field_path="summary",
                    description="Minor wording detail missing.",
                )
            ],
            r"severity in \{critical, major\}",
        ),
        (
            "needs_iteration",
            [],
            r"severity in \{critical, major\}",
        ),
        (
            "reject",
            [],
            r"at least one issue",
        ),
    ],
)
def test_critic_feedback_validator_rejects_invalid_verdicts(
    verdict: str,
    issues: list[CriticIssue],
    expected_match: str,
) -> None:
    """
    GIVEN an invalid CriticFeedback combination of verdict and issues
    WHEN validated
    THEN Pydantic raises ValidationError matching the expected invariant.
    """
    with pytest.raises(ValidationError, match=expected_match):
        CriticFeedback(
            verdict=verdict,  # type: ignore[arg-type]
            issues=issues,
            confidence_in_review=75,
        )


def test_acb_response_model_validate() -> None:
    """
    GIVEN a normalized service result dictionary containing boss_trace
    WHEN ACBResponse.model_validate is called
    THEN it constructs a valid ACBResponse preserving estimation fields and trace.
    """
    estimation = ExecutiveEstimation(
        summary="Executive summary for a web SaaS platform.",
        confidence_pct=82,
        headline_cost_range=CostRange(min_eur=40000, max_eur=55000),
        headline_duration_range=DurationRange(min_weeks=8, max_weeks=12),
        confidence_level="medium",
        top_three_risks=["Third-party OAuth integration"],
        go_no_go_recommendation="Go with a 2-week discovery spike.",
    )
    trace = BossTrace(
        iterations=[
            ACBIteration(
                iteration=0,
                decision_after="accept",
                critic_verdict="accept",
                critic_confidence=90,
                issue_summary=[],
            )
        ],
        final_decision="accept",
        iterations_run=1,
    )
    service_result = {
        "result": estimation,
        "tier": Tier.EXECUTIVE,
        "tier_rule": "header_x_user_tier",
        "cached": False,
        "prompt_version": "v3",
        "model": DEFAULT_FAST_PROFILE,
        "provider": "litellm",
        "cost_usd": 0.0024,
        "usage": TokenUsage(input_tokens=400, output_tokens=200, total_tokens=600),
        "boss_trace": trace,
    }

    response = ACBResponse.model_validate(service_result)

    assert response.tier == Tier.EXECUTIVE
    assert response.boss_trace.final_decision == "accept"
    assert response.boss_trace.iterations_run == 1
    assert response.usage.total_tokens == 600


def test_settings_acb_defaults_and_bounds() -> None:
    """
    GIVEN default Settings in TEST environment
    WHEN inspecting ACB configuration fields
    THEN acb_max_iterations defaults to 2 and acb_critic_model defaults to DEFAULT_FAST_PROFILE.
    """
    settings = Settings(app_env=Environment.TEST)
    assert settings.acb_max_iterations == 2
    assert settings.acb_critic_model == DEFAULT_FAST_PROFILE

    with pytest.raises(ValidationError):
        Settings(app_env=Environment.TEST, acb_max_iterations=0)

    with pytest.raises(ValidationError):
        Settings(app_env=Environment.TEST, acb_max_iterations=6)
