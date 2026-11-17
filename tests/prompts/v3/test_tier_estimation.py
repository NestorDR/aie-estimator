# tests/prompts/v3/test_tier_estimation.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import render_tier_estimation_prompt
from aie_estimator.prompts.tier_config import TIER_CONFIG
from aie_estimator.schemas.estimations import DeveloperEstimation, ExecutiveEstimation, PmEstimation
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.sessions.tier_resolver import Tier


@pytest.mark.parametrize(
    ("tier", "expected_marker", "expected_schema"),
    [
        (Tier.DEVELOPER, "DeveloperEstimation", DeveloperEstimation),
        (Tier.PM, "PmEstimation", PmEstimation),
        (Tier.EXECUTIVE, "ExecutiveEstimation", ExecutiveEstimation),
    ],
)
def test_render_tier_estimation_prompt_v3_all_tiers(
    tier: Tier,
    expected_marker: str,
    expected_schema: type,
) -> None:
    """
    GIVEN valid estimation parameters and a stakeholder Tier
    WHEN render_tier_estimation_prompt is called for v3
    THEN it renders the tier-specific system prompt, shared partials, and user prompt under StrictUndefined.
    """
    config = TIER_CONFIG[tier]
    assert config.schema is expected_schema

    system_prompt, user_prompt = render_tier_estimation_prompt(
        description="Build an internal billing portal with RBAC and Stripe integration.",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        tier=tier,
        version="v3",
        metadata=ProjectMetadata(project_name="Billing Hub", mentioned_technologies=["FastAPI"]),
    )

    assert expected_marker in system_prompt
    assert "62.50 EUR/hour" in system_prompt
    assert "<project_metadata>" in system_prompt
    assert "Billing Hub" in system_prompt
    assert "Build an internal billing portal" in user_prompt


def test_render_tier_estimation_prompt_v3_without_metadata() -> None:
    """
    GIVEN metadata=None for a stateless estimation request
    WHEN render_tier_estimation_prompt is called for v3
    THEN <project_metadata> is omitted cleanly without raising UndefinedError.
    """
    system_prompt, _ = render_tier_estimation_prompt(
        description="Build a simple marketing site with contact form.",
        project_type="web_saas",
        detail_level="summary",
        output_format="narrative",
        tier=Tier.PM,
        version="v3",
        metadata=None,
    )

    assert "<project_metadata>" not in system_prompt
    assert "PmEstimation" in system_prompt


def test_render_tier_estimation_prompt_v3_with_and_without_compressed_history() -> None:
    """
    GIVEN v3 tier estimation prompts with and without conversation_summary and anchored_turns
    WHEN render_tier_estimation_prompt is called
    THEN <conversation_summary> and <anchored_turns> render only when populated.
    """
    from aie_estimator.sessions.models import Message

    without_history, _ = render_tier_estimation_prompt(
        description="Build a billing portal with Stripe.",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        tier=Tier.DEVELOPER,
        version="v3",
    )
    assert "<conversation_summary>" not in without_history
    assert "<anchored_turns>" not in without_history

    with_history, _ = render_tier_estimation_prompt(
        description="Build a billing portal with Stripe.",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        tier=Tier.DEVELOPER,
        version="v3",
        conversation_summary="Prior turns established a 3-person team.",
        anchored_turns=[
            Message(role="user", content="Scope is frozen under NDA."),
            Message(role="assistant", content="Acknowledged frozen scope."),
        ],
    )
    assert "<conversation_summary>" in with_history
    assert "Prior turns established a 3-person team." in with_history
    assert "<anchored_turns>" in with_history
    assert "[user] Scope is frozen under NDA." in with_history

