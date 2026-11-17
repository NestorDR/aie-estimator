# tests/prompts/test_loader.py

# --- Third Party Libraries ---
from jinja2.exceptions import UndefinedError
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import (
    _ENV,
    render_critic_prompt,
    render_metadata_extraction_prompt,
    render_tier_estimation_prompt,
    resolve_prompt_version,
)
from aie_estimator.schemas.acb import CriticFeedback, CriticIssue
from aie_estimator.schemas.estimations import (
    BaseTierEstimation,
    ComponentEstimation,
    CostRange,
    DetailLevel,
    DeveloperEstimation,
    DurationRange,
    ExecutiveEstimation,
    HoursRange,
    OutputFormat,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
)
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.sessions.tier_resolver import Tier


def _sample_estimation_result() -> PmEstimation:
    """Build a minimal valid PmEstimation for metadata extraction prompt tests."""
    return PmEstimation(
        summary="Web SaaS MVP delivery estimate.",
        confidence_pct=85,
        phases=[
            Phase(
                name="Implementation",
                duration_weeks=4,
                cost_eur=10_000,
                summary="Build backend API and frontend client.",
            ),
        ],
        team_composition=TeamComposition(developers=2, designers=1),
        total_duration_weeks=4,
        total_cost_eur=10_000,
    )


def test_resolve_prompt_version_fallback_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN no explicit version requested or an unconfigured version override
    WHEN resolve_prompt_version is called
    THEN it resolves against get_settings().active_prompt_version and raises ValueError for unknown directories.
    """
    assert resolve_prompt_version(None) == "v3"
    assert resolve_prompt_version("") == "v3"
    assert resolve_prompt_version("v1") == "v1"

    monkeypatch.setenv("ACTIVE_PROMPT_VERSION", "invalid_v")
    with pytest.raises(ValueError, match=r"Invalid prompt version 'invalid_v'"):
        resolve_prompt_version()


@pytest.mark.parametrize(
    ("previous", "expected_snippets", "absent_snippets"),
    [
        (
            ProjectMetadata(),
            ["(empty — this is the first turn)", "Drop Rails and use Node"],
            ["SOC 2 Type II"],
        ),
        (
            ProjectMetadata(
                project_name="Atlas",
                assumed_team_size=4,
                mentioned_technologies=["Rails", "PostgreSQL"],
                agreed_scope="MVP analytics portal",
                explicit_constraints=["SOC 2 Type II"],
                rejected_options=["Microservices"],
            ),
            [
                "- project_name: Atlas",
                "- assumed_team_size: 4",
                "- mentioned_technologies: Rails, PostgreSQL",
                "- agreed_scope: MVP analytics portal",
                "- explicit_constraints: SOC 2 Type II",
                "- rejected_options: Microservices",
            ],
            ["(empty — this is the first turn)"],
        ),
    ],
    ids=["empty_previous_metadata", "populated_six_field_metadata"],
)
def test_render_metadata_extraction_prompt_renders_previous_and_turn_context(
    previous: ProjectMetadata,
    expected_snippets: list[str],
    absent_snippets: list[str],
) -> None:
    """
    GIVEN empty or populated six-field ProjectMetadata and a completed EstimationResult
    WHEN render_metadata_extraction_prompt renders metadata_extraction/v1
    THEN system and user prompts include all expected metadata fields, transcript, and phase breakdown.

    :param previous: Prior ProjectMetadata snapshot.
    :param expected_snippets: Substrings required in the rendered user prompt.
    :param absent_snippets: Substrings that must not appear in the rendered user prompt.
    """
    system_prompt, user_prompt = render_metadata_extraction_prompt(
        transcript="Drop Rails and use Node instead.",
        result=_sample_estimation_result(),
        previous=previous,
    )

    assert "explicit_constraints" in system_prompt
    assert "rejected_options" in system_prompt
    assert "retracted_constraints" in system_prompt
    assert "Implementation (4w, 10000 EUR)" in user_prompt
    for snippet in expected_snippets:
        assert snippet in user_prompt
    for snippet in absent_snippets:
        assert snippet not in user_prompt


def test_render_metadata_extraction_user_template_strict_undefined() -> None:
    """
    GIVEN metadata_extraction/v1/user.j2 rendered directly without required transcript context
    WHEN StrictUndefined evaluates the template
    THEN UndefinedError is raised immediately.
    """
    template = _ENV.get_template("metadata_extraction/v1/user.j2")
    with pytest.raises(UndefinedError, match=r"transcript"):
        template.render(previous=ProjectMetadata(), result=_sample_estimation_result())


@pytest.mark.parametrize(
    ("tier", "candidate_factory", "expected_json_snippet"),
    [
        (
            Tier.DEVELOPER,
            lambda: DeveloperEstimation(
                summary="Developer breakdown for auth and API.",
                confidence_pct=80,
                components=[ComponentEstimation(name="Auth Service", estimated_hours=40)],
                total_hours_range=HoursRange(min_hours=35, max_hours=50),
            ),
            "Auth Service",
        ),
        (
            Tier.PM,
            _sample_estimation_result,
            "total_duration_weeks",
        ),
        (
            Tier.EXECUTIVE,
            lambda: ExecutiveEstimation(
                summary="Executive estimate for B2B portal.",
                confidence_pct=78,
                headline_cost_range=CostRange(min_eur=25000, max_eur=35000),
                headline_duration_range=DurationRange(min_weeks=6, max_weeks=9),
                confidence_level="medium",
                top_three_risks=["SSO scope"],
                go_no_go_recommendation="Go with discovery phase.",
            ),
            "headline_cost_range",
        ),
    ],
)
def test_render_critic_prompt_across_all_tiers(
    tier: Tier,
    candidate_factory: object,
    expected_json_snippet: str,
) -> None:
    """
    GIVEN a candidate estimation for any stakeholder tier
    WHEN render_critic_prompt renders critic/v1 templates
    THEN both system and user prompts render under StrictUndefined with serialized candidate JSON.
    """
    candidate: BaseTierEstimation = candidate_factory()  # type: ignore[operator]
    system_prompt, user_prompt = render_critic_prompt(
        transcript="Build an authenticated B2B portal in 8 weeks.",
        metadata=ProjectMetadata(project_name="Portal", mentioned_technologies=["FastAPI"]),
        tier=tier,
        result=candidate,
    )

    assert "arithmetic_error" in system_prompt
    assert f"<resolved_tier>{tier.value}</resolved_tier>" in user_prompt
    assert expected_json_snippet in user_prompt


def test_render_tier_estimation_prompt_with_and_without_critic_feedback() -> None:
    """
    GIVEN estimation/v3/user.j2 rendered with and without CriticFeedback
    WHEN render_tier_estimation_prompt is invoked
    THEN initial passes omit <critic_feedback> and revision passes include all flagged issues.
    """
    _, initial_user = render_tier_estimation_prompt(
        description="Build an internal reporting tool with RBAC.",
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        tier=Tier.PM,
        version="v3",
    )
    assert "<critic_feedback>" not in initial_user

    feedback = CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="missing_component",
                severity="major",
                field_path="phases",
                description="RBAC implementation is missing from phases.",
                suggested_fix="Add an RBAC task to the backend phase.",
            )
        ],
        confidence_in_review=85,
    )
    _, revision_user = render_tier_estimation_prompt(
        description="Build an internal reporting tool with RBAC.",
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        tier=Tier.PM,
        version="v3",
        critic_feedback=feedback,
    )
    assert "<critic_feedback>" in revision_user
    assert "[major] missing_component @ phases" in revision_user
    assert "Add an RBAC task to the backend phase." in revision_user


def test_render_conversation_summary_prompt_renders_prior_and_evicted() -> None:
    """
    GIVEN a prior rolling summary and evicted user/assistant messages
    WHEN render_conversation_summary_prompt is invoked
    THEN system and user prompts render under StrictUndefined with both prior summary and evicted turns.
    """
    from aie_estimator.prompts.loader import render_conversation_summary_prompt
    from aie_estimator.sessions.models import Message

    system, user = render_conversation_summary_prompt(
        previous_summary="Team agreed on FastAPI backend.",
        evicted=[
            Message(role="user", content="Add PostgreSQL and Redis."),
            Message(role="assistant", content="Added 16 hours for data layer."),
        ],
    )
    assert "conversation summarizer" in system
    assert "Team agreed on FastAPI backend." in user
    assert "[user] Add PostgreSQL and Redis." in user
    assert "[assistant] Added 16 hours for data layer." in user

    _, first_pass_user = render_conversation_summary_prompt(
        previous_summary=None,
        evicted=[Message(role="user", content="Initial scope.")],
    )
    assert "(no previous summary" in first_pass_user


