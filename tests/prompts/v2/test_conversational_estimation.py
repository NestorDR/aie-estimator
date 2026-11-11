# tests/prompts/v2/test_conversational_estimation.py

# --- Third Party Libraries ---
from jinja2.exceptions import UndefinedError
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import (
    _ENV,
    list_prompt_versions,
    render_conversational_prompt,
)
from aie_estimator.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from aie_estimator.sessions.models import ProjectMetadata


def test_list_prompt_versions_includes_v2() -> None:
    """
    GIVEN the estimation prompts directory with v1 and v2 suites
    WHEN list_prompt_versions is called for estimation
    THEN both v1 and v2 are discovered as complete suites.
    """
    assert list_prompt_versions("estimation") == ["v1", "v2"]


def test_render_conversational_prompt_empty_metadata() -> None:
    """
    GIVEN an empty ProjectMetadata on the first conversational turn
    WHEN render_conversational_prompt executes with default version
    THEN it renders the empty-state message inside <project_metadata>, authority directive, and v1 examples.
    """
    system_prompt, user_prompt = render_conversational_prompt(
        description="Build an internal inventory dashboard with role-based access.",
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(),
    )

    assert "<project_metadata>" in system_prompt
    assert "(No project context yet — this is the first turn of the session.)" in system_prompt
    assert "treat this block as the source of truth" in system_prompt
    assert "updated full estimation" in system_prompt
    assert "<examples>" in system_prompt
    assert "Project type: internal_tool." in user_prompt
    assert "integrating the project_metadata context and the conversation history" in user_prompt


def test_render_conversational_prompt_full_metadata_renders_all_six_fields() -> None:
    """
    GIVEN a fully populated 6-field ProjectMetadata instance
    WHEN render_conversational_prompt executes
    THEN all six fields appear inside <project_metadata> without placeholder strings.
    """
    metadata = ProjectMetadata(
        project_name="Atlas Billing",
        assumed_team_size=4,
        mentioned_technologies=["FastAPI", "PostgreSQL", "Redis"],
        agreed_scope="Multi-tenant invoicing MVP with Stripe webhooks",
        explicit_constraints=["EU data residency", "Launch before Q3"],
        rejected_options=["Microservices", "GraphQL"],
    )

    system_prompt, _ = render_conversational_prompt(
        description="Add automated PDF invoice generation.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.NARRATIVE,
        metadata=metadata,
    )

    assert "- Project name: Atlas Billing" in system_prompt
    assert "- Assumed team size: 4 full-time engineers" in system_prompt
    assert "- Technologies mentioned: FastAPI, PostgreSQL, Redis" in system_prompt
    assert "- Agreed scope: Multi-tenant invoicing MVP with Stripe webhooks" in system_prompt
    assert "- Explicit constraints:" in system_prompt
    assert "  - EU data residency" in system_prompt
    assert "  - Launch before Q3" in system_prompt
    assert "- Rejected options (do not propose these again):" in system_prompt
    assert "  - Microservices" in system_prompt
    assert "  - GraphQL" in system_prompt


def test_render_conversational_prompt_partial_metadata_omits_unset_fields() -> None:
    """
    GIVEN a partially populated ProjectMetadata with only project_name and explicit_constraints
    WHEN render_conversational_prompt executes
    THEN populated fields render while unset fields are completely omitted without unknown/none placeholders.
    """
    metadata = ProjectMetadata(
        project_name="Pulse Tracker",
        explicit_constraints=["HIPAA audit logging"],
    )

    system_prompt, _ = render_conversational_prompt(
        description="Add audit trail export.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.SUMMARY,
        output_format=OutputFormat.LINE_ITEMS,
        metadata=metadata,
    )

    assert "- Project name: Pulse Tracker" in system_prompt
    assert "  - HIPAA audit logging" in system_prompt
    assert "Assumed team size:" not in system_prompt
    assert "Technologies mentioned:" not in system_prompt
    assert "Agreed scope:" not in system_prompt
    assert "Rejected options" not in system_prompt


def test_render_conversational_prompt_accepts_enriched_transcript_over_2000_chars() -> None:
    """
    GIVEN an enriched transcript exceeding the transactional 2000-character limit
    WHEN render_conversational_prompt executes
    THEN the full transcript renders cleanly into the user prompt without truncation or validation error.
    """
    enriched = "Base scope.\n\n" + ("x" * 2500)
    _, user_prompt = render_conversational_prompt(
        description=enriched,
        project_type=ProjectType.DATA_PIPELINE,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
        metadata=ProjectMetadata(),
    )

    assert ("x" * 2500) in user_prompt


def test_render_conversational_prompt_strict_undefined_raises() -> None:
    """
    GIVEN estimation/v2/system.j2 rendered without required metadata context variables
    WHEN Jinja2 renders with StrictUndefined
    THEN UndefinedError is raised immediately.
    """
    template = _ENV.get_template("estimation/v2/system.j2")
    with pytest.raises(UndefinedError):
        template.render(
            project_type="web_saas",
            detail_level="medium",
            output_format="phases_table",
            version="v2",
        )


def test_render_conversational_prompt_rejects_invalid_setting_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an invalid conversational_prompt_version in settings
    WHEN render_conversational_prompt is called without an override
    THEN ValueError is raised naming the invalid version.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("CONVERSATIONAL_PROMPT_VERSION", "v99")
    with pytest.raises(ValueError, match=r"Invalid prompt version 'v99'"):
        render_conversational_prompt(
            description="Estimate customer portal.",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.SUMMARY,
            output_format=OutputFormat.PHASES_TABLE,
            metadata=ProjectMetadata(),
        )

