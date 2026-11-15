# tests/prompts/test_loader.py

# --- Third Party Libraries ---
from jinja2.exceptions import UndefinedError
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import (
    _ENV,
    render_metadata_extraction_prompt,
    resolve_prompt_version,
)
from aie_estimator.schemas.estimations import Phase, PmEstimation, TeamComposition
from aie_estimator.sessions.models import ProjectMetadata


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

