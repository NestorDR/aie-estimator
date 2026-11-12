# tests/prompts/test_loader.py

# --- Third Party Libraries ---
from jinja2.exceptions import UndefinedError
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import (
    _ENV,
    list_prompt_versions,
    render_metadata_extraction_prompt,
    validate_prompt_version,
)
from aie_estimator.schemas.estimations import EstimationResult, Phase
from aie_estimator.sessions.models import ProjectMetadata


def _sample_estimation_result() -> EstimationResult:
    """Build a minimal valid EstimationResult for metadata extraction prompt tests."""
    return EstimationResult(
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
        total_duration_weeks=4,
        total_cost_eur=10_000,
    )


def test_list_and_validate_metadata_extraction_prompt_versions() -> None:
    """
    GIVEN the metadata_extraction prompts directory containing system.j2 and user.j2 without examples.j2
    WHEN list_prompt_versions and validate_prompt_version run for metadata_extraction
    THEN v1 is discovered and unknown versions raise ValueError.
    """
    assert list_prompt_versions("metadata_extraction") == ["v1"]
    assert validate_prompt_version("v1", use_case="metadata_extraction") == "v1"
    with pytest.raises(ValueError, match=r"Invalid prompt version"):
        validate_prompt_version("v99", use_case="metadata_extraction")


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

