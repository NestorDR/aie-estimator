# tests/prompts/v1/test_estimation.py

# --- Python Standard Library ---
from pathlib import Path

# --- Third Party Libraries ---
from jinja2.exceptions import UndefinedError
import pytest

# --- App modules ---
from aie_estimator.prompts.loader import (
    _ENV,
    list_prompt_versions,
    render_estimation_prompt,
    resolve_prompt_version,
    validate_prompt_version,
)
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def test_render_estimation_prompt_user_description_interpolation() -> None:
    """
    GIVEN a EstimationRequest with a specific project description and project type
    WHEN render_estimation_prompt is invoked
    THEN the rendered user prompt encapsulates the description within <project_description> tags and states the project type.
    """
    description = "Mobile application for warehouse inventory tracking with barcode scanning."
    req = EstimationRequest(
        description=description,
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system_prompt, user_prompt = render_estimation_prompt(req, version="v1")

    assert "<project_description>" in user_prompt
    assert "</project_description>" in user_prompt
    assert description in user_prompt
    assert "Project type: mobile_app." in user_prompt
    assert len(system_prompt) > 0


@pytest.mark.parametrize(
    ("output_format", "expected_fragment", "unexpected_fragment"),
    [
        (
            OutputFormat.PHASES_TABLE,
            "Markdown table with columns",
            "flowing prose",
        ),
        (
            OutputFormat.LINE_ITEMS,
            "numbered list of granular line items",
            "Markdown table with columns",
        ),
        (
            OutputFormat.NARRATIVE,
            "flowing prose",
            "Markdown table with columns",
        ),
    ],
)
def test_render_estimation_prompt_output_format_conditionals(
    output_format: OutputFormat,
    expected_fragment: str,
    unexpected_fragment: str,
) -> None:
    """
    GIVEN a EstimationRequest with a specific output format
    WHEN render_estimation_prompt is invoked
    THEN the system prompt includes instructions for that layout and excludes other format layouts.

    :param output_format: Target layout format enum.
    :param expected_fragment: Substring that must appear in the rendered system prompt.
    :param unexpected_fragment: Substring that must be absent from the rendered system prompt.
    """
    req = EstimationRequest(
        description="Enterprise platform for customer relationship management.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=output_format,
    )

    system_prompt, _ = render_estimation_prompt(req, version="v1")

    assert expected_fragment in system_prompt
    assert unexpected_fragment not in system_prompt


@pytest.mark.parametrize(
    ("detail_level", "expected_fragment", "unexpected_fragment"),
    [
        (
            DetailLevel.SUMMARY,
            "totals and the overall delivery window",
            "assumptions per phase",
        ),
        (
            DetailLevel.MEDIUM,
            "plus a one-line description",
            "assumptions per phase",
        ),
        (
            DetailLevel.DETAILED,
            "assumptions per phase",
            "totals and the overall delivery window",
        ),
    ],
)
def test_render_estimation_prompt_detail_level_conditionals(
    detail_level: DetailLevel,
    expected_fragment: str,
    unexpected_fragment: str,
) -> None:
    """
    GIVEN a EstimationRequest with a specific detail level
    WHEN render_estimation_prompt is invoked
    THEN the system prompt activates the corresponding detail guidance block and omits others.

    :param detail_level: Target detail level enum.
    :param expected_fragment: Substring expected in the rendered system prompt.
    :param unexpected_fragment: Substring that must be omitted from the rendered system prompt.
    """
    req = EstimationRequest(
        description="Automated continuous integration pipeline migration project.",
        project_type=ProjectType.INTERNAL_TOOL,
        detail_level=detail_level,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system_prompt, _ = render_estimation_prompt(req, version="v1")

    assert expected_fragment in system_prompt
    assert unexpected_fragment not in system_prompt


def test_render_estimation_prompt_includes_canonical_examples() -> None:
    """
    GIVEN a valid EstimationRequest
    WHEN render_estimation_prompt is invoked
    THEN the rendered system prompt includes the few-shot examples block from examples.j2.
    """
    req = EstimationRequest(
        description="Scalable streaming data ingestion and aggregation service.",
        project_type=ProjectType.DATA_PIPELINE,
        detail_level=DetailLevel.DETAILED,
        output_format=OutputFormat.PHASES_TABLE,
    )

    system_prompt, _ = render_estimation_prompt(req, version="v1")

    assert "<examples>" in system_prompt
    assert "</examples>" in system_prompt
    assert "<example id=" in system_prompt
    assert "A small B2B SaaS to manage employee equipment loans" in system_prompt
    assert "Native mobile app (iOS + Android) for a meal-prep startup" in system_prompt


def test_render_estimation_prompt_strict_undefined_missing_variable_raises() -> None:
    """
    GIVEN a template rendered with a missing required variable
    WHEN evaluation executes under StrictUndefined
    THEN an UndefinedError is raised.
    """
    template = _ENV.get_template("estimation/v1/system.j2")
    with pytest.raises(UndefinedError, match=r"is undefined"):
        template.render(
            description="Valid description here with enough characters.",
            project_type="mobile_app",
            detail_level="detailed",
            # output_format intentionally omitted to trigger strict undefined
        )


def test_render_estimation_prompt_invalid_version_raises_validation_error() -> None:
    """
    GIVEN a non-existent template version identifier
    WHEN render_estimation_prompt is invoked
    THEN a ValueError is raised with informative message.
    """
    req = EstimationRequest(
        description="A production grade enterprise analytics platform.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )

    with pytest.raises(ValueError, match=r"Invalid prompt version"):
        render_estimation_prompt(req, version="v99")


def test_list_prompt_versions_discovers_only_complete_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a use-case directory with complete and incomplete version subdirectories
    WHEN list_prompt_versions is called
    THEN only versions containing system.j2, user.j2, and examples.j2 are returned.

    :param tmp_path: Temporary directory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    prompts_root = tmp_path / "prompts"
    estimation_dir = prompts_root / "estimation"
    estimation_dir.mkdir(parents=True)

    # Complete v1
    v1_dir = estimation_dir / "v1"
    v1_dir.mkdir()
    (v1_dir / "system.j2").write_text("system", encoding="utf-8")
    (v1_dir / "user.j2").write_text("user", encoding="utf-8")
    (v1_dir / "examples.j2").write_text("examples", encoding="utf-8")

    # Incomplete v2 (missing examples.j2)
    v2_dir = estimation_dir / "v2"
    v2_dir.mkdir()
    (v2_dir / "system.j2").write_text("system", encoding="utf-8")
    (v2_dir / "user.j2").write_text("user", encoding="utf-8")

    # Complete v3
    v3_dir = estimation_dir / "v3"
    v3_dir.mkdir()
    (v3_dir / "system.j2").write_text("system", encoding="utf-8")
    (v3_dir / "user.j2").write_text("user", encoding="utf-8")
    (v3_dir / "examples.j2").write_text("examples", encoding="utf-8")

    monkeypatch.setattr("aie_estimator.prompts.loader._BASE_DIR", prompts_root)

    discovered = list_prompt_versions("estimation")
    assert discovered == ["v1", "v3"]

    # Unknown use case directory returns empty list
    assert list_prompt_versions("nonexistent_use_case") == []


def test_validate_prompt_version_success_and_failure() -> None:
    """
    GIVEN a valid and an unconfigured prompt version
    WHEN validate_prompt_version is executed
    THEN the valid version is returned and the unknown one raises a ValueError naming it.
    """
    assert validate_prompt_version("v1") == "v1"
    with pytest.raises(ValueError, match=r"Invalid prompt version 'unknown'"):
        validate_prompt_version("unknown")


def test_resolve_prompt_version_fallback_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN no explicit version requested
    WHEN resolve_prompt_version is called
    THEN it resolves and validates against get_settings().active_prompt_version.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    assert resolve_prompt_version(None) == "v1"
    assert resolve_prompt_version("") == "v1"

    monkeypatch.setenv("ACTIVE_PROMPT_VERSION", "invalid_v")
    with pytest.raises(ValueError, match=r"Invalid prompt version 'invalid_v'"):
        resolve_prompt_version()


def test_list_prompt_versions_non_estimation_use_case_omits_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a non-estimation use-case directory containing only system.j2 and user.j2
    WHEN list_prompt_versions is called for that use case
    THEN the version is recognized without requiring examples.j2.

    :param tmp_path: Temporary directory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    prompts_root = tmp_path / "prompts"
    extractor_v1 = prompts_root / "metadata_extraction" / "v1"
    extractor_v1.mkdir(parents=True)
    (extractor_v1 / "system.j2").write_text("system", encoding="utf-8")
    (extractor_v1 / "user.j2").write_text("user", encoding="utf-8")

    incomplete_v2 = prompts_root / "metadata_extraction" / "v2"
    incomplete_v2.mkdir(parents=True)
    (incomplete_v2 / "system.j2").write_text("system", encoding="utf-8")

    monkeypatch.setattr("aie_estimator.prompts.loader._BASE_DIR", prompts_root)

    assert list_prompt_versions("metadata_extraction") == ["v1"]

