# src/aie_estimator/prompts/loader.py
"""
Prompts loader and renderer module for software estimation.

This module defines utilities for loading, rendering, and managing Jinja2-based templates for estimation requests.

Template files are located at ``app/prompts/<use_case>/<version>/<role>.j2``.

Versioning allows you to change the prompts by modifying a string at the invocation (``version=“v2”``),
 rather than refactoring the code.
"""

# --- Python Standard Library ---
from pathlib import Path
from typing import Any, Final

# --- Third Party Libraries ---
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# --- App modules ---
from aie_estimator.schemas.estimations import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from aie_estimator.sessions.models import ProjectMetadata
from aie_estimator.settings import get_settings

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent
_DEFAULT_REQUIRED_PROMPT_FILES: Final[frozenset[str]] = frozenset({"system.j2", "user.j2"})
_REQUIRED_FILES_BY_USE_CASE: Final[dict[str, frozenset[str]]] = {
    "estimation": frozenset({"system.j2", "user.j2", "examples.j2"}),
}

_ENV: Final[Environment] = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701 - Prompts are plain text messages dispatched to LLMs, not HTML in a browser.
    keep_trailing_newline=True,
)


def _enum_value(value: object) -> str:
    """Return the underlying string value for an enum or string-like input."""
    return str(value.value) if hasattr(value, "value") else str(value)


def list_prompt_versions(use_case: str = "estimation") -> list[str]:
    """
    Discover all complete, valid prompt template versions for a specific use case.

    A version is considered valid and complete if its directory exists under
    ``prompts/<use_case>/<version>`` and contains all required template files
    configured for that use case.

    :param use_case: The use case directory name (defaults to "estimation").

    :return: A sorted list of valid prompt version identifiers (e.g., ["v1", "v2"]).
    """
    target_dir = _BASE_DIR / use_case
    if not target_dir.is_dir():
        return []

    required_files = _REQUIRED_FILES_BY_USE_CASE.get(use_case, _DEFAULT_REQUIRED_PROMPT_FILES)
    versions: list[str] = []
    for entry in target_dir.iterdir():
        if entry.is_dir() and required_files.issubset({f.name for f in entry.iterdir() if f.is_file()}):
            versions.append(entry.name)

    return sorted(versions)


def validate_prompt_version(version: str, use_case: str = "estimation") -> str:
    """
    Validate that the specified prompt version exists and contains complete templates.

    :param version: Prompt version identifier to validate.
    :param use_case: The use case directory name (defaults to "estimation").

    :return: The validated version string.
    :raises ValueError: If the version is not found or is incomplete.
    """
    valid_versions = list_prompt_versions(use_case)
    if version not in valid_versions:
        supported = ", ".join(valid_versions) if valid_versions else "none"
        raise ValueError(f"Invalid prompt version '{version}' for use case '{use_case}'. Supported: {supported}")
    return version


def resolve_prompt_version(
    version: str | None = None,
    use_case: str = "estimation",
) -> str:
    """
    Resolve the prompt version, prioritizing an explicit override or falling back to settings.

    :param version: Optional explicit prompt version override.
    :param use_case: The use case directory name (defaults to "estimation").

    :return: The resolved and validated prompt version identifier.
    :raises ValueError: If the resolved version is not a valid prompt version.
    """
    target_version = version or get_settings().active_prompt_version
    return validate_prompt_version(target_version, use_case)


def _render_estimation_templates(
    *,
    description: str,
    project_type: ProjectType | str,
    detail_level: DetailLevel | str,
    output_format: OutputFormat | str,
    version: str,
    metadata: ProjectMetadata | None = None,
) -> tuple[str, str]:
    """Render ``estimation/<version>/system.j2`` and ``user.j2`` from shared estimation inputs."""
    context: dict[str, Any] = {
        "description": description,
        "project_type": _enum_value(project_type),
        "detail_level": _enum_value(detail_level),
        "output_format": _enum_value(output_format),
        "version": version,
    }
    if metadata is not None:
        context["metadata"] = metadata

    system = _ENV.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _ENV.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user


def render_estimation_prompt(
    request: EstimationRequest,
    version: str | None = None,
) -> tuple[str, str]:
    """Render the system and user prompts for the software estimation domain.

    :param request: Validated estimation request parameters.
    :param version: Optional prompt template version identifier. Defaults to active_prompt_version from settings.

    :return: A tuple of ``(system_prompt, user_prompt)`` ready for LLM provider dispatch.
    :raises ValueError: If the requested or configured version is invalid or incomplete.
    :raises jinja2.exceptions.TemplateNotFound: If templates for the requested version do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    return _render_estimation_templates(
        description=request.description,
        project_type=request.project_type,
        detail_level=request.detail_level,
        output_format=request.output_format,
        version=resolve_prompt_version(version),
    )


def render_conversational_prompt(
    *,
    description: str,
    project_type: ProjectType | str,
    detail_level: DetailLevel | str,
    output_format: OutputFormat | str,
    metadata: ProjectMetadata,
    version: str | None = None,
) -> tuple[str, str]:
    """Render the conversational estimation system and user prompts with distilled project metadata.

    :param description: Turn transcript (optionally enriched with extracted attachment text).
    :param project_type: Target project domain category.
    :param detail_level: Desired breakdown depth.
    :param output_format: Desired presentation format.
    :param metadata: Distilled session ``ProjectMetadata`` injected into ``<project_metadata>``.
    :param version: Optional prompt template version override. Defaults to ``conversational_prompt_version``.

    :return: A tuple of ``(system_prompt, user_prompt)`` ready for multi-turn provider dispatch.
    :raises ValueError: If the requested or configured version is invalid or incomplete.
    :raises jinja2.exceptions.TemplateNotFound: If templates for the requested version do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    resolved_version = resolve_prompt_version(version or get_settings().conversational_prompt_version)
    return _render_estimation_templates(
        description=description,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        version=resolved_version,
        metadata=metadata,
    )
