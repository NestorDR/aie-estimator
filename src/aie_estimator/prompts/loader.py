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
from typing import Final

# --- Third Party Libraries ---
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationRequest
from aie_estimator.settings import get_settings

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent
_REQUIRED_PROMPT_FILES: Final[frozenset[str]] = frozenset({"system.j2", "user.j2", "examples.j2"})

_ENV: Final[Environment] = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701 - Prompts are plain text messages dispatched to LLMs, not HTML in a browser.
    keep_trailing_newline=True,
)


def list_prompt_versions(use_case: str = "estimation") -> list[str]:
    """
    Discover all complete, valid prompt template versions for a specific use case.

    A version is considered valid and complete if its directory exists under
    ``prompts/<use_case>/<version>`` and contains all required template files
    (``system.j2``, ``user.j2``, and ``examples.j2``).

    :param use_case: The use case directory name (defaults to "estimation").

    :return: A sorted list of valid prompt version identifiers (e.g., ["v1"]).
    """
    target_dir = _BASE_DIR / use_case
    if not target_dir.is_dir():
        return []

    versions: list[str] = []
    for entry in target_dir.iterdir():
        if entry.is_dir() and _REQUIRED_PROMPT_FILES.issubset({f.name for f in entry.iterdir() if f.is_file()}):
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


def resolve_prompt_version(version: str | None = None, use_case: str = "estimation") -> str:
    """
    Resolve the prompt version, prioritizing an explicit override or falling back to application settings.

    :param version: Optional explicit prompt version override.
    :param use_case: The use case directory name (defaults to "estimation").

    :return: The resolved and validated prompt version identifier.
    :raises ValueError: If the resolved version is not a valid prompt version.
    """
    target_version = version if (version is not None and version != "") else get_settings().active_prompt_version
    return validate_prompt_version(target_version, use_case)


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
    resolved_version = resolve_prompt_version(version)

    project_type = request.project_type.value if hasattr(request.project_type, "value") else str(request.project_type)
    detail_level = request.detail_level.value if hasattr(request.detail_level, "value") else str(request.detail_level)
    output_format = request.output_format.value if hasattr(request.output_format, "value") else str(request.output_format)

    context = {
        "description": request.description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "version": resolved_version,
    }

    system = _ENV.get_template(f"estimation/{resolved_version}/system.j2").render(**context)
    user = _ENV.get_template(f"estimation/{resolved_version}/user.j2").render(**context)
    return system, user
