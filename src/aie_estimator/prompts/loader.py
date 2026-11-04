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
from aie_estimator.schemas.estimations import PromptEstimationRequest

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

_ENV: Final[Environment] = Environment(
    loader=FileSystemLoader(_BASE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,  # noqa: S701 - Prompts are plain text messages dispatched to LLMs, not HTML in a browser.
    keep_trailing_newline=True,
)


def render_estimation_prompt(
    request: PromptEstimationRequest,
    version: str,
) -> tuple[str, str]:
    """Render the system and user prompts for the software estimation domain.

    :param request: Validated prompt estimation request parameters.
    :param version: Prompt template version identifier.

    :return: A tuple of ``(system_prompt, user_prompt)`` ready for LLM provider dispatch.
    :raises jinja2.exceptions.TemplateNotFound: If templates for the requested version do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    project_type = (
        request.project_type.value
        if hasattr(request.project_type, "value")
        else str(request.project_type)
    )
    detail_level = (
        request.detail_level.value
        if hasattr(request.detail_level, "value")
        else str(request.detail_level)
    )
    output_format = (
        request.output_format.value
        if hasattr(request.output_format, "value")
        else str(request.output_format)
    )

    context = {
        "description": request.description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }

    system = _ENV.get_template(f"estimation/{version}/system.j2").render(**context)
    user = _ENV.get_template(f"estimation/{version}/user.j2").render(**context)
    return system, user
