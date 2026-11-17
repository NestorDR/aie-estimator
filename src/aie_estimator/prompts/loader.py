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
from aie_estimator.prompts.tier_config import TIER_CONFIG
from aie_estimator.schemas.acb import CriticFeedback
from aie_estimator.schemas.estimations import (
    BaseTierEstimation,
    DetailLevel,
    OutputFormat,
    ProjectType,
)
from aie_estimator.sessions.models import Message, ProjectMetadata
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.settings import get_settings

_BASE_DIR: Final[Path] = Path(__file__).resolve().parent

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


def resolve_prompt_version(
    version: str | None = None,
    use_case: str = "estimation",
) -> str:
    """
    Resolve the prompt version, prioritizing an explicit override or falling back to settings.

    :param version: Optional explicit prompt version override.
    :param use_case: The use case directory name (defaults to "estimation").

    :return: The resolved prompt version identifier.
    :raises ValueError: If the resolved version directory does not exist.
    """
    target_version = version or get_settings().active_prompt_version
    if not (_BASE_DIR / use_case / target_version).is_dir():
        raise ValueError(f"Invalid prompt version '{target_version}' for use case '{use_case}'.")
    return target_version


def render_tier_estimation_prompt(
    *,
    description: str,
    project_type: ProjectType | str,
    detail_level: DetailLevel | str,
    output_format: OutputFormat | str,
    tier: Tier | str,
    version: str | None = None,
    metadata: ProjectMetadata | None = None,
    conversation_summary: str | None = None,
    anchored_turns: list[Message] | None = None,
    critic_feedback: CriticFeedback | None = None,
) -> tuple[str, str]:
    """
    Render tier-specific ``estimation/<version>/<tier>.j2`` and ``user.j2`` templates.

    :param description: Project description or enriched conversational turn transcript.
    :param project_type: Target project domain category.
    :param detail_level: Desired breakdown depth.
    :param output_format: Desired presentation layout.
    :param tier: Resolved stakeholder ``Tier`` selecting the persona template.
    :param version: Optional prompt template version override; defaults to ``active_prompt_version``.
    :param metadata: Optional distilled session ``ProjectMetadata`` for conversational sessions.
    :param conversation_summary: Optional rolling summary of older evicted turns.
    :param anchored_turns: Optional verbatim anchored turns retained across compression.
    :param critic_feedback: Optional structured ``CriticFeedback`` for Actor revision passes.

    :return: Tuple of ``(system_prompt, user_prompt)`` ready for LLM provider dispatch.
    :raises ValueError: If the resolved prompt version or tier is invalid.
    :raises jinja2.exceptions.TemplateNotFound: If templates for the requested version do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    resolved_version = resolve_prompt_version(version)
    tier_config = TIER_CONFIG[tier]
    context: dict[str, Any] = {
        "description": description,
        "project_type": _enum_value(project_type),
        "detail_level": _enum_value(detail_level),
        "output_format": _enum_value(output_format),
        "version": resolved_version,
        "metadata": metadata,
        "conversation_summary": conversation_summary,
        "anchored_turns": anchored_turns or [],
        "critic_feedback": critic_feedback,
    }
    system = _ENV.get_template(f"estimation/{resolved_version}/{tier_config.template}").render(**context)
    user = _ENV.get_template(f"estimation/{resolved_version}/user.j2").render(**context)
    return system, user


def render_metadata_extraction_prompt(
    *,
    transcript: str,
    result: BaseTierEstimation,
    previous: ProjectMetadata,
) -> tuple[str, str]:
    """
    Render system and user prompts for post-turn ``ProjectMetadata`` delta extraction.

    :param transcript: Latest user turn transcript (optionally enriched with attachment text).
    :param result: Validated tier estimation result produced for the latest turn.
    :param previous: Current ``ProjectMetadata`` state prior to this turn's update.

    :return: A tuple of ``(system_prompt, user_prompt)`` ready for structured LLM extraction.
    :raises jinja2.exceptions.TemplateNotFound: If templates for ``metadata_extraction/v1`` do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    system = _ENV.get_template("metadata_extraction/v1/system.j2").render(
        transcript=transcript,
        result=result,
        previous=previous,
    )
    user = _ENV.get_template("metadata_extraction/v1/user.j2").render(
        transcript=transcript,
        result=result,
        previous=previous,
    )
    return system, user


def render_critic_prompt(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    tier: Tier | str,
    result: BaseTierEstimation,
) -> tuple[str, str]:
    """
    Render system and user prompts for structured Critic auditing of a candidate estimation.

    :param transcript: Enriched user turn transcript under evaluation.
    :param metadata: Accumulated session ``ProjectMetadata``.
    :param tier: Resolved stakeholder ``Tier`` for the candidate estimation.
    :param result: Validated candidate tier estimation to audit.

    :return: Tuple of ``(system_prompt, user_prompt)`` ready for structured ``CriticFeedback`` dispatch.
    :raises jinja2.exceptions.TemplateNotFound: If templates for ``critic/v1`` do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    context: dict[str, Any] = {
        "transcript": transcript,
        "metadata": metadata,
        "tier": _enum_value(tier),
        "result_json": result.model_dump_json(indent=2),
    }
    system = _ENV.get_template("critic/v1/system.j2").render(**context)
    user = _ENV.get_template("critic/v1/user.j2").render(**context)
    return system, user


def render_conversation_summary_prompt(
    *,
    previous_summary: str | None,
    evicted: list[Message],
) -> tuple[str, str]:
    """
    Render system and user prompts for cumulative summarization of evicted conversation turns.

    :param previous_summary: Optional prior rolling summary text.
    :param evicted: Chronologically ordered ``Message`` entries evicted from the sliding window.

    :return: Tuple of ``(system_prompt, user_prompt)`` ready for structured LLM summarization.
    :raises jinja2.exceptions.TemplateNotFound: If templates for ``conversation_summary/v1`` do not exist.
    :raises jinja2.exceptions.UndefinedError: If required context parameters are missing during rendering.
    """
    context: dict[str, Any] = {
        "previous_summary": previous_summary,
        "evicted": evicted,
    }
    system = _ENV.get_template("conversation_summary/v1/system.j2").render(**context)
    user = _ENV.get_template("conversation_summary/v1/user.j2").render(**context)
    return system, user
