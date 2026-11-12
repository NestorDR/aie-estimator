# src/aie_estimator/sessions/metadata_extractor.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.model_catalog import get_model_catalog
from aie_estimator.prompts.loader import render_metadata_extraction_prompt
from aie_estimator.schemas.estimations import EstimationResult, LOW_CONFIDENCE_THRESHOLD
from aie_estimator.services.providers import GenerationOptions, call_llm
from aie_estimator.sessions.models import MetadataExtractionResult, ProjectMetadata
from aie_estimator.settings import Settings, get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_NULL_LITERALS: Final[frozenset[str]] = frozenset({"null", "none", "n/a", "nil"})
_EXTRACTOR_MAX_TOKENS: Final[int] = 1000
_EXTRACTOR_TEMPERATURE: Final[float] = 0.0
_EXTRACTOR_MAX_RETRIES: Final[int] = 2


@dataclass(frozen=True, slots=True)
class MetadataExtractionOutcome:
    """Result of a post-turn metadata extraction attempt including auxiliary token usage and cost."""

    metadata: ProjectMetadata
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


def _normalize_optional_text(value: str | None) -> str | None:
    """Strip whitespace and discard empty or null-like string literals."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in _NULL_LITERALS:
        return None
    return cleaned


def _normalize_string_list(items: list[str]) -> list[str]:
    """Strip whitespace, drop empty/null-like items, and deduplicate case-insensitively."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_item in items:
        item = raw_item.strip()
        key = item.lower()
        if not item or key in _NULL_LITERALS or key in seen:
            continue
        cleaned.append(item)
        seen.add(key)
    return cleaned


def sanitize_extraction_delta(raw: MetadataExtractionResult) -> MetadataExtractionResult:
    """
    Apply deterministic post-validation to normalize LLM-extracted metadata deltas.

    :param raw: Raw ``MetadataExtractionResult`` returned by structured provider extraction.

    :return: Normalized ``MetadataExtractionResult`` with stripped strings, cleared null literals, and deduplicated lists.
    """
    return MetadataExtractionResult(
        project_name=_normalize_optional_text(raw.project_name),
        assumed_team_size=raw.assumed_team_size,
        mentioned_technologies=_normalize_string_list(raw.mentioned_technologies),
        agreed_scope=_normalize_optional_text(raw.agreed_scope),
        explicit_constraints=_normalize_string_list(raw.explicit_constraints),
        rejected_options=_normalize_string_list(raw.rejected_options),
        retracted_constraints=_normalize_string_list(raw.retracted_constraints),
        retracted_options=_normalize_string_list(raw.retracted_options),
    )


def resolve_extractor_options(opts: GenerationOptions, settings: Settings) -> GenerationOptions:
    """
    Build ``GenerationOptions`` for the secondary metadata extraction call.

    Uses ``settings.metadata_extractor_model`` under ``litellm`` and falls back to
    ``opts.effective_model`` when ``opts.engine`` is ``"native"`` because logical
    profiles cannot be dispatched natively.

    :param opts: Generation options used for the primary estimation turn.
    :param settings: Application configuration providing extractor model selection.

    :return: Configured ``GenerationOptions`` for metadata extraction.
    """
    target_model = opts.effective_model if opts.engine == "native" else settings.metadata_extractor_model
    spec = get_model_catalog().get(target_model)
    reasoning_effort = "low" if (spec is not None and spec.is_reasoning) else None
    return GenerationOptions(
        model=target_model,
        max_tokens=_EXTRACTOR_MAX_TOKENS,
        reasoning_effort=reasoning_effort,
        engine=opts.engine,
        temperature=_EXTRACTOR_TEMPERATURE,
        max_retries=_EXTRACTOR_MAX_RETRIES,
    )


async def update_metadata(
    *,
    previous: ProjectMetadata,
    transcript: str,
    result: EstimationResult,
    opts: GenerationOptions,
) -> MetadataExtractionOutcome:
    """
    Extract incremental project facts from the latest turn and merge them into ``previous``.

    Runs an asynchronous structured-output LLM call followed by deterministic post-validation.
    Skips disabled or low-confidence (``< 30``) turns and degrades gracefully on any error by
    logging a warning and returning ``previous`` unchanged.

    :param previous: Current ``ProjectMetadata`` stored on the session.
    :param transcript: Latest user turn transcript (optionally enriched with attachments).
    :param result: Scope-filtered ``EstimationResult`` produced for the latest turn.
    :param opts: Primary turn generation options used to select engine and native fallback model.

    :return: ``MetadataExtractionOutcome`` with updated metadata, token counts, and optional cost.
    """
    settings = get_settings()
    if not settings.enable_metadata_extraction:
        return MetadataExtractionOutcome(metadata=previous)

    if result.confidence_pct < LOW_CONFIDENCE_THRESHOLD:
        log.info(
            "metadata_extraction_skipped",
            reason="out_of_scope",
            confidence_pct=result.confidence_pct,
        )
        return MetadataExtractionOutcome(metadata=previous)

    try:
        system_prompt, user_prompt = render_metadata_extraction_prompt(
            transcript=transcript,
            result=result,
            previous=previous,
        )
        extractor_opts = resolve_extractor_options(opts, settings)
        response = await call_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            opts=extractor_opts,
            response_model=MetadataExtractionResult,
        )

        parsed_delta = response.get("result")
        if not isinstance(parsed_delta, MetadataExtractionResult):
            raise TypeError(f"Unexpected metadata extraction result type: {type(parsed_delta)}")

        sanitized_delta = sanitize_extraction_delta(parsed_delta)
        merged = previous.merge_with(sanitized_delta)

        usage = response.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        raw_cost = response.get("cost_usd")
        cost_usd = float(raw_cost) if raw_cost is not None else None

        log.info(
            "metadata_extraction_completed",
            model=response.get("model"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            project_name=merged.project_name,
            team_size=merged.assumed_team_size,
            tech_count=len(merged.mentioned_technologies),
            constraint_count=len(merged.explicit_constraints),
            rejected_count=len(merged.rejected_options),
        )
        return MetadataExtractionOutcome(
            metadata=merged,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "metadata_extraction_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:200],
        )
        return MetadataExtractionOutcome(metadata=previous)
