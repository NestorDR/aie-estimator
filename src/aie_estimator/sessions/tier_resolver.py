# src/aie_estimator/sessions/tier_resolver.py

# --- Python Standard Library ---
from enum import Enum
import re
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.sessions.models import ProjectMetadata

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


class Tier(str, Enum):
    """Stakeholder profile tier governing prompt framing and output schema."""

    DEVELOPER = "developer"
    PM = "pm"
    EXECUTIVE = "executive"


# NDA: Non-Disclosure Agreement
_NDA_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(nda|non[- ]?disclosure|confidential|under embargo|legal hold)\b",
    re.IGNORECASE,
)
_REGULATORY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(hipaa|gdpr|sox|soc\s?2|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b",
    re.IGNORECASE,
)
_DEV_KEYWORDS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(docker|kubernetes|k8s|microservice|microservices|terraform|iac|"
    r"helm|grpc|graphql|kafka|airflow|spark|rabbitmq|sql|nosql)\b",
    re.IGNORECASE,
)
_MIN_DEV_KEYWORD_HITS: Final[int] = 2
_SMALL_TEAM_MAX_SIZE: Final[int] = 2


def _has_nda(transcript: str, metadata: ProjectMetadata) -> bool:
    if _NDA_PATTERN.search(transcript):
        return True
    return bool(metadata.agreed_scope and _NDA_PATTERN.search(metadata.agreed_scope))


def _has_regulatory_context(transcript: str, metadata: ProjectMetadata) -> bool:
    if _REGULATORY_PATTERN.search(transcript):
        return True
    if metadata.agreed_scope and _REGULATORY_PATTERN.search(metadata.agreed_scope):
        return True
    return any(_REGULATORY_PATTERN.search(tech) for tech in metadata.mentioned_technologies)


def _is_technical_audience(transcript: str) -> bool:
    # Require at least two distinct technical keywords so a single incidental mention
    # (e.g., "we run on docker") does not flip a PM request into developer tier.
    hits = _DEV_KEYWORDS_PATTERN.findall(transcript)
    return len({token.lower() for token in hits}) >= _MIN_DEV_KEYWORD_HITS


def _is_small_team(metadata: ProjectMetadata) -> bool:
    return metadata.assumed_team_size is not None and metadata.assumed_team_size <= _SMALL_TEAM_MAX_SIZE


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata | None = None,
    caller_tier: Tier | None = None,
    override: Tier | None = None,
) -> tuple[Tier, str]:
    """
    Resolve the effective stakeholder tier and the matching rule name.

    Precedence:
    1. ``caller_tier`` from trusted ``CallerContext`` (``"caller_context"``).
    2. ``override`` from explicit request parameter (``"explicit_override"``).
    3. Ordered rule predicates over ``transcript`` and ``metadata``.
    4. Default fallback to ``Tier.PM`` (``"default"``).

    :param transcript: Current project description or turn transcript.
    :param metadata: Optional distilled ``ProjectMetadata`` for the active session.
    :param caller_tier: Optional trusted tier from ``CallerContext``.
    :param override: Optional explicit tier override supplied in the request.

    :return: Tuple of ``(resolved_tier, rule_name)``.
    """
    if caller_tier is not None:
        log.info("tier_resolved", tier=caller_tier.value, rule="caller_context")
        return caller_tier, "caller_context"

    if override is not None:
        log.info("tier_resolved", tier=override.value, rule="explicit_override")
        return override, "explicit_override"

    resolved_metadata = metadata or ProjectMetadata()
    checks = (
        ("nda_detected", Tier.EXECUTIVE, _has_nda(transcript, resolved_metadata)),
        ("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context(transcript, resolved_metadata)),
        ("technical_audience", Tier.DEVELOPER, _is_technical_audience(transcript)),
        ("small_team_pm", Tier.PM, _is_small_team(resolved_metadata)),
    )
    for rule_name, tier, matched in checks:
        if matched:
            log.info("tier_resolved", tier=tier.value, rule=rule_name)
            return tier, rule_name

    log.info("tier_resolved", tier=Tier.PM.value, rule="default")
    return Tier.PM, "default"
