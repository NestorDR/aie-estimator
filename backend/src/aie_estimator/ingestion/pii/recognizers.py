# src/aie_estimator/ingestion/pii/recognizers.py

"""
Domain-specific Presidio recognizers for proprietary enterprise identifiers.

Presidio detects universal PII (``PERSON``, ``EMAIL_ADDRESS``, ``LOCATION``, etc.) out of
the box, but estimation corpora also carry commercial identifiers—canonical budget IDs
(``BUDGET-YYYY-NNNN``), internal client codes (``CLI-*`` / ``CLT-INT-*``), and known client
organization names—that reveal sensitive business relationships.

These deterministic recognizers emit high-confidence scores (``0.95``) so they win overlap
resolution against generic numeric or NER (Named Entity Recognition) detections, share
canonical regexes with online guardrails via ``utils/constants.py``, and register under the
configured language so they activate alongside the spaCy NLP pipeline.
"""

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from presidio_analyzer import Pattern, PatternRecognizer

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.utils.constants import PII_BUDGET_ID_RE, PII_CLIENT_CODE_RE

KNOWN_CLIENT_ORGANIZATIONS: Final[tuple[str, ...]] = (
    "Acme Spain Ltd.",
    "BetaNorte Distribution",
    "Levante Cooperative",
    "Toledo Industries",
)
# 0.95 beats spaCy's default 0.85 NER (Named Entity Recognition) score so domain entities
# (e.g., "Acme Spain Ltd.") win overlap resolution.
_DOMAIN_PATTERN_SCORE: Final[float] = 0.95


def build_domain_recognizers(
    *,
    supported_language: str = "en",
    client_names: list[str] | None = None,
) -> list[PatternRecognizer]:
    """
    Construct the domain-specific Presidio recognizers for budget IDs, client codes, and known clients.

    :param supported_language: ISO 639-1 language code registered with Presidio.
    :param client_names: Optional exact client organization strings; defaults to ``KNOWN_CLIENT_ORGANIZATIONS``.

    :return: List of configured PatternRecognizer instances.
    """
    return [
        PatternRecognizer(
            supported_entity=PiiFlag.BUDGET_ID.value,
            name="BudgetIdRecognizer",
            patterns=[Pattern(name="budget_id_canonical", regex=PII_BUDGET_ID_RE.pattern, score=_DOMAIN_PATTERN_SCORE)],
            supported_language=supported_language,
        ),
        PatternRecognizer(
            supported_entity=PiiFlag.CLIENT_CODE.value,
            name="ClientCodeRecognizer",
            patterns=[Pattern(name="client_code_internal", regex=PII_CLIENT_CODE_RE.pattern, score=_DOMAIN_PATTERN_SCORE)],
            supported_language=supported_language,
        ),
        PatternRecognizer(
            supported_entity=PiiFlag.ORGANIZATION.value,
            name="KnownClientRecognizer",
            # Presidio's `deny_list` is an exact-match list of sensitive terms TO flag (not an ignore list).
            deny_list=client_names if client_names is not None else list(KNOWN_CLIENT_ORGANIZATIONS),
            deny_list_score=_DOMAIN_PATTERN_SCORE,
            supported_language=supported_language,
        ),
    ]
