# src/aie_estimator/utils/constants.py

"""
Shared domain constants and precompiled regular expressions.

Centralizes cross-cutting defaults and PII regex patterns so online guardrails and
offline ingestion recognizers import only the exact patterns they need without
duplicating regex definitions.
"""

# --- Python Standard Library ---
import re
from typing import Final

# region Time

SECONDS_PER_DAY: Final[int] = 86_400

# endregion

# region Models & Providers

# Logical names to group model categories, decoupling the app from specific vendors
DEFAULT_FAST_PROFILE: Final[str] = "reasoner-fast"
DEFAULT_DEEP_PROFILE: Final[str] = "reasoner-deep"

LOGICAL_PROFILE_NAMES: Final[set[str]] = {DEFAULT_FAST_PROFILE, DEFAULT_DEEP_PROFILE}

# Default vendor models for direct native calls and fallback configuration
DEFAULT_GPT_LLM: Final[str] = "gpt-6-luna"
DEFAULT_CLAUDE_LLM: Final[str] = "claude-haiku-5-5"

UNKNOWN_PROVIDER: Final[str] = "unknown"

# Supported LLM providers and their model prefixes, used to identify vendors and map API key env vars
PROVIDER_PREFIXES: Final[dict[str, tuple[str, ...]]] = {
    "openai": ("openai/", "gpt-"),
    "anthropic": ("anthropic/", "claude-"),
    "gemini": ("gemini/",),
    "mistral": ("mistral/",),
    "scaleway": ("scaleway/",),
    "nebius": ("nebius/",),
}

# endregion

# region Conversational Sessions

DEFAULT_SESSION_MAX_TURNS: Final[int] = 6
DEFAULT_SESSION_MAX_ANCHOR_PAIRS: Final[int] = 4
DEFAULT_SESSION_TTL_SECONDS: Final[int] = SECONDS_PER_DAY

# endregion

# region Ingestion

# Disguised null strings normalized to missing values across tabular and text cleaners
NULL_PLACEHOLDERS: Final[frozenset[str]] = frozenset({"tbd", "n/a", "na", "", "null", "none", "-"})

# endregion

# region PII Patterns

# Standard RFC 5322-style email addresses requiring a >=2-char alphabetic TLD to avoid matching code attributes.
PII_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# International Bank Account Numbers (ISO country code + 2 check digits + 10-30 alphanumeric BBAN characters).
PII_IBAN_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

# International phone numbers and 10-to-13-digit numeric identifiers (avoids dates, IPs, and semver).
PII_NUMERIC_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d")

# Major credit card formats (Visa, MasterCard, AmEx, Discover, and generic 4x4 grouped 16-digit cards).
PII_CREDIT_CARD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|2[2-7]\d{2}|6(?:011|5\d{2}))[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    r"|"
    r"\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b"
    r"|"
    r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"
)

# Canonical internal budget identifiers (e.g., BUDGET-2024-0001).
PII_BUDGET_ID_RE: Final[re.Pattern[str]] = re.compile(r"\bBUDGET-\d{4}-\d{4}\b")

# Internal client codes (e.g., CLI-0042, CLT-INT-A047).
PII_CLIENT_CODE_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:CLI|CLT-INT)-[A-Z0-9]{3,8}\b")

# endregion
