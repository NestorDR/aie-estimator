# src/aie_estimator/utils/constants.py

# --- Python Standard Library ---
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
