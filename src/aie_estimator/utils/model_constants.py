# src/aie_estimator/utils/model_constants.py

# --- Python Standard Library ---
from typing import Final

# Logical names to group model categories, decoupling the app from specific vendors
DEFAULT_FAST_PROFILE: Final[str] = "reasoner-fast"
DEFAULT_DEEP_PROFILE: Final[str] = "reasoner-deep"

# Logical names to group model categories, decoupling the app from specific vendors
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
