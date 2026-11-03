# src/aie_estimator/schemas/models.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final

# Logical names to group model categories, decoupling the app from specific vendors
DEFAULT_FAST_PROFILE: Final[str] = "reasoner-fast"
DEFAULT_DEEP_PROFILE: Final[str] = "reasoner-deep"

# Default vendor models for direct native calls and fallback configuration
DEFAULT_GPT_LLM: Final[str] = "gpt-6-luna"
DEFAULT_CLAUDE_LLM: Final[str] = "claude-haiku-5-5"

UNKNOWN_PROVIDER: Final[str] = "unknown"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Metadata specification for supported large language models.

    :param provider: Upstream LLM provider identifier ('openai', 'anthropic', 'litellm', etc.).
    :param is_reasoning: Whether the model supports chain-of-thought reasoning / thinking capabilities.
    :param is_profile: Whether this entry represents a logical routing profile rather than a physical model.
    """

    provider: str
    is_reasoning: bool = False
    is_profile: bool = False
