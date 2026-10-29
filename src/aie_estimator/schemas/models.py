# src/aie_estimator/schemas/models.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    Metadata specification for supported large language models.

    :param provider: Upstream LLM provider identifier ('openai' or 'anthropic').
    :param is_reasoning: Whether the model supports chain-of-thought reasoning / thinking capabilities.
    """

    provider: Literal["openai", "anthropic"]
    is_reasoning: bool = False


MODEL_CATALOG: Final[dict[str, ModelSpec]] = {
    # OpenAI Standard
    "gpt-4o-mini": ModelSpec(provider="openai", is_reasoning=False),
    "gpt-4o": ModelSpec(provider="openai", is_reasoning=False),
    # OpenAI Reasoning
    "o3-mini": ModelSpec(provider="openai", is_reasoning=True),
    "gpt-5-nano": ModelSpec(provider="openai", is_reasoning=True),
    "gpt-6-luna": ModelSpec(provider="openai", is_reasoning=True),
    # Anthropic
    "claude-haiku-4-5-20251001": ModelSpec(provider="anthropic", is_reasoning=True),
}

SUPPORTED_MODELS: Final[tuple[str, ...]] = tuple(MODEL_CATALOG.keys())
