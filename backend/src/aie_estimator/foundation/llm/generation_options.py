# src/aie_estimator/foundation/llm/generation_options.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final, Literal

# --- App modules ---
from aie_estimator.model_catalog import ProviderMode, get_model_provider, resolve_engine_model


ReasoningEffort = Literal["low", "medium", "high"]

DEFAULT_MAX_TOKENS: Final[int] = 6000
DEFAULT_MAX_RETRIES: Final[int] = 3


@dataclass
class GenerationOptions:
    """Request-specific parameters controlling upstream provider execution."""

    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: ReasoningEffort | None = None
    engine: ProviderMode = "litellm"
    max_retries: int = DEFAULT_MAX_RETRIES

    @property
    def effective_model(self) -> str:
        """
        Return the requested model override or the default model for the active engine.

        :return: Resolved model or logical profile identifier.
        """
        return self.model if self.model is not None else resolve_engine_model(self.engine, None)

    @property
    def provider(self) -> str:
        """
        Resolve the upstream provider identifier for the effective model.

        :return: Provider identifier string.
        """
        return get_model_provider(self.effective_model)
