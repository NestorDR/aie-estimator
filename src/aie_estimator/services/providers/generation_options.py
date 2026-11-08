# src/aie_estimator/services/providers/generation_options.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final, Literal

# --- App modules ---
from aie_estimator.model_catalog import get_model_provider
from aie_estimator.schemas.estimations import ProviderMode
from aie_estimator.settings import get_settings


ReasoningEffort = Literal["low", "medium", "high"]

DEFAULT_MAX_TOKENS: Final[int] = 4000
DEFAULT_TEMPERATURE: Final[float] = 0.3
DEFAULT_MAX_RETRIES: Final[int] = 3


@dataclass
class GenerationOptions:
    """Request-specific parameters controlling upstream provider execution."""

    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: ReasoningEffort | None = None
    engine: ProviderMode = "litellm"
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES

    @property
    def effective_model(self) -> str:
        """
        Return the requested model override or the default model from settings.

        :return: Resolved model or logical profile identifier.
        """
        return self.model or get_settings().llm_model

    @property
    def provider(self) -> str:
        """
        Resolve the upstream provider identifier for the effective model.

        :return: Provider identifier string.
        """
        return get_model_provider(self.effective_model)
