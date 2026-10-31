# src/aie_estimator/services/__init__.py

# --- Python Standard Library ---
from typing import Final

DEFAULT_MAX_TOKENS: Final[int] = 4000


class LLMServiceError(Exception):
    """Raised when the LLM provider or router call fails."""
