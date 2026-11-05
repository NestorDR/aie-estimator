# src/aie_estimator/services/exceptions.py


class LLMProviderError(Exception):
    """Raised when an LLM provider, client connection, or routing call fails."""
