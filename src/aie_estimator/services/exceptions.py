# src/aie_estimator/services/exceptions.py

# --- Third Party Libraries ---
from instructor.core import InstructorRetryException


class LLMProviderError(Exception):
    """Raised when an LLM provider, client connection, or routing call fails."""


def unwrap_exception(exc: Exception) -> Exception:
    """
    Extract the underlying causal exception when wrapped by InstructorRetryException.

    :param exc: The exception caught at an Instructor-wrapped provider call site.

    :return: The underlying causal exception if present, otherwise the original exception.
    """
    while isinstance(exc, InstructorRetryException) and isinstance(exc.__cause__, Exception):
        exc = exc.__cause__
    return exc
