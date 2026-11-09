# tests/services/test_exceptions.py

# --- Third Party Libraries ---
from instructor.core import InstructorRetryException
import pytest

# --- App modules ---
from aie_estimator.services.exceptions import LLMProviderError, unwrap_exception


def _create_retry_exception(cause: Exception | None) -> InstructorRetryException:
    """
    Helper to construct an InstructorRetryException with a specific __cause__.

    :param cause: Inner exception to set as cause.
    :return: Constructed InstructorRetryException.
    """
    exc = InstructorRetryException(
        "Retry failed",
        last_completion=None,
        messages=[],
        n_attempts=1,
        total_usage=None,
    )
    exc.__cause__ = cause
    return exc


@pytest.mark.parametrize(
    ("input_exception", "expected_unwrapped"),
    [
        (ValueError("plain error"), ValueError),
        (LLMProviderError("provider error"), LLMProviderError),
        (_create_retry_exception(ValueError("inner cause")), ValueError),
        (_create_retry_exception(_create_retry_exception(RuntimeError("nested cause"))), RuntimeError),
        (_create_retry_exception(None), InstructorRetryException),
    ],
    ids=[
        "unwrapped_standard_error",
        "unwrapped_provider_error",
        "single_wrapped_retry_exception",
        "nested_wrapped_retry_exception",
        "retry_exception_without_cause",
    ],
)
def test_unwrap_exception(input_exception: Exception, expected_unwrapped: type[Exception]) -> None:
    """
    GIVEN an exception that may or may not be wrapped inside InstructorRetryException
    WHEN unwrap_exception is called
    THEN the innermost causal exception or the original exception is returned.

    :param input_exception: The input exception to unwrap.
    :param expected_unwrapped: Expected type of the unwrapped exception.
    """
    result = unwrap_exception(input_exception)
    assert isinstance(result, expected_unwrapped)

