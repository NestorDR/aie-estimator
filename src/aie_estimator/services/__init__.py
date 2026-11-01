# src/aie_estimator/services/__init__.py

# --- Python Standard Library ---
from typing import Any, Final

DEFAULT_MAX_TOKENS: Final[int] = 4000


class LLMServiceError(Exception):
    """Raised when the LLM provider or router call fails."""


def format_provider_response(
    *,
    estimation: str,
    model: str,
    provider: str,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    """
    Standardize the internal dictionary contract across provider dispatch engines.

    :param estimation: Generated Markdown estimation text.
    :param model: Upstream model identifier reported by provider.
    :param provider: Upstream provider identifier.
    :param finish_reason: Upstream model completion finish reason.
    :param input_tokens: Ingress prompt token count.
    :param output_tokens: Egress completion token count.
    :param cost_usd: Execution cost in USD reported by the routing engine.

    :return: Standardized response dictionary for downstream evaluation and cache storage.
    """
    response: dict[str, Any] = {
        "estimation": estimation,
        "model": model,
        "provider": provider,
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    if cost_usd is not None:
        response["cost_usd"] = cost_usd
    return response

