# src/aie_estimator/services/providers/formatting.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from pydantic import BaseModel

# --- App modules ---
from aie_estimator.services.providers.pricing import estimate_cost


def normalize_turn_messages(
    user_prompt: str | None,
    messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """
    Validate and normalize single-turn or multi-turn non-system conversation messages.

    Enforces mutual exclusion between ``user_prompt`` and ``messages``, rejects empty
    message lists, and rejects ``"system"`` role entries inside ``messages``.

    :param user_prompt: Single-turn user prompt string, or ``None`` when ``messages`` is supplied.
    :param messages: Optional list of prior and current ``user``/``assistant`` message dicts.

    :return: Normalized list of ``{"role": ..., "content": ...}`` turn dictionaries.
    :raises ValueError: If inputs are ambiguous, missing, empty, or contain a ``"system"`` role.
    """
    if (user_prompt is None) == (messages is None):
        raise ValueError("Provide either 'user_prompt' or 'messages', but not both or neither.")
    if user_prompt is not None:
        return [{"role": "user", "content": user_prompt}]
    if not messages:
        raise ValueError("'messages' cannot be empty.")
    if any(entry.get("role") == "system" for entry in messages):
        raise ValueError("'messages' must not contain a 'system' role entry; pass 'system_prompt' instead.")
    return list(messages)


def build_chat_messages(
    user_prompt: str | None,
    system_prompt: str | None,
    messages: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Assemble an OpenAI/LiteLLM-compatible chat message array with an optional leading system message.

    :param user_prompt: Single-turn user prompt string, or ``None`` when ``messages`` is supplied.
    :param system_prompt: Optional system instructions placed as the initial ``system`` message.
    :param messages: Optional multi-turn list of ``user``/``assistant`` message dicts.

    :return: Ordered list of chat message dictionaries ready for OpenAI or LiteLLM dispatch.
    :raises ValueError: If turn inputs violate :func:`normalize_turn_messages` invariants.
    """
    turn_messages = normalize_turn_messages(user_prompt, messages)
    formatted: list[dict[str, Any]] = []
    if system_prompt:
        formatted.append({"role": "system", "content": system_prompt})
    formatted.extend(turn_messages)
    return formatted


def format_provider_response(
    *,
    result: BaseModel,
    model: str,
    provider: str,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
) -> dict[str, Any]:
    """
    Standardize the internal dictionary contract across provider dispatch engines.

    :param result: Validated Pydantic structured output instance.
    :param model: Upstream model identifier reported by provider.
    :param provider: Upstream provider identifier.
    :param finish_reason: Upstream model completion finish reason.
    :param input_tokens: Ingress prompt token count.
    :param output_tokens: Egress completion token count.
    :param cost_usd: Execution cost in USD reported by the routing engine, or None to calculate fallback.

    :return: Standardized response dictionary for downstream cache storage and router responses.
    """
    response: dict[str, Any] = {
        "result": result,
        "model": model,
        "provider": provider,
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }
    resolved_cost = cost_usd if cost_usd is not None else estimate_cost(model=model, tokens_in=input_tokens, tokens_out=output_tokens)
    if resolved_cost is not None:
        response["cost_usd"] = resolved_cost
    return response
