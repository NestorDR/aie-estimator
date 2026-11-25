# src/aie_estimator/services/providers/pricing.py

# --- Python Standard Library ---
from typing import Final


# Cost per 1M tokens in USD. Update as vendor pricing changes.
MODEL_COSTS: Final[dict[str, dict[str, float]]] = {
    # OpenAI - https://openai.com/business/pricing/#api
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    # Anthropic - https://platform.claude.com/docs/en/about-claude/pricing
    "claude-haiku-5-5": {"input": 0.25, "output": 1.25},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    # Google - https://ai.google.dev/gemini-api/docs/pricing
    "gemini-3.5-flash-lite": {"input": 0.15, "output": 0.125},
    "gemini-3.8-flash": {"input": 0.375, "output": 1.875},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """
    Estimate execution cost in USD based on input and output token counts.

    :param model: Upstream model identifier or routing name.
    :param tokens_in: Ingress prompt token count.
    :param tokens_out: Egress completion token count.

    :return: Calculated cost in USD rounded to 6 decimal places, or None if the model is unconfigured.
    """
    # Normalize vendor-prefixed identifiers (e.g. "openai/gpt-6-luna") to base cost catalog keys
    clean_model = model.strip().lower().split("/")[-1]

    costs = MODEL_COSTS.get(clean_model)
    if costs is None:
        for prefix, rates in MODEL_COSTS.items():
            if clean_model.startswith(prefix):
                costs = rates
                break

    if costs is None:
        return None

    cost = (tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000
    return round(cost, 6)
