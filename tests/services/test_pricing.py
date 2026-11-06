# tests/services/test_pricing.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.services.providers.pricing import estimate_cost


@pytest.mark.parametrize(
    ("model", "tokens_in", "tokens_out", "expected_cost"),
    [
        ("gpt-6-luna", 1_000_000, 1_000_000, 0.60),
        ("openai/gpt-6-luna", 500_000, 200_000, 0.15),
        ("gpt-4o-mini", 0, 0, 0.0),
        ("  GPT-4O-MINI  ", 1_000_000, 1_000_000, 0.75),
        ("claude-haiku-5-5-20261022", 1_000, 1_000, 0.0015),
    ],
)
def test_estimate_cost_configured_models(
    model: str,
    tokens_in: int,
    tokens_out: int,
    expected_cost: float,
) -> None:
    """
    GIVEN configured model names and token counts
    WHEN estimate_cost is executed
    THEN it calculates and rounds execution cost in USD accurately.
    """
    assert estimate_cost(model, tokens_in, tokens_out) == expected_cost


@pytest.mark.parametrize(
    "unknown_model",
    [
        "unsupported-model-x",
        "deepseek-ai/DeepSeek-V4-Flash",
        "",
        "unknown/model",
    ],
)
def test_estimate_cost_unconfigured_model_returns_none(unknown_model: str) -> None:
    """
    GIVEN an unconfigured or empty model identifier
    WHEN estimate_cost is executed
    THEN it returns None so callers can treat cost as unknown.
    """
    assert estimate_cost(unknown_model, 1_000, 1_000) is None
