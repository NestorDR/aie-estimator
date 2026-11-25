# tests/generation/conversation/compression/test_anchors.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.generation.conversation.compression.anchors import AnchorDetector, _AnchorClassification
from aie_estimator.generation.conversation.models import Message

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("content", "expected_anchor", "expected_rule"),
    [
        ("We signed the contract yesterday for Q1 delivery.", True, "signed_contract"),
        ("This project is under NDA and requires HIPAA compliance.", True, "nda"),
        ("The scope is frozen at the agreed MVP backlog.", True, "scope_frozen"),
        ("Our budget is locked at 45000 EUR.", True, "budget_locked"),
        ("We have a hard deadline of June 15.", True, "deadline_hard"),
        ("Can we add a dark mode toggle to the dashboard?", False, None),
    ],
)
async def test_heuristic_anchor_detector(
    content: str,
    expected_anchor: bool,
    expected_rule: str | None,
) -> None:
    """
    GIVEN a user message with or without durable commitment phrases
    WHEN AnchorDetector runs with strategy="heuristic"
    THEN commitment turns are promoted with matching rule labels and ordinary turns are not.
    """
    detector = AnchorDetector(strategy="heuristic")
    match = await detector.detect(Message(role="user", content=content))

    assert match.is_anchor is expected_anchor
    assert match.input_tokens == 0
    assert match.output_tokens == 0
    if expected_rule:
        assert expected_rule in match.matched_rules
    else:
        assert match.matched_rules == []


async def test_llm_anchor_detector_success_and_fallback() -> None:
    """
    GIVEN AnchorDetector with strategy="llm"
    WHEN call_llm succeeds vs. raises an error
    THEN success records LLM usage and failure falls back cleanly to heuristic matching.
    """
    detector = AnchorDetector(strategy="llm", model="reasoner-fast")
    msg = Message(role="user", content="We signed the SOW for phase 1.")

    with patch(
        "aie_estimator.generation.conversation.compression.anchors.call_llm",
        new=AsyncMock(
            return_value={
                "result": _AnchorClassification(is_anchor=True, reason="Signed SOW"),
                "usage": {"input_tokens": 25, "output_tokens": 8},
                "cost_usd": 0.0002,
            }
        ),
    ):
        ok_match = await detector.detect(msg)

    assert ok_match.is_anchor is True
    assert ok_match.matched_rules == ["llm_classifier"]
    assert ok_match.input_tokens == 25
    assert ok_match.output_tokens == 8
    assert ok_match.cost_usd == pytest.approx(0.0002)

    with patch(
        "aie_estimator.generation.conversation.compression.anchors.call_llm",
        new=AsyncMock(side_effect=RuntimeError("provider down")),
    ):
        fallback_match = await detector.detect(msg)

    assert fallback_match.is_anchor is True
    assert "signed_contract" in fallback_match.matched_rules
    assert fallback_match.input_tokens == 0
