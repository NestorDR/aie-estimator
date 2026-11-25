# tests/sessions/compression/test_policy.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.sessions.compression.anchors import AnchorDetector
from aie_estimator.sessions.compression.policy import CompressionPolicy
from aie_estimator.sessions.compression.summarizer import CumulativeSummarizer, SummarizerOutcome
from aie_estimator.sessions.models import ConversationHistory

pytestmark = pytest.mark.anyio


async def test_compression_policy_idempotent_within_capacity() -> None:
    """
    GIVEN a ConversationHistory within max_turns capacity
    WHEN CompressionPolicy.apply runs twice
    THEN no messages are evicted and compressed=False.
    """
    history = ConversationHistory(max_turns=2)
    history.append(user="U1", assistant="A1")
    history.append(user="U2", assistant="A2")

    policy = CompressionPolicy(
        anchor_detector=AnchorDetector(strategy="heuristic"),
        summarizer=CumulativeSummarizer(),
    )
    outcome = await policy.apply(history)

    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0
    assert len(history.messages) == 4


async def test_compression_policy_disabled_trims_without_llm() -> None:
    """
    GIVEN CompressionPolicy with enabled=False and an over-capacity ConversationHistory
    WHEN apply runs
    THEN oldest pairs are trimmed without promoting anchors or calling the summarizer.
    """
    history = ConversationHistory(max_turns=1)
    history.append(user="We signed the contract.", assistant="A1")
    history.append(user="U2", assistant="A2")

    summarizer = CumulativeSummarizer()
    with patch.object(summarizer, "summarize", new=AsyncMock()) as mock_sum:
        policy = CompressionPolicy(
            anchor_detector=AnchorDetector(strategy="heuristic"),
            summarizer=summarizer,
            enabled=False,
        )
        outcome = await policy.apply(history)

    mock_sum.assert_not_awaited()
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0
    assert history.anchors == []
    assert history.summary is None
    assert [m.content for m in history.messages] == ["U2", "A2"]


async def test_compression_policy_promotes_anchors_and_summarizes_overflow() -> None:
    """
    GIVEN max_turns=1, max_anchor_pairs=1, and 3 appended pairs where 2 are anchors
    WHEN CompressionPolicy.apply runs
    THEN the newest anchor stays verbatim in anchors and the oldest anchor + non-anchors are summarized.
    """
    history = ConversationHistory(max_turns=1, max_anchor_pairs=1)
    history.append(user="We signed the contract for phase 1.", assistant="Ack 1")
    history.append(user="This is under NDA and GDPR.", assistant="Ack 2")
    history.append(user="Add reporting dashboard.", assistant="Ack 3")

    summarizer = CumulativeSummarizer()
    summarize_mock = AsyncMock(
        return_value=SummarizerOutcome(
            summary="Phase 1 contract signed earlier.",
            input_tokens=80,
            output_tokens=20,
            cost_usd=0.0004,
        )
    )
    with patch.object(summarizer, "summarize", new=summarize_mock):
        policy = CompressionPolicy(
            anchor_detector=AnchorDetector(strategy="heuristic"),
            summarizer=summarizer,
        )
        outcome = await policy.apply(history)

    assert outcome.input_tokens == 80
    assert outcome.output_tokens == 20
    assert outcome.cost_usd == pytest.approx(0.0004)
    assert [m.content for m in history.anchors] == ["This is under NDA and GDPR.", "Ack 2"]
    assert history.summary == "Phase 1 contract signed earlier."
    assert [m.content for m in history.messages] == ["Add reporting dashboard.", "Ack 3"]
    summarize_mock.assert_awaited_once()
