# tests/sessions/compression/test_summarizer.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.sessions.compression.summarizer import CumulativeSummarizer, _SummaryEnvelope
from aie_estimator.sessions.models import Message

pytestmark = pytest.mark.anyio


async def test_summarizer_empty_evicted_is_noop() -> None:
    """
    GIVEN no evicted messages
    WHEN CumulativeSummarizer.summarize is called
    THEN it returns previous_summary unchanged with zero token usage.
    """
    summarizer = CumulativeSummarizer()
    outcome = await summarizer.summarize(previous_summary="Existing summary.", evicted=[])

    assert outcome.summary == "Existing summary."
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0


async def test_summarizer_updates_summary_and_records_telemetry() -> None:
    """
    GIVEN evicted messages and a successful call_llm response
    WHEN CumulativeSummarizer.summarize runs
    THEN it returns the trimmed summary with updated=True and LLM usage/cost telemetry.
    """
    summarizer = CumulativeSummarizer(model="reasoner-fast")
    evicted = [
        Message(role="user", content="Use FastAPI and PostgreSQL."),
        Message(role="assistant", content="Estimated 4 weeks."),
    ]

    with patch(
        "aie_estimator.sessions.compression.summarizer.call_llm",
        new=AsyncMock(
            return_value={
                "result": _SummaryEnvelope(summary="  Client selected FastAPI and PostgreSQL.  "),
                "usage": {"input_tokens": 110, "output_tokens": 35},
                "cost_usd": 0.0005,
                "model": "gpt-6-luna",
            }
        ),
    ):
        outcome = await summarizer.summarize(previous_summary="Initial discovery.", evicted=evicted)

    assert outcome.summary == "Client selected FastAPI and PostgreSQL."
    assert outcome.input_tokens == 110
    assert outcome.output_tokens == 35
    assert outcome.cost_usd == pytest.approx(0.0005)


async def test_summarizer_failure_preserves_previous_summary() -> None:
    """
    GIVEN an LLM failure during cumulative summarization
    WHEN CumulativeSummarizer.summarize runs
    THEN it degrades gracefully by returning previous_summary unchanged with updated=False.
    """
    summarizer = CumulativeSummarizer()
    evicted = [Message(role="user", content="Turn 1"), Message(role="assistant", content="Resp 1")]

    with patch(
        "aie_estimator.sessions.compression.summarizer.call_llm",
        new=AsyncMock(side_effect=RuntimeError("timeout")),
    ):
        outcome = await summarizer.summarize(previous_summary="Prior durable summary.", evicted=evicted)

    assert outcome.summary == "Prior durable summary."
    assert outcome.input_tokens == 0
    assert outcome.output_tokens == 0
