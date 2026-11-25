# tests/evals/test_stress_metrics.py

# --- Python Standard Library ---
from datetime import UTC, datetime

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.sessions import SessionInfoResponse
from aie_estimator.sessions.models import ProjectMetadata, TurnObservation
from evals.stress.metrics import (
    DEFAULT_COST_BUDGET_USD,
    DEFAULT_LATENCY_BUDGET_MS,
    CostBudgetMetric,
    FactField,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)


def _make_observation(*, latency_ms: int = 1200, cost_usd: float | None = 0.01) -> TurnObservation:
    """Build a minimal ``TurnObservation`` fixture for stress metric unit tests."""
    return TurnObservation(
        turn_index=1,
        session_id="sess-1",
        enriched_transcript_chars=120,
        attachments_total_chars=0,
        attachments_truncated_chars=0,
        messages_in_window=2,
        anchors_count=0,
        summary_chars=0,
        tokens_in=200,
        tokens_out=100,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cache_hit_kind="none",
        last_resolved_tier="pm",
    )


def _make_snapshot(
    *,
    project_name: str | None = "Nimbus",
    mentioned_technologies: list[str] | None = None,
    agreed_scope: str | None = "Vendor catalogue and RFQ workflow",
    summary: str | None = "Initial release budget is 60k EUR.",
) -> SessionInfoResponse:
    """Build a minimal ``SessionInfoResponse`` fixture for memory-drift tests."""
    now = datetime.now(UTC)
    return SessionInfoResponse(
        session_id="sess-1",
        turn_index=3,
        message_count=6,
        max_turns=6,
        has_summary=summary is not None,
        summary_chars=len(summary) if summary else 0,
        summary=summary,
        metadata=ProjectMetadata(
            project_name=project_name,
            mentioned_technologies=mentioned_technologies if mentioned_technologies is not None else ["React", "Postgres"],
            agreed_scope=agreed_scope,
        ),
        last_tier_rule="explicit_override",
        created_at=now,
        last_accessed_at=now,
    )


@pytest.mark.parametrize(
    ("latency_ms", "budget_ms", "expected_passed"),
    [
        (1200, DEFAULT_LATENCY_BUDGET_MS, True),
        (8000, 8000, True),
        (8001, 8000, False),
    ],
)
def test_latency_budget_metric(
    latency_ms: int,
    budget_ms: int,
    expected_passed: bool,
) -> None:
    """
    GIVEN a ``TurnObservation`` with a given ``latency_ms`` and a ``LatencyBudgetMetric``
    WHEN ``evaluate`` is called
    THEN latency <= budget passes with score 1.0 and latency > budget fails with score 0.0.
    """
    result = LatencyBudgetMetric(budget_ms=budget_ms).evaluate(_make_observation(latency_ms=latency_ms))

    assert result.name == "latency_budget"
    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)
    assert f"{latency_ms} ms vs budget {budget_ms} ms" == result.details


@pytest.mark.parametrize(
    ("cost_usd", "budget_usd", "expected_passed", "expected_detail_fragment"),
    [
        (0.01, DEFAULT_COST_BUDGET_USD, True, "$0.010000 vs budget $0.050000"),
        (0.05, 0.05, True, "$0.050000 vs budget $0.050000"),
        (0.050001, 0.05, False, "$0.050001 vs budget $0.050000"),
        (None, DEFAULT_COST_BUDGET_USD, False, "unpriced turn"),
    ],
)
def test_cost_budget_metric(
    cost_usd: float | None,
    budget_usd: float,
    expected_passed: bool,
    expected_detail_fragment: str,
) -> None:
    """
    GIVEN a ``TurnObservation`` with numeric or ``None`` ``cost_usd``
    WHEN ``CostBudgetMetric.evaluate`` is called
    THEN cost <= budget passes, cost > budget fails, and ``None`` explicitly fails as unpriced.
    """
    result = CostBudgetMetric(budget_usd=budget_usd).evaluate(_make_observation(cost_usd=cost_usd))

    assert result.name == "cost_budget"
    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)
    assert expected_detail_fragment in result.details


@pytest.mark.parametrize(
    ("fact", "fact_field", "expected_passed"),
    [
        ("nimbus", "project_name", True),
        ("postgres", "technologies", True),
        ("rfq workflow", "scope", True),
        ("60k eur", "summary", True),
        ("explicit_override", "any", True),
        ("missing-fact", "any", False),
    ],
)
def test_memory_drift_metric_field_matching(
    fact: str,
    fact_field: FactField,
    expected_passed: bool,
) -> None:
    """
    GIVEN a ``SessionInfoResponse`` snapshot and a ``FactField`` target
    WHEN ``MemoryDriftMetric.evaluate`` is called
    THEN case-insensitive substrings present in the selected field pass and absent ones fail.
    """
    result = MemoryDriftMetric(fact=fact, fact_field=fact_field).evaluate(_make_snapshot())

    assert result.name == "memory_drift"
    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)
    assert ("present" if expected_passed else "missing") in result.details


@pytest.mark.parametrize("fact_field", ["project_name", "technologies", "scope", "summary"])
def test_memory_drift_metric_handles_empty_or_none_fields(fact_field: FactField) -> None:
    """
    GIVEN a ``SessionInfoResponse`` whose targeted metadata or summary fields are ``None`` or empty
    WHEN ``MemoryDriftMetric.evaluate`` is called
    THEN it safely returns ``passed=False`` with ``score=0.0``.
    """
    snapshot = _make_snapshot(
        project_name=None,
        mentioned_technologies=[],
        agreed_scope=None,
        summary=None,
    )
    result = MemoryDriftMetric(fact="Nimbus", fact_field=fact_field).evaluate(snapshot)

    assert result.passed is False
    assert result.score == 0.0
    assert "missing" in result.details
