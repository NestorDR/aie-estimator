# evals/stress/metrics.py

"""
Deterministic stress evaluation metrics over ``TurnObservation`` and ``SessionInfoResponse``.

These evaluators are separated from ``evals.metrics`` because they score per-turn telemetry
and post-turn session snapshots rather than ``(GoldenCase, EstimationResponse)`` pairs.
They reuse ``MetricResult`` from ``evals.metrics`` so reporting remains uniform.
"""

# --- Python Standard Library ---
from typing import Final, Literal

# --- App modules ---
from aie_estimator.schemas.sessions import SessionInfoResponse
from aie_estimator.sessions.models import TurnObservation
from evals.metrics import MetricResult

DEFAULT_LATENCY_BUDGET_MS: Final[int] = 8000
DEFAULT_COST_BUDGET_USD: Final[float] = 0.05

FactField = Literal["project_name", "technologies", "scope", "summary", "any"]


class LatencyBudgetMetric:
    """Evaluate whether a single turn's latency stays within a millisecond budget."""

    name: str = "latency_budget"

    def __init__(self, budget_ms: int = DEFAULT_LATENCY_BUDGET_MS) -> None:
        """
        Initialize the latency budget evaluator.

        :param budget_ms: Maximum allowed per-turn latency in milliseconds.
        """
        self.budget_ms = budget_ms

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        """
        Score ``observation.latency_ms`` against ``self.budget_ms``.

        :param observation: Per-turn telemetry snapshot.

        :return: Deterministic ``MetricResult`` for ``latency_budget``.
        """
        passed = observation.latency_ms <= self.budget_ms
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"{observation.latency_ms} ms vs budget {self.budget_ms} ms",
        )


class CostBudgetMetric:
    """Evaluate whether a single turn's cost stays within a USD budget."""

    name: str = "cost_budget"

    def __init__(self, budget_usd: float = DEFAULT_COST_BUDGET_USD) -> None:
        """
        Initialize the cost budget evaluator.

        :param budget_usd: Maximum allowed per-turn cost in USD.
        """
        self.budget_usd = budget_usd

    def evaluate(self, observation: TurnObservation) -> MetricResult:
        """
        Score ``observation.cost_usd`` against ``self.budget_usd``, failing on ``None``.

        :param observation: Per-turn telemetry snapshot.

        :return: Deterministic ``MetricResult`` for ``cost_budget``.
        """
        if observation.cost_usd is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details="cost_usd is None (unpriced turn)",
            )

        passed = observation.cost_usd <= self.budget_usd
        return MetricResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            details=f"${observation.cost_usd:.6f} vs budget ${self.budget_usd:.6f}",
        )


class MemoryDriftMetric:
    """Evaluate whether an earlier-turn fact survives in a ``SessionInfoResponse`` snapshot."""

    name: str = "memory_drift"

    def __init__(self, fact: str, fact_field: FactField = "any") -> None:
        """
        Initialize the memory-drift substring matcher.

        :param fact: Target substring expected to survive in the session snapshot.
        :param fact_field: Snapshot slice to search (defaults to ``"any"``).
        """
        self.fact = fact
        self.fact_field = fact_field

    def evaluate(self, snapshot: SessionInfoResponse) -> MetricResult:
        """
        Check case-insensitively whether ``self.fact`` appears in the selected snapshot slice.

        :param snapshot: Post-turn session inspection payload.

        :return: Deterministic ``MetricResult`` for ``memory_drift``.
        """
        if self.fact_field == "project_name":
            target_text = snapshot.metadata.project_name or ""
        elif self.fact_field == "technologies":
            target_text = " ".join(snapshot.metadata.mentioned_technologies)
        elif self.fact_field == "scope":
            target_text = snapshot.metadata.agreed_scope or ""
        elif self.fact_field == "summary":
            target_text = snapshot.summary or ""
        else:
            target_text = snapshot.model_dump_json()

        found = self.fact.lower() in target_text.lower()
        return MetricResult(
            name=self.name,
            score=1.0 if found else 0.0,
            passed=found,
            details=f"fact={self.fact!r} field={self.fact_field} {'present' if found else 'missing'}",
        )
