# CAG Stress Testing

## Overview

The `../backend/evals` suite measures multi-turn Context-Augmented Generation (CAG) degradation past `DEFAULT_SESSION_MAX_TURNS` (`6` in [`constants.py`](../backend/src/aie_estimator/utils/constants.py)) across latency, cost, and memory recall:

- **Scenarios** ([`../backend/evals`](../backend/evals/stress/scenarios.py)): Three 20-turn profiles—`growing` (`Nimbus`), `pivot` (`Helios`, replacing `React Native` with `Flutter` at turn 5), and `contradiction` (`Atlas`, revising budget from `30000` to `80000` EUR at turn 8)—plus single-turn attachment size test defaults (`ATTACHMENT_SWEEP_TRANSCRIPT`, `attachment_canary_facts(size_kb)` delegating to [`canary_facts`](../backend/evals/stress/fixtures/build_pdfs.py) for `0`, `5`, `20`, `50`, and `100` KB PDFs).
- **Metrics** ([`../backend/evals`](../backend/evals/stress/metrics.py)): Three deterministic evaluators reusing `MetricResult` from [`../backend/evals`](../backend/evals/metrics.py):
  - `LatencyBudgetMetric`: checks `TurnObservation.latency_ms <= budget_ms` (default `DEFAULT_LATENCY_BUDGET_MS = 8000`).
  - `CostBudgetMetric`: checks `TurnObservation.cost_usd <= budget_usd` (default `DEFAULT_COST_BUDGET_USD = 0.05`), failing explicitly when `cost_usd is None`.
  - `MemoryDriftMetric`: performs case-insensitive substring matching against [`SessionInfoResponse`](../backend/src/aie_estimator/schemas/sessions.py) scoped by `FactField` (`"project_name"`, `"technologies"`, `"scope"`, `"summary"`, or `"any"`).
- **Runner** ([`../backend/evals`](../backend/evals/stress/run.py)): Orchestrates sessions against `/api/v1/sessions`, writes per-turn telemetry and metric verdicts to `../backend/evals`, and generates P50/P95 summary and Markdown curve tables via [`build_report_tables`](../backend/evals/stress/run.py) when `--report-tables` is passed.

## Usage Examples

Programmatic scenario and metric evaluation:

```python
from backend.evals.stress.metrics import CostBudgetMetric, LatencyBudgetMetric, MemoryDriftMetric
from backend.evals import all_scenarios, attachment_canary_facts, get_scenario

growing = get_scenario("growing")
turn_1 = growing.turns[0]

latency_ok = LatencyBudgetMetric(budget_ms=8000).evaluate(observation)
cost_ok = CostBudgetMetric(budget_usd=0.05).evaluate(observation)
drift_ok = MemoryDriftMetric(fact=turn_1.fact_introduced or "", fact_field=turn_1.fact_field).evaluate(snapshot)
canaries_50kb = attachment_canary_facts(50)
```

Offline smoke test using `--dry-run` (strictly for smoke testing and CI verification):

```bash
uv run python -m evals.stress.run --dry-run --output evals/stress/dry_run.csv --report-tables evals/stress/dry_tables.md
```

Live baseline run against a local server (or via [`../backend/auto/stress.cmd`](../backend/auto/stress.cmd) on Windows):

```bash
uv run python -m evals.stress.run --http http://127.0.0.1:8000 --repeats 1 --output evals/stress/results.csv --report-tables evals/stress/tables.md
```

## Technical Considerations

- **Single 20-Turn Script + Checkpoints**: Each profile runs as one causal 20-turn session sampled at `CHECKPOINT_TURNS = (1, 3, 6, 10, 20)` rather than separate truncated sessions for `N ∈ {1, 3, 6, 10}`, halving LLM calls while preserving identical pre- and post-compression states.
- **Decoupled Two-Pass Default**: By default, `run_stress` executes the three 20-turn scenarios at `0 KB` (60 rows) plus single-turn attachment size tests across `0,5,20,50,100 KB` (5 rows), isolating history length from attachment truncation and yielding 65 rows per repeat without a 300-turn Cartesian run unless `--full-matrix` is set.
- **Separation from Golden Dataset Metrics**: Stress metrics live apart from [`../backend/evals`](../backend/evals/metrics.py) because they evaluate [`TurnObservation`](../backend/src/aie_estimator/sessions/models.py) and [`SessionInfoResponse`](../backend/src/aie_estimator/schemas/sessions.py) rather than `(GoldenCase, EstimationResponse)`.
- **Option A Fact-Tracker & `"any"` vs `"technologies"`**: Each `ScenarioTurn` declares at most one `fact_introduced` substring and `fact_field: FactField`. Because `"any"` searches `SessionInfoResponse.model_dump_json()` (including `metadata.rejected_options`), stack-pivot checks target `fact_field="technologies"` (`metadata.mentioned_technologies`).
- **Unpriced Turn Failure & Constructor Purity**: `CostBudgetMetric.evaluate()` returns `score=0.0`, `passed=False` (`"cost_usd is None (unpriced turn)"`) when `cost_usd` is `None`. Metric `__init__` methods do not validate positive budgets; CLI/runner callers validate budget arguments upfront.
- **Session Snapshot Telemetry & Error Isolation**: After each `POST /api/v1/sessions/{session_id}/estimate`, the runner fetches `GET /api/v1/sessions/{session_id}` to read `last_turn_observed` and evaluate `MemoryDriftMetric` without log scraping. A mid-session HTTP or runtime exception records the error string in the CSV row and aborts only the affected session.
