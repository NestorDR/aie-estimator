# LLM Evaluation Harness

## Overview

The LLM evaluation harness in [`evals/`](../evals/) measures estimation quality, statistical variance, and qualitative alignment across the Golden Dataset ([`evals/golden_dataset.json`](../evals/golden_dataset.json)) against live LLM providers. It implements the three evaluation families:

- **Family 1 (Hard Deterministic Invariants)** in [`evals/metrics.py`](../evals/metrics.py): `evaluate_schema_and_tier`, `evaluate_numeric_bounds`, and `evaluate_component_recall`.
- **Family 2 (Soft Statistical Consistency)** in [`evals/metrics.py`](../evals/metrics.py): `evaluate_consistency_cv` measuring Coefficient of Variation (`CV = stdev / mean`) and `Tier` stability across repeated runs.
- **Family 3 (Selective LLM-as-Judge)** in [`evals/judge.py`](../evals/judge.py) and [`evals/judge.j2`](../evals/judge.j2): `evaluate_with_judge` scoring `coherence`, `actionability`, and `tier_appropriateness` on a 1–5 Likert scale (normalized to `0.0`–`1.0`) with explanatory `reasoning`.

## Usage Examples

Run the evaluation CLI from the repository root via `uv` or [`auto/eval.cmd`](../auto/eval.cmd):

```powershell
# Family 1 deterministic evaluation (in-process ASGI app)
uv run python -m evals.run --limit 3

# Full Families 1-3 baseline run with JSON report output
uv run python -m evals.run --runs 3 --llm-judge --output evals/baseline.json

# Windows wrapper (equivalent flags)
auto\eval --runs 3 --llm-judge --output evals\baseline.json

# Target a running FastAPI server
uv run python -m evals.run --http http://127.0.0.1:8000 --llm-judge
```

## Technical Considerations

- **Offline Test Isolation**: `evals/` lives outside `testpaths = ["tests"]` in [`pyproject.toml`](../pyproject.toml), so `rtk uv run pytest --cov-fail-under=85` never executes live LLM calls. Offline unit tests in [`tests/evals/`](../tests/evals/) verify dataset validation, Family 1–2 math, and mocked Family 3 judge parsing.
- **Tier-Aware Dispatch**: Family 1 and Family 2 dispatch over `DeveloperEstimation`, `PmEstimation`, and `ExecutiveEstimation` using a scalar anchor (`total_hours_range` midpoint, `total_cost_eur`, and `headline_cost_range` midpoint) rather than brittle exact-string assertions.
- **Cache Interaction on Family 2**: When running `--runs N` (`N > 1`), disable exact/semantic Redis caching in the target environment so repeated identical requests measure true model variance rather than cache hits (`cached_runs` is reported per case).
- **Guardrail Mapping**: When `expected_out_of_scope` cases trigger Layer 1 input guardrails (HTTP 400/422), [`evals/run.py`](../evals/run.py) maps the rejection to a zero-cost, `confidence_pct=0` `Out of scope:` envelope so Family 1 scores it accurately.
