# evals/run.py

# --- Python Standard Library ---
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Final

# --- Third Party Libraries ---
import httpx
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationResponse
from aie_estimator.server import create_app
from evals.dataset import GoldenCase, load_dataset
from evals.judge import evaluate_with_judge
from evals.metrics import MetricResult, evaluate_consistency_cv, run_family1_metrics

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_ESTIMATE_ENDPOINT: Final[str] = "/api/v1/estimate"
_DEFAULT_INPROCESS_URL: Final[str] = "http://inprocess.eval"
_HTTP_BAD_REQUEST: Final[int] = 400
_HTTP_UNPROCESSABLE: Final[int] = 422
_HTTP_OK: Final[int] = 200
_REQUEST_TIMEOUT_SECONDS: Final[float] = 120.0
_GUARDRAIL_MARKERS: Final[tuple[str, ...]] = ("guardrail", "injection", "moderation", "pii", "blocked")


def _build_request_payload(case: GoldenCase) -> dict[str, Any]:
    """
    Build the JSON request body for ``POST /api/v1/estimate`` from a ``GoldenCase``.

    :param case: Golden Dataset scenario.

    :return: Dictionary payload matching ``EstimationRequest``.
    """
    payload: dict[str, Any] = {
        "description": case.description,
        "project_type": case.project_type.value,
        "detail_level": case.detail_level.value,
        "output_format": case.output_format.value,
    }
    if case.tier is not None:
        payload["tier"] = case.tier.value
    return payload


async def _invoke_case_once(client: httpx.AsyncClient, case: GoldenCase) -> EstimationResponse | None:
    """
    Execute a single estimation request and parse ``EstimationResponse``.

    :param client: Configured async HTTP client.
    :param case: Golden Dataset scenario.

    :return: Parsed ``EstimationResponse``, or ``None`` if blocked by input guardrails as expected.
    :raises RuntimeError: If the endpoint returns an unexpected error status.
    """
    response = await client.post(_ESTIMATE_ENDPOINT, json=_build_request_payload(case))
    if response.status_code == _HTTP_OK:
        return EstimationResponse.model_validate(response.json())

    body_lower = response.text.lower()
    if (
        case.expected_out_of_scope
        and response.status_code in (_HTTP_BAD_REQUEST, _HTTP_UNPROCESSABLE)
        and any(marker in body_lower for marker in _GUARDRAIL_MARKERS)
    ):
        return None

    raise RuntimeError(f"HTTP {response.status_code} for case '{case.id}': {response.text[:300]}")


async def _evaluate_case(
    client: httpx.AsyncClient,
    case: GoldenCase,
    *,
    runs: int,
    use_judge: bool,
    judge_model: str | None,
) -> dict[str, Any]:
    """
    Run Families 1-3 for a single ``GoldenCase`` and return its structured report entry.

    :param client: Configured async HTTP client.
    :param case: Golden Dataset scenario.
    :param runs: Number of repeated invocations for Family 2 consistency.
    :param use_judge: Whether to invoke Family 3 LLM-as-Judge on the first response.
    :param judge_model: Optional model/profile override for the judge LLM.

    :return: Dictionary containing case metrics, pass status, latency, and cost.
    """
    started = time.perf_counter()
    responses: list[EstimationResponse] = []
    try:
        for _ in range(runs):
            result = await _invoke_case_once(client, case)
            if result is None:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                guardrail_metrics = [
                    MetricResult("schema_and_tier_adherence", 1.0, True, "blocked by input guardrail"),
                    MetricResult("numeric_bounds", 1.0, True, "blocked by input guardrail"),
                ]
                return {
                    "id": case.id,
                    "resolved_tier": (case.expected_tier or case.tier).value if (case.expected_tier or case.tier) else "guardrail",
                    "tier_rule": "input_guardrail",
                    "model": "guardrail",
                    "cached_runs": 0,
                    "cost_usd": 0.0,
                    "latency_ms": elapsed_ms,
                    "passed": True,
                    "metrics": [asdict(metric) for metric in guardrail_metrics],
                }
            responses.append(result)
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        log.error("eval_case_failed", case_id=case.id, error=str(exc), latency_ms=elapsed_ms)
        return {
            "id": case.id,
            "passed": False,
            "error": str(exc),
            "latency_ms": elapsed_ms,
            "metrics": [],
        }

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    primary = responses[0]
    metrics: list[MetricResult] = run_family1_metrics(case, primary)

    if runs > 1:
        metrics.append(evaluate_consistency_cv(responses))

    if use_judge:
        metrics.append(await evaluate_with_judge(case, primary, judge_model=judge_model))

    case_passed = all(metric.passed for metric in metrics)
    cached_runs = sum(1 for item in responses if item.cached)
    total_cost_usd = sum(item.cost_usd or 0.0 for item in responses)

    log.info(
        "eval_case_evaluated",
        case_id=case.id,
        tier=primary.tier.value,
        passed=case_passed,
        cached_runs=cached_runs,
        latency_ms=elapsed_ms,
    )
    return {
        "id": case.id,
        "resolved_tier": primary.tier.value,
        "tier_rule": primary.tier_rule,
        "model": primary.model,
        "cached_runs": cached_runs,
        "cost_usd": round(total_cost_usd, 6),
        "latency_ms": elapsed_ms,
        "passed": case_passed,
        "metrics": [asdict(metric) for metric in metrics],
    }


def _print_summary(report: dict[str, Any]) -> None:
    """
    Print a compact per-case summary table and overall pass tally to stdout.

    :param report: Completed evaluation report dictionary.
    """
    print(f"\n=== Evaluation Summary ({report['passed_cases']}/{report['total_cases']} cases passed) ===")
    for entry in report["cases"]:
        status = "PASS" if entry["passed"] else "FAIL"
        if "error" in entry:
            print(f"[{status}] {entry['id']}: ERROR {entry['error']}")
            continue
        metric_bits = ", ".join(f"{m['name']}={'OK' if m['passed'] else 'FAIL'}({m['score']})" for m in entry["metrics"])
        print(f"[{status}] {entry['id']} [tier={entry['resolved_tier']}] ({entry['latency_ms']}ms) -> {metric_bits}")


async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """
    Execute the evaluation suite across the Golden Dataset and optionally persist JSON output.

    :param args: Parsed CLI arguments.

    :return: Full evaluation report dictionary.
    """
    cases = load_dataset(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]

    log.info(
        "eval_run_started",
        case_count=len(cases),
        runs=args.runs,
        llm_judge=args.llm_judge,
        target=args.http or "inprocess",
    )

    if args.http:
        client = httpx.AsyncClient(base_url=args.http.rstrip("/"), timeout=_REQUEST_TIMEOUT_SECONDS)
    else:
        transport = httpx.ASGITransport(app=create_app())
        client = httpx.AsyncClient(transport=transport, base_url=_DEFAULT_INPROCESS_URL, timeout=_REQUEST_TIMEOUT_SECONDS)

    case_results: list[dict[str, Any]] = []
    async with client:
        for case in cases:
            case_results.append(
                await _evaluate_case(
                    client,
                    case,
                    runs=args.runs,
                    use_judge=args.llm_judge,
                    judge_model=args.judge_model,
                )
            )

    passed_count = sum(1 for item in case_results if item["passed"])
    report: dict[str, Any] = {
        "total_cases": len(case_results),
        "passed_cases": passed_count,
        "runs_per_case": args.runs,
        "llm_judge": args.llm_judge,
        "cases": case_results,
    }
    _print_summary(report)
    log.info("eval_run_completed", passed_cases=passed_count, total_cases=len(case_results))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("eval_report_saved", output=str(out_path))

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the Golden Dataset evaluation runner.

    :param argv: Optional argument list; defaults to ``sys.argv[1:]``.

    :return: Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run Family 1-3 LLM evaluations against the Golden Dataset.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Path to a custom Golden Dataset JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of dataset cases to run.")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per case for Family 2 CV consistency.")
    parser.add_argument("--llm-judge", action="store_true", help="Enable Family 3 structured LLM-as-Judge scoring.")
    parser.add_argument("--judge-model", type=str, default=None, help="Optional model/profile override for the judge.")
    parser.add_argument("--http", type=str, default=None, help="Base URL of a running server (defaults to in-process ASGI).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file path to save the evaluation report.")
    return parser.parse_args(argv)


def main() -> None:
    """CLI entrypoint for ``python -m evals.run``."""
    args = parse_args()
    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
