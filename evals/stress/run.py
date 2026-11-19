# evals/stress/run.py

"""
Async stress runner for CAG multi-turn and attachment-sweep evaluation.

Executes scripted stress scenarios and attachment-size sweeps against either a live HTTP
estimator (``--http``) or an in-process FastAPI ``ASGITransport`` app (default / ``--dry-run``),
writes per-turn telemetry and deterministic metric verdicts to CSV, and optionally generates
Markdown summary and curve tables from the CSV.
"""

# --- Python Standard Library ---
import argparse
import asyncio
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
import sys
import time
from typing import Any, Final

# --- Third Party Libraries ---
import httpx
import structlog

# --- App modules ---
from aie_estimator.dependencies import get_estimation_service
from aie_estimator.schemas.estimations import (
    DetailLevel,
    OutputFormat,
    Phase,
    PmEstimation,
    ProjectType,
    TeamComposition,
)
from aie_estimator.schemas.sessions import SessionInfoResponse
from aie_estimator.server import create_app
from aie_estimator.sessions.models import Session, TurnObservation
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.utils.constants import DEFAULT_FAST_PROFILE
from evals.stress.metrics import (
    DEFAULT_COST_BUDGET_USD,
    DEFAULT_LATENCY_BUDGET_MS,
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import (
    ATTACHMENT_SWEEP_DETAIL_LEVEL,
    ATTACHMENT_SWEEP_OUTPUT_FORMAT,
    ATTACHMENT_SWEEP_PROJECT_TYPE,
    ATTACHMENT_SWEEP_TRANSCRIPT,
    CHECKPOINT_TURNS,
    Scenario,
    ScenarioTurn,
    attachment_canary_facts,
    get_scenario,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_SESSIONS_ENDPOINT: Final[str] = "/api/v1/sessions"
_DEFAULT_INPROCESS_URL: Final[str] = "http://inprocess.stress"
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 180.0
_DEFAULT_OUTPUT_PATH: Final[Path] = Path("evals/stress/results.csv")
_FIXTURES_DIR: Final[Path] = Path(__file__).parent / "fixtures"
_DEFAULT_SCENARIOS: Final[str] = "growing,pivot,contradiction"
_DEFAULT_ATTACHMENT_SIZES: Final[str] = "0,5,20,50,100"

_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "sweep",
    "scenario",
    "attachment_size_kb",
    "repeat",
    "tracked_fact",
    "error",
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "attachments_truncated_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "wall_clock_ms",
    "latency_budget_score",
    "latency_budget_passed",
    "cost_budget_score",
    "cost_budget_passed",
    "memory_drift_score",
    "memory_drift_passed",
)


@dataclass(frozen=True)
class _RunUnit:
    """Single session execution unit in the stress matrix."""

    sweep: str
    name: str
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    turns: tuple[ScenarioTurn, ...]
    size_kb: int
    repeat: int


class _DryRunEstimationService:
    """Deterministic in-process mock service for offline ``--dry-run`` verification."""

    async def generate_session_estimation(
        self,
        *,
        session: Session,
        transcript: str,
        project_type: ProjectType,
        detail_level: DetailLevel,
        output_format: OutputFormat,
        attachments: list[tuple[str, bytes]] | None = None,
        opts: Any = None,
        caller_tier: Tier | None = None,
        tier_override: Tier | None = None,
    ) -> dict[str, Any]:
        """
        Simulate a conversational estimation turn and populate ``session.last_turn_observed``.

        :param session: Active session container updated in-place.
        :param transcript: Current user turn transcript.
        :param project_type: Target project category.
        :param detail_level: Granularity level.
        :param output_format: Presentation format.
        :param attachments: Optional uploaded ``(filename, bytes)`` pairs.
        :param opts: Optional generation options carrying ``model``.
        :param caller_tier: Unused in dry-run.
        :param tier_override: Unused in dry-run.

        :return: Valid ``EstimationResponse`` payload dictionary.
        """
        del project_type, detail_level, output_format, caller_tier, tier_override
        attach_bytes = sum(len(content) for _, content in (attachments or ()))
        attach_chars = attach_bytes // 2
        session.turn_index += 1
        session.last_tier = "pm"
        session.last_tier_rule = "dry_run"
        session.history.append(user=transcript, assistant="dry-run estimation")
        if session.turn_index == 1:
            session.history.summary = transcript[:200]
            if "Nimbus" in transcript:
                session.metadata.project_name = "Nimbus"
            elif "Helios" in transcript:
                session.metadata.project_name = "Helios"
            elif "Vega" in transcript:
                session.metadata.project_name = "Vega"

        tokens_in = 400 + session.turn_index * 80 + attach_chars // 4
        tokens_out = 180
        cost_usd = round(0.0005 + tokens_in * 0.000001, 6)
        latency_ms = 250 + session.turn_index * 20 + attach_bytes // 512
        model_name = opts.model if opts and getattr(opts, "model", None) else DEFAULT_FAST_PROFILE

        session.last_turn_observed = TurnObservation(
            turn_index=session.turn_index,
            session_id=session.session_id,
            enriched_transcript_chars=len(transcript) + attach_chars,
            attachments_total_chars=attach_chars,
            attachments_truncated_chars=0,
            messages_in_window=min(len(session.history.messages), 6),
            anchors_count=len(session.history.anchors),
            summary_chars=len(session.history.summary or ""),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cache_hit_kind="none",
            last_resolved_tier="pm",
        )
        pm_result = PmEstimation(
            summary="Dry-run PM estimation.",
            confidence_pct=85,
            phases=[
                Phase(
                    name="Core Delivery",
                    duration_weeks=4,
                    cost_eur=12_000,
                    summary="Simulated delivery phase.",
                )
            ],
            team_composition=TeamComposition(developers=2, designers=1),
            total_duration_weeks=4,
            total_cost_eur=12_000,
        )
        return {
            "result": pm_result,
            "tier": Tier.PM,
            "tier_rule": "dry_run",
            "prompt_version": "dry_run",
            "cached": False,
            "model": model_name,
            "provider": "dry_run",
            "cost_usd": cost_usd,
            "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
        }


def _attachment_files(size_kb: int) -> list[tuple[str, tuple[str, bytes, str]]] | None:
    """
    Load the calibrated PDF fixture for ``size_kb``, or return ``None`` for ``0`` KB.

    :param size_kb: Attachment size in kilobytes.

    :return: Multipart ``files`` payload list for ``httpx``, or ``None``.
    :raises FileNotFoundError: If the expected PDF fixture is missing.
    """
    if size_kb == 0:
        return None
    path = _FIXTURES_DIR / f"attach_{size_kb}kb.pdf"
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture {path} missing — run `uv run python -m evals.stress.fixtures.build_pdfs` first."
        )
    return [("attachments", (path.name, path.read_bytes(), "application/pdf"))]


def _build_run_units(
    scenarios: list[Scenario],
    sizes: list[int],
    repeats: int,
    *,
    full_matrix: bool,
) -> list[_RunUnit]:
    """
    Build the ordered list of session execution units.

    :param scenarios: Selected 20-turn scenarios.
    :param sizes: Selected attachment sizes in KB.
    :param repeats: Number of repetitions per configuration.
    :param full_matrix: If ``True``, run full Cartesian ``scenarios x sizes x repeats``.

    :return: List of ``_RunUnit`` specifications.
    """
    units: list[_RunUnit] = []
    if full_matrix:
        for scenario in scenarios:
            for size_kb in sizes:
                for repeat in range(repeats):
                    units.append(
                        _RunUnit(
                            sweep="full_matrix",
                            name=scenario.name,
                            project_type=scenario.project_type,
                            detail_level=scenario.detail_level,
                            output_format=scenario.output_format,
                            turns=scenario.turns,
                            size_kb=size_kb,
                            repeat=repeat,
                        )
                    )
        return units

    for scenario in scenarios:
        for repeat in range(repeats):
            units.append(
                _RunUnit(
                    sweep="multi_turn",
                    name=scenario.name,
                    project_type=scenario.project_type,
                    detail_level=scenario.detail_level,
                    output_format=scenario.output_format,
                    turns=scenario.turns,
                    size_kb=0,
                    repeat=repeat,
                )
            )

    attach_turn = (ScenarioTurn(transcript=ATTACHMENT_SWEEP_TRANSCRIPT),)
    for size_kb in sizes:
        for repeat in range(repeats):
            units.append(
                _RunUnit(
                    sweep="attachment",
                    name=f"attachment_{size_kb}kb",
                    project_type=ATTACHMENT_SWEEP_PROJECT_TYPE,
                    detail_level=ATTACHMENT_SWEEP_DETAIL_LEVEL,
                    output_format=ATTACHMENT_SWEEP_OUTPUT_FORMAT,
                    turns=attach_turn,
                    size_kb=size_kb,
                    repeat=repeat,
                )
            )
    return units


def _open_client(
    *,
    http_base_url: str | None,
    timeout_seconds: float,
    dry_run: bool,
) -> httpx.AsyncClient:
    """
    Create an ``httpx.AsyncClient`` for either live HTTP or in-process ASGI execution.

    :param http_base_url: Optional live server base URL.
    :param timeout_seconds: Request timeout in seconds.
    :param dry_run: When ``True``, overrides ``get_estimation_service`` with a deterministic mock.

    :return: Configured ``httpx.AsyncClient`` instance.
    """
    if http_base_url:
        return httpx.AsyncClient(base_url=http_base_url.rstrip("/"), timeout=timeout_seconds)

    app = create_app()
    if dry_run:
        mock_service = _DryRunEstimationService()
        app.dependency_overrides[get_estimation_service] = lambda: mock_service
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url=_DEFAULT_INPROCESS_URL, timeout=timeout_seconds)


async def _execute_turn(
    client: httpx.AsyncClient,
    session_id: str,
    unit: _RunUnit,
    turn: ScenarioTurn,
    files: list[tuple[str, tuple[str, bytes, str]]] | None,
    model: str,
) -> tuple[TurnObservation, SessionInfoResponse, int]:
    """
    Execute one estimation turn and fetch the post-turn ``SessionInfoResponse`` snapshot.

    :param client: Active async HTTP/ASGI client.
    :param session_id: Target session UUID string.
    :param unit: Current ``_RunUnit`` configuration.
    :param turn: Current ``ScenarioTurn`` payload.
    :param files: Optional multipart attachment files.
    :param model: Pinned model or profile name.

    :return: Tuple of ``(TurnObservation, SessionInfoResponse, wall_clock_ms)``.
    :raises RuntimeError: If ``last_turn_observed`` is missing from the session snapshot.
    """
    form_data = {
        "transcript": turn.transcript,
        "project_type": unit.project_type.value,
        "detail_level": unit.detail_level.value,
        "output_format": unit.output_format.value,
        "model": model,
    }
    t0 = time.perf_counter()
    response = await client.post(f"{_SESSIONS_ENDPOINT}/{session_id}/estimate", data=form_data, files=files)
    wall_ms = int((time.perf_counter() - t0) * 1000)
    response.raise_for_status()

    snap_resp = await client.get(f"{_SESSIONS_ENDPOINT}/{session_id}")
    snap_resp.raise_for_status()
    snapshot = SessionInfoResponse.model_validate(snap_resp.json())
    if snapshot.last_turn_observed is None:
        raise RuntimeError(f"Session {session_id} snapshot is missing last_turn_observed.")
    return snapshot.last_turn_observed, snapshot, wall_ms


def _evaluate_drift(
    *,
    unit: _RunUnit,
    turn_index: int,
    tracked_fact: str,
    drift_metric: MemoryDriftMetric | None,
    snapshot: SessionInfoResponse,
) -> tuple[float | str, bool | str]:
    """
    Evaluate ``MemoryDriftMetric`` for multi-turn or attachment-sweep turns.

    :param unit: Current ``_RunUnit`` metadata.
    :param turn_index: 1-based turn index from ``TurnObservation``.
    :param tracked_fact: Tracked turn-1 anchor fact (empty for attachment sweep).
    :param drift_metric: Pre-built ``MemoryDriftMetric`` for multi-turn runs, or ``None``.
    :param snapshot: Post-turn ``SessionInfoResponse``.

    :return: Tuple of ``(score, passed)``, or ``("", "")`` when skipped (turn 1 / 0 KB attachment).
    """
    if unit.sweep == "attachment":
        canaries = attachment_canary_facts(unit.size_kb)
        if not canaries:
            return "", ""
        results = [MemoryDriftMetric(fact=fact, fact_field="any").evaluate(snapshot) for fact in canaries]
        avg_score = round(sum(r.score for r in results) / len(results), 3)
        return avg_score, all(r.passed for r in results)

    if drift_metric is None or turn_index == 1 or not tracked_fact:
        return "", ""
    res = drift_metric.evaluate(snapshot)
    return res.score, res.passed


async def _run_one_unit(
    client: httpx.AsyncClient,
    unit: _RunUnit,
    model: str,
    latency_metric: LatencyBudgetMetric,
    cost_metric: CostBudgetMetric,
    writer: csv.DictWriter[str],
) -> int:
    """
    Execute all turns of a single ``_RunUnit`` and stream CSV rows.

    :param client: Active async HTTP/ASGI client.
    :param unit: Session execution unit.
    :param model: Pinned model/profile identifier.
    :param latency_metric: Configured ``LatencyBudgetMetric``.
    :param cost_metric: Configured ``CostBudgetMetric``.
    :param writer: Open CSV ``DictWriter``.

    :return: Number of CSV rows written.
    """
    files = _attachment_files(unit.size_kb)
    create_resp = await client.post(_SESSIONS_ENDPOINT)
    create_resp.raise_for_status()
    session_id = str(create_resp.json()["session_id"])

    tracked_fact = unit.turns[0].fact_introduced or ""
    drift_metric = (
        MemoryDriftMetric(fact=tracked_fact, fact_field=unit.turns[0].fact_field)
        if tracked_fact
        else None
    )

    rows_written = 0
    for turn in unit.turns:
        row: dict[str, Any] = {
            "sweep": unit.sweep,
            "scenario": unit.name,
            "attachment_size_kb": unit.size_kb,
            "repeat": unit.repeat,
            "tracked_fact": tracked_fact or ("|".join(attachment_canary_facts(unit.size_kb)) if unit.sweep == "attachment" else ""),
            "error": "",
        }
        try:
            obs, snapshot, wall_ms = await _execute_turn(client, session_id, unit, turn, files, model)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            writer.writerow(row)
            return rows_written + 1

        lat_res = latency_metric.evaluate(obs)
        cost_res = cost_metric.evaluate(obs)
        drift_score, drift_passed = _evaluate_drift(
            unit=unit,
            turn_index=obs.turn_index,
            tracked_fact=tracked_fact,
            drift_metric=drift_metric,
            snapshot=snapshot,
        )
        row.update(
            {
                **obs.model_dump(),
                "wall_clock_ms": wall_ms,
                "latency_budget_score": lat_res.score,
                "latency_budget_passed": lat_res.passed,
                "cost_budget_score": cost_res.score,
                "cost_budget_passed": cost_res.passed,
                "memory_drift_score": drift_score,
                "memory_drift_passed": drift_passed,
            }
        )
        writer.writerow(row)
        rows_written += 1

    return rows_written


def _percentile(sorted_values: list[int], pct: float) -> int:
    """Compute nearest-rank integer percentile from a pre-sorted list."""
    if not sorted_values:
        return 0
    if pct <= 50.0:
        return int(median(sorted_values))
    idx = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * (pct / 100.0)) - 1))
    return sorted_values[idx]


def build_report_tables(csv_path: Path) -> str:
    """
    Compute the P50/P95 summary table and three Markdown curve tables from ``results.csv``.

    :param csv_path: Path to the generated stress-run CSV file.

    :return: Formatted Markdown string containing the summary and three curve tables.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("error")]

    lines: list[str] = ["## Summary Table", ""]
    lines.append("| Scenario / Sweep | KB | Rows | P50 ms | P95 ms | Total Cost ($) | Cache Hit % | Mean Drift Recall |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    by_group: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_group.setdefault((row["scenario"], row["attachment_size_kb"]), []).append(row)

    for (scenario, kb), group in sorted(by_group.items()):
        lats = sorted(int(r["latency_ms"]) for r in group)
        cost = sum(float(r["cost_usd"]) for r in group if r["cost_usd"])
        hits = sum(1 for r in group if r["cache_hit_kind"] != "none")
        drifts = [float(r["memory_drift_score"]) for r in group if r["memory_drift_score"] != ""]
        drift_str = f"{100.0 * sum(drifts) / len(drifts):.1f}%" if drifts else "n/a"
        lines.append(
            f"| {scenario} | {kb} | {len(group)} | {_percentile(lats, 50)} | {_percentile(lats, 95)} | "
            f"{cost:.4f} | {100.0 * hits / len(group):.1f}% | {drift_str} |"
        )

    lines.extend(["", "## Curve 1: Latency vs Tokens In", ""])
    lines.append("| Bucket (tokens_in) | Rows | Mean tokens_in | P50 ms | P95 ms |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    buckets: dict[str, list[dict[str, str]]] = {"<1000": [], "1000-2999": [], "3000-9999": [], ">=10000": []}
    for row in rows:
        tok = int(row["tokens_in"])
        key = "<1000" if tok < 1000 else "1000-2999" if tok < 3000 else "3000-9999" if tok < 10000 else ">=10000"
        buckets[key].append(row)
    for label, group in buckets.items():
        if not group:
            continue
        lats = sorted(int(r["latency_ms"]) for r in group)
        mean_tok = int(sum(int(r["tokens_in"]) for r in group) / len(group))
        lines.append(f"| {label} | {len(group)} | {mean_tok} | {_percentile(lats, 50)} | {_percentile(lats, 95)} |")

    lines.extend(["", "## Curve 2: Cumulative Cost vs Turn Index (Multi-Turn)", ""])
    lines.append("| Turn | growing ($) | pivot ($) | contradiction ($) |")
    lines.append("| ---: | ---: | ---: | ---: |")
    mt_rows = [r for r in rows if r["sweep"] != "attachment"]
    cum_cost: dict[str, dict[int, list[float]]] = {}
    for row in mt_rows:
        sc = row["scenario"]
        turn_idx = int(row["turn_index"])
        cum_cost.setdefault(sc, {}).setdefault(turn_idx, []).append(float(row["cost_usd"] or 0.0))

    running: dict[str, float] = {"growing": 0.0, "pivot": 0.0, "contradiction": 0.0}
    max_turn = max((int(r["turn_index"]) for r in mt_rows), default=0)
    for t in range(1, max_turn + 1):
        for sc in running:
            vals = cum_cost.get(sc, {}).get(t, [])
            if vals:
                running[sc] += sum(vals) / len(vals)
        lines.append(f"| {t} | {running['growing']:.4f} | {running['pivot']:.4f} | {running['contradiction']:.4f} |")

    lines.extend(["", "## Curve 3: MemoryDriftMetric Recall vs N", ""])
    lines.append("| Checkpoint Turn (N) | growing | pivot | contradiction |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for cp in CHECKPOINT_TURNS:
        cells: list[str] = []
        for sc in ("growing", "pivot", "contradiction"):
            vals = [
                float(r["memory_drift_score"])
                for r in mt_rows
                if r["scenario"] == sc and int(r["turn_index"]) == cp and r["memory_drift_score"] != ""
            ]
            cells.append(f"{100.0 * sum(vals) / len(vals):.1f}%" if vals else "n/a")
        lines.append(f"| {cp} | {cells[0]} | {cells[1]} | {cells[2]} |")

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate CLI arguments for ``python -m evals.stress.run``.

    :param argv: Optional argument list; defaults to ``sys.argv[1:]``.

    :return: Validated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m evals.stress.run",
        description="Run CAG stress scenarios and attachment sweeps against the estimator API.",
    )
    parser.add_argument("--http", type=str, default=None, help="Base URL of a running estimator (defaults to in-process ASGI).")
    parser.add_argument("--scenarios", type=str, default=_DEFAULT_SCENARIOS, help="Comma-separated scenario names.")
    parser.add_argument("--attachment-sizes", type=str, default=_DEFAULT_ATTACHMENT_SIZES, help="Comma-separated PDF sizes in KB.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repetitions per sweep unit (default: 1).")
    parser.add_argument("--full-matrix", action="store_true", help="Run full Cartesian scenarios x attachment_sizes matrix.")
    parser.add_argument("--model", type=str, default=DEFAULT_FAST_PROFILE, help="Model or profile alias to pin across turns.")
    parser.add_argument("--latency-budget-ms", type=int, default=DEFAULT_LATENCY_BUDGET_MS, help="Per-turn latency budget in ms.")
    parser.add_argument("--cost-budget-usd", type=float, default=DEFAULT_COST_BUDGET_USD, help="Per-turn cost budget in USD.")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_SECONDS, help="HTTP request timeout in seconds.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_PATH, help="Output CSV file path.")
    parser.add_argument("--report-tables", type=Path, default=None, help="Optional Markdown file path for generated summary and curve tables.")
    parser.add_argument("--dry-run", action="store_true", help="Run in-process with deterministic mocked EstimationService.")

    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.latency_budget_ms <= 0:
        parser.error("--latency-budget-ms must be positive.")
    if args.cost_budget_usd <= 0:
        parser.error("--cost-budget-usd must be positive.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    if args.dry_run and args.http:
        parser.error("--dry-run cannot be combined with --http.")
    return args


async def run_stress(args: argparse.Namespace) -> int:
    """
    Execute the configured stress sweep and write CSV (and optional Markdown tables).

    :param args: Validated CLI arguments.

    :return: Total number of CSV rows written.
    """
    scenarios = [get_scenario(name.strip()) for name in args.scenarios.split(",") if name.strip()]
    sizes = [int(s.strip()) for s in args.attachment_sizes.split(",") if s.strip()]
    for size_kb in sizes:
        if size_kb != 0:
            _attachment_files(size_kb)

    units = _build_run_units(scenarios, sizes, args.repeats, full_matrix=args.full_matrix)
    latency_metric = LatencyBudgetMetric(budget_ms=args.latency_budget_ms)
    cost_metric = CostBudgetMetric(budget_usd=args.cost_budget_usd)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log.info("stress_run_started", units=len(units), model=args.model, dry_run=args.dry_run, target=args.http or "inprocess")

    total_rows = 0
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        async with _open_client(http_base_url=args.http, timeout_seconds=args.timeout, dry_run=args.dry_run) as client:
            for unit in units:
                rows_written = await _run_one_unit(client, unit, args.model, latency_metric, cost_metric, writer)
                total_rows += rows_written
                fh.flush()

    tables_md = build_report_tables(args.output)
    if args.report_tables is not None:
        args.report_tables.parent.mkdir(parents=True, exist_ok=True)
        args.report_tables.write_text(tables_md, encoding="utf-8")

    log.info("stress_run_completed", rows=total_rows, output=str(args.output))
    return total_rows


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m evals.stress.run``."""
    args = parse_args(argv)
    asyncio.run(run_stress(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
