# tests/evals/test_stress_run.py

"""Unit and integration tests for ``evals.stress.run``."""

# --- Python Standard Library ---
import asyncio
import csv
from pathlib import Path
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from evals.stress.run import _CSV_COLUMNS, parse_args, run_stress


@pytest.mark.parametrize(
    "argv",
    [
        ["--repeats", "0"],
        ["--latency-budget-ms", "0"],
        ["--cost-budget-usd", "-0.01"],
        ["--timeout", "0"],
        ["--dry-run", "--http", "http://127.0.0.1:8000"],
    ],
)
def test_parse_args_rejects_invalid_values(argv: list[str]) -> None:
    """
    GIVEN invalid numeric budgets or conflicting --dry-run/--http flags
    WHEN parse_args is invoked
    THEN argparse exits with a non-zero SystemExit.
    """
    with pytest.raises(SystemExit) as exc_info:
        parse_args(argv)

    assert exc_info.value.code != 0


def test_run_stress_dry_run_decoupled_writes_csv_and_tables(tmp_path: Path) -> None:
    """
    GIVEN default decoupled sweep arguments with --dry-run and --report-tables
    WHEN run_stress executes against the in-process ASGI app
    THEN 65 CSV rows with the exact schema and non-empty Markdown tables are written.
    """
    csv_path = tmp_path / "results.csv"
    tables_path = tmp_path / "tables.md"
    args = parse_args(["--dry-run", "--output", str(csv_path), "--report-tables", str(tables_path)])

    row_count = asyncio.run(run_stress(args))

    assert row_count == 65
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames or ()) == _CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 65
    assert all(row["error"] == "" for row in rows)

    tables_text = tables_path.read_text(encoding="utf-8")
    assert "## Summary Table" in tables_text
    assert "## Curve 1: Latency vs Tokens In" in tables_text
    assert "## Curve 2: Cumulative Cost vs Turn Index (Multi-Turn)" in tables_text
    assert "## Curve 3: MemoryDriftMetric Recall vs N" in tables_text


def test_run_stress_full_matrix_row_count(tmp_path: Path) -> None:
    """
    GIVEN --full-matrix with 1 scenario and 2 attachment sizes
    WHEN run_stress executes in --dry-run mode
    THEN 40 rows (1 x 2 x 20 turns) are written to CSV.
    """
    csv_path = tmp_path / "full.csv"
    args = parse_args(
        [
            "--dry-run",
            "--scenarios",
            "growing",
            "--attachment-sizes",
            "0,5",
            "--full-matrix",
            "--output",
            str(csv_path),
        ]
    )

    row_count = asyncio.run(run_stress(args))

    assert row_count == 40


def test_run_stress_mid_session_error_aborts_unit_and_records_row(tmp_path: Path) -> None:
    """
    GIVEN a session where turn 1 fails with a RuntimeError
    WHEN run_stress executes a 20-turn scenario
    THEN it records 1 error row and aborts remaining turns in that session.
    """
    csv_path = tmp_path / "error.csv"
    args = parse_args(
        [
            "--dry-run",
            "--scenarios",
            "growing",
            "--attachment-sizes",
            "0",
            "--full-matrix",
            "--output",
            str(csv_path),
        ]
    )

    with patch("evals.stress.run._execute_turn", new=AsyncMock(side_effect=RuntimeError("boom"))):
        row_count = asyncio.run(run_stress(args))

    assert row_count == 1
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert "RuntimeError: boom" in rows[0]["error"]
