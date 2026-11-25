# tests/ingestion/test_inspect_budgets.py

"""Unit tests for the read-only stdout CLI inspector of cleaned Session 6 budgets."""

# --- Python Standard Library ---
from collections.abc import Callable
import json
from pathlib import Path

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.inspect_budgets import inspect_cleaned_budgets, main
from aie_estimator.ingestion.orchestrator import IngestionRejected
from aie_estimator.ingestion.pii.analyzer import PiiAnalyzer
from aie_estimator.settings import Environment, Settings


@pytest.mark.anyio
async def test_inspect_cleaned_budgets_pseudonymizes_in_memory(
    make_stub_analyzer: Callable[[dict[str, str]], PiiAnalyzer],
) -> None:
    """
    GIVEN the seed corpus and a deterministic PiiAnalyzer stub
    WHEN inspect_cleaned_budgets runs with pseudonymize=True
    THEN it returns 2 validated CleanedBudget models with zero raw PII leaked.
    """
    analyzer = make_stub_analyzer(
        {
            "BUDGET-2024-0001": PiiFlag.BUDGET_ID.value,
            "BUDGET-2024-0002": PiiFlag.BUDGET_ID.value,
            "Acme Spain Ltd.": PiiFlag.ORGANIZATION.value,
            "BetaNorte Distribution": PiiFlag.ORGANIZATION.value,
            "CLI-0042": PiiFlag.CLIENT_CODE.value,
            "CLI-0078": PiiFlag.CLIENT_CODE.value,
            "Laura Miller": PiiFlag.PERSON.value,
            "laura.miller@acme.com": PiiFlag.EMAIL_ADDRESS.value,
            "James Reynolds": PiiFlag.PERSON.value,
            "j.reynolds@betanorte.com": PiiFlag.EMAIL_ADDRESS.value,
        }
    )
    settings = Settings(app_env=Environment.TEST)

    budgets = await inspect_cleaned_budgets(
        settings=settings,
        pseudonymize=True,
        analyzer=analyzer,
    )

    assert len(budgets) == 2
    serialized = json.dumps([budget.model_dump(mode="json") for budget in budgets])
    for raw_value in ("Acme Spain Ltd.", "BetaNorte Distribution", "Laura Miller", "laura.miller@acme.com", "CLI-0042"):
        assert raw_value not in serialized
    assert all(len(budget.phases) >= 3 for budget in budgets)


def test_main_prints_valid_json_with_no_pii(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    GIVEN the CLI inspector invoked with --no-pii
    WHEN main executes against the default seed corpus
    THEN stdout contains a valid JSON array of 2 cleaned budgets, stderr logs homogeneous events, and exit code is 0.
    """
    exit_code = main(["--no-pii"])

    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert len(parsed) == 2
    assert parsed[0]["budget_id"] == "BUDGET-2024-0001"
    assert parsed[0]["client_name"] == "Acme Spain Ltd."
    assert len(parsed[0]["phases"]) == 4
    assert "ingestion_inspection_started" in captured.err
    assert "ingestion_inspection_completed" in captured.err


def test_main_default_pseudonymized_mode_prints_safe_json(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    make_stub_analyzer: Callable[[dict[str, str]], PiiAnalyzer],
) -> None:
    """
    GIVEN the CLI inspector invoked with default pseudonymization enabled
    WHEN main executes with a stubbed PresidioPiiAnalyzer
    THEN stdout contains valid JSON with pseudonymized PII and exit code is 0.
    """
    stub = make_stub_analyzer(
        {
            "Acme Spain Ltd.": PiiFlag.ORGANIZATION.value,
            "Laura Miller": PiiFlag.PERSON.value,
        }
    )
    monkeypatch.setattr("aie_estimator.ingestion.inspect_budgets.PresidioPiiAnalyzer", lambda **_kwargs: stub)

    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert len(parsed) == 2
    assert "Acme Spain Ltd." not in captured.out
    assert "Laura Miller" not in captured.out
    assert "ingestion_inspection_completed" in captured.err


def test_main_returns_nonzero_and_logs_failure_on_error(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    GIVEN a missing catalog path or rejected source error
    WHEN main executes
    THEN stdout stays empty, stderr logs ingestion_inspection_failed, and exit code is 1.
    """
    bad_settings = Settings(app_env=Environment.TEST, catalog_path=tmp_path / "missing.yaml")
    monkeypatch.setattr("aie_estimator.ingestion.inspect_budgets.get_settings", lambda: bad_settings)

    exit_code = main(["--no-pii"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Error:" in captured.err
    assert "ingestion_inspection_failed" in captured.err


@pytest.mark.anyio
async def test_inspect_cleaned_budgets_rejects_non_included_source() -> None:
    """
    GIVEN a catalog source whose decision is not include
    WHEN inspect_cleaned_budgets is awaited
    THEN IngestionRejected is raised.
    """
    with pytest.raises(IngestionRejected, match=r"decision='review'"):
        await inspect_cleaned_budgets(
            settings=Settings(app_env=Environment.TEST),
            source_name="meeting_transcripts",
            pseudonymize=False,
        )
