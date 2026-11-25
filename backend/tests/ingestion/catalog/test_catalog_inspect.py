# tests/ingestion/catalog/test_catalog_inspect.py

# --- Python Standard Library ---
from pathlib import Path

# --- App modules ---
from aie_estimator.ingestion.catalog.inspect import FolderFacts, inspect_folder, inspect_root


def test_inspect_folder_missing_returns_empty(tmp_path: Path) -> None:
    """
    GIVEN a non-existent folder path
    WHEN inspect_folder is called
    THEN it returns zeroed FolderFacts without raising.
    """
    missing = tmp_path / "does_not_exist"
    facts = inspect_folder(missing)
    assert facts == FolderFacts(folder=missing)


def test_inspect_root_missing_returns_empty_list(tmp_path: Path) -> None:
    """
    GIVEN a non-existent root path
    WHEN inspect_root is called
    THEN it returns an empty list.
    """
    assert inspect_root(tmp_path / "missing_root") == []


def test_inspect_root_collects_subdirs_and_direct_files(tmp_path: Path) -> None:
    """
    GIVEN a corpus root with child directories and top-level files
    WHEN inspect_root is called
    THEN it returns factual metrics per child directory plus one summary for direct root files.
    """
    budgets = tmp_path / "budgets"
    budgets.mkdir()
    (budgets / "b1.json").write_text('{"id": 1}', encoding="utf-8")
    (budgets / "b2.JSON").write_text('{"id": 2}', encoding="utf-8")

    (tmp_path / "rate_card.xlsx").write_bytes(b"xlsx-bytes")

    results = inspect_root(tmp_path)
    assert len(results) == 2

    budgets_facts = results[0]
    assert budgets_facts.folder == budgets
    assert budgets_facts.file_count == 2
    assert budgets_facts.formats_detected == frozenset({"json"})
    assert budgets_facts.latest_modified is not None
    assert budgets_facts.total_size_mb >= 0.0

    root_facts = results[1]
    assert root_facts.folder == tmp_path
    assert root_facts.file_count == 1
    assert root_facts.formats_detected == frozenset({"xlsx"})
