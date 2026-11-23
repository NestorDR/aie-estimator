# tests/ingestion/test_loaders.py

# --- Python Standard Library ---
from pathlib import Path

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader


def test_filesystem_loader_reads_matching_directory_files(tmp_path: Path) -> None:
    """
    GIVEN a directory containing matching and non-matching files
    WHEN iter_blobs runs for a target extension
    THEN only matching files are yielded in sorted relative-path order.
    """
    budgets_dir = tmp_path / "budgets"
    budgets_dir.mkdir()
    (budgets_dir / "b.json").write_bytes(b'{"id": 2}')
    (budgets_dir / "a.JSON").write_bytes(b'{"id": 1}')
    (budgets_dir / "ignore.txt").write_bytes(b"ignored")

    blobs = list(FileSystemLoader(tmp_path).iter_blobs("budgets", {"json"}))
    assert [blob.relative_path for blob in blobs] == ["budgets/a.JSON", "budgets/b.json"]
    assert [blob.bytes_ for blob in blobs] == [b'{"id": 1}', b'{"id": 2}']


def test_filesystem_loader_reads_single_file(tmp_path: Path) -> None:
    """
    GIVEN a catalog location pointing directly to a single file
    WHEN iter_blobs runs with its format
    THEN that single LoadedBlob is yielded.
    """
    target = tmp_path / "single.txt"
    target.write_bytes(b"hello")

    blobs = list(FileSystemLoader(tmp_path).iter_blobs("single.txt", {"txt"}))
    assert len(blobs) == 1
    assert blobs[0].relative_path == "single.txt"
    assert blobs[0].bytes_ == b"hello"


def test_filesystem_loader_raises_for_missing_location(tmp_path: Path) -> None:
    """
    GIVEN a catalog location that does not exist under data_root
    WHEN iter_blobs is consumed
    THEN FileNotFoundError is raised.
    """
    with pytest.raises(FileNotFoundError, match=r"does not exist"):
        list(FileSystemLoader(tmp_path).iter_blobs("missing", {"json"}))
