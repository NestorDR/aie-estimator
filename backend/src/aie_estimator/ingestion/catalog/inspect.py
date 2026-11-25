# src/aie_estimator/ingestion/catalog/inspect.py

# --- Python Standard Library ---
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_BYTES_PER_MIB: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class FolderFacts:
    """Factual filesystem metrics collected for a directory or file group."""

    folder: Path
    file_count: int = 0
    total_size_mb: float = 0.0
    latest_modified: datetime | None = None
    formats_detected: frozenset[str] = frozenset()


def _summarize_files(folder: Path, files: Iterable[Path]) -> FolderFacts:
    """
    Aggregate file count, size in MiB, latest mtime, and extensions for files.

    :param folder: Directory path associated with the file collection.
    :param files: Iterable of file paths to measure.

    :return: Populated FolderFacts summary.
    """
    file_count = 0
    total_bytes = 0
    latest_modified: datetime | None = None
    formats: set[str] = set()

    for path in files:
        stat = path.stat()
        file_count += 1
        total_bytes += stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if latest_modified is None or modified > latest_modified:
            latest_modified = modified
        if path.suffix:
            formats.add(path.suffix.lower().lstrip("."))

    facts = FolderFacts(
        folder=folder,
        file_count=file_count,
        total_size_mb=round(total_bytes / _BYTES_PER_MIB, 4),
        latest_modified=latest_modified,
        formats_detected=frozenset(formats),
    )
    log.info(
        "catalog_source_inspected",
        folder=str(folder),
        file_count=facts.file_count,
        total_size_mb=facts.total_size_mb,
    )
    return facts


def inspect_folder(folder: Path) -> FolderFacts:
    """
    Collect factual filesystem metrics recursively for a single directory.

    :param folder: Directory path to inspect.

    :return: FolderFacts for all files under folder, or empty facts if missing.
    """
    if not folder.exists():
        return FolderFacts(folder=folder)
    return _summarize_files(folder, (path for path in folder.rglob("*") if path.is_file()))


def inspect_root(root: Path) -> list[FolderFacts]:
    """
    Inspect each immediate child directory of root plus any direct files under root.

    :param root: Root corpus directory to inspect.

    :return: List of FolderFacts entries, or empty list if root does not exist.
    """
    if not root.exists():
        return []

    results = [inspect_folder(child) for child in sorted(p for p in root.iterdir() if p.is_dir())]
    direct_files = [path for path in root.iterdir() if path.is_file()]
    if direct_files:
        results.append(_summarize_files(root, direct_files))
    return results
