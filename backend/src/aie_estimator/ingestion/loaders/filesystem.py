# src/aie_estimator/ingestion/loaders/filesystem.py

"""
Filesystem loader.

Resolves the ``CatalogSource.location`` against ``settings.ingestion_data_root``,
walks the resulting path, and yields one ``LoadedBlob`` per file whose
extension matches the source's declared format.

A non-existent location is a hard error: a catalog that points nowhere is
silently wrong, the worst kind of broken.
"""

# --- Python Standard Library ---
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedBlob:
    """Single source file payload with its path relative to the configured data root."""

    relative_path: str
    bytes_: bytes


class FileSystemLoader:
    """Loads raw file bytes for a catalog source from a local filesystem root."""

    def __init__(self, data_root: Path | str) -> None:
        """
        Initialize the loader rooted at ``data_root``.

        :param data_root: Base directory against which catalog source locations are resolved.
        """
        self._data_root = Path(data_root)

    def iter_blobs(self, location: str, formats: set[str]) -> Iterator[LoadedBlob]:
        """
        Yield matching file blobs under ``location`` sorted deterministically by path.

        :param location: Relative file or directory path from the catalog entry.
        :param formats: Lowercase file extensions (without leading dots) to include.

        :return: Iterator of LoadedBlob instances.
        :raises FileNotFoundError: If the resolved source location does not exist.
        """
        path = self._data_root / location
        if not path.exists():
            raise FileNotFoundError(f"Catalog location '{location}' does not exist under '{self._data_root}'.")

        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())

        for candidate in candidates:
            ext = candidate.suffix.lower().lstrip(".")
            if ext not in formats:
                continue
            yield LoadedBlob(
                relative_path=candidate.relative_to(self._data_root).as_posix(),
                bytes_=candidate.read_bytes(),
            )
