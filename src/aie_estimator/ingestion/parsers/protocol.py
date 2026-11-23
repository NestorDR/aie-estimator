# src/aie_estimator/ingestion/parsers/protocol.py

"""
Why a Protocol instead of an abstract base class? Two reasons:

* Parsers share no implementation — there is no useful method to inherit.
* Structural typing makes it trivial to drop in a fake during tests without
  inheriting from anything; any class with ``supported_formats`` and ``parse``
  satisfies the contract.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Protocol

# --- App modules ---
from aie_estimator.ingestion.catalog.models import CatalogSource
from aie_estimator.ingestion.documents.models import Document
from aie_estimator.ingestion.loaders.filesystem import LoadedBlob


@dataclass(frozen=True)
class ParseContext:
    """Run-scoped catalog and timestamp context supplied to format parsers."""

    source: CatalogSource
    source_version: str
    ingested_at: datetime


class Parser(Protocol):
    """Structural contract implemented by format-specific ingestion parsers."""

    supported_formats: ClassVar[set[str]]

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        """
        Parse raw source bytes into canonical Document instances.

        :param blob: Loaded source file bytes and relative path.
        :param context: Run-scoped catalog and timestamp metadata.

        :return: Iterable of parsed Document objects.
        """
        ...
