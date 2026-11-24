# tests/ingestion/test_orchestrator.py

# --- Python Standard Library ---
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.catalog.models import (
    CatalogDecision,
    CatalogSource,
    DataCatalog,
    PiiFlag,
    QualityScore,
    Sensitivity,
)
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata, compute_content_hash
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader, LoadedBlob
from aie_estimator.ingestion.orchestrator import IngestionRejected, ingest_source
from aie_estimator.ingestion.parsers.protocol import ParseContext
from aie_estimator.ingestion.parsers.registry import ParserRegistry, default_registry


class _SpoofingJsonParser:
    supported_formats: ClassVar[set[str]] = {"json"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        yield Document(
            id="spoof-1",
            text="canonical body",
            content_hash="0" * 64,
            metadata=DocumentMetadata(
                source_name="spoofed_source",
                source_version="9.9.9",
                source_location="wrong/path.json",
                ingested_at=datetime(2000, 1, 1, tzinfo=UTC),
                lineage=("Spoofed",),
                contains_pii=False,
                pii_flags=(),
                access_level="public",
                extra={"custom": 1},
            ),
        )


def test_ingest_source_processes_seed_budgets() -> None:
    """
    GIVEN the repository catalog and seed corpus
    WHEN ingest_source runs for historical_budgets
    THEN 2 valid seed budgets emit Documents, 3 route to quarantined, and 1 is discarded.
    """
    catalog = load_catalog(Path("data/catalog/catalog.yaml"))
    result = ingest_source(
        catalog=catalog,
        source_name="historical_budgets",
        loader=FileSystemLoader(Path("data/seed")),
        registry=default_registry(),
    )
    assert len(result.documents) == 2
    assert len(result.quarantined) == 3
    assert result.report["discarded_rows"] == 1
    assert all(doc.metadata.source_name == "historical_budgets" for doc in result.documents)
    assert all(doc.metadata.contains_pii is True for doc in result.documents)
    assert all(doc.content_hash == compute_content_hash(doc.text) for doc in result.documents)


@pytest.mark.parametrize(
    ("source_name", "expected_pattern"),
    [
        ("nonexistent_source", r"not found in catalog"),
        ("meeting_transcripts", r"decision='review'"),
        ("official_rate_card", r"decision='exclude'"),
    ],
)
def test_ingest_source_rejects_unknown_or_non_included_sources(source_name: str, expected_pattern: str) -> None:
    """
    GIVEN an unknown, review, or exclude catalog source
    WHEN ingest_source is invoked
    THEN IngestionRejected is raised.
    """
    catalog = load_catalog(Path("data/catalog/catalog.yaml"))
    with pytest.raises(IngestionRejected, match=expected_pattern):
        ingest_source(
            catalog=catalog,
            source_name=source_name,
            loader=FileSystemLoader(Path("data/seed")),
            registry=default_registry(),
        )


def test_normalizer_overrides_spoofed_parser_metadata(tmp_path: Path) -> None:
    """
    GIVEN a parser that attempts to spoof source_name, sensitivity, lineage, and content_hash
    WHEN ingest_source runs through normalizers/canonical.py
    THEN authoritative catalog metadata and computed SHA-256 content_hash overwrite parser values.
    """
    data_dir = tmp_path / "budgets"
    data_dir.mkdir()
    (data_dir / "item.json").write_bytes(b"{}")

    catalog = DataCatalog(
        version="2.1.0",
        sources=(
            CatalogSource(
                name="historical_budgets",
                location="budgets",
                format="json",
                quality=QualityScore(completeness=4, consistency=4, actuality=4, reliability=4),
                sensitivity=Sensitivity(
                    has_pii=True,
                    pii_flags=(PiiFlag.BUDGET_ID,),
                    access_level="confidential",
                ),
                lineage=("CRM Export",),
                decision=CatalogDecision.INCLUDE,
            ),
        ),
    )
    registry = ParserRegistry()
    registry.register(_SpoofingJsonParser())

    result = ingest_source(
        catalog=catalog,
        source_name="historical_budgets",
        loader=FileSystemLoader(tmp_path),
        registry=registry,
    )
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.content_hash == compute_content_hash("canonical body")
    assert doc.metadata.source_name == "historical_budgets"
    assert doc.metadata.source_version == "2.1.0"
    assert doc.metadata.source_location == "budgets/item.json"
    assert doc.metadata.lineage == ("CRM Export",)
    assert doc.metadata.contains_pii is True
    assert doc.metadata.pii_flags == (PiiFlag.BUDGET_ID,)
    assert doc.metadata.access_level == "confidential"
    assert doc.metadata.extra == {"custom": 1}
