# tests/ingestion/test_parsers.py

# --- Python Standard Library ---
from datetime import UTC, datetime
from pathlib import Path

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import (
    CatalogDecision,
    CatalogSource,
    PiiFlag,
    QualityScore,
    Sensitivity,
)
from aie_estimator.ingestion.loaders.filesystem import LoadedBlob
from aie_estimator.ingestion.parsers.budget_json import BudgetJsonParser
from aie_estimator.ingestion.parsers.protocol import ParseContext
from aie_estimator.ingestion.parsers.registry import ParserRegistry, default_registry
from aie_estimator.ingestion.parsers.transcript_txt import TranscriptTxtParser


def _make_source(name: str, fmt: str) -> CatalogSource:
    return CatalogSource(
        name=name,
        location="budgets" if fmt == "json" else "transcripts",
        format=fmt,  # type: ignore[arg-type]
        quality=QualityScore(completeness=4, consistency=4, actuality=4, reliability=4),
        sensitivity=Sensitivity(has_pii=True, pii_flags=(PiiFlag.PERSON,), access_level="confidential"),
        lineage=("CRM",),
        decision=CatalogDecision.INCLUDE,
    )


def test_default_registry_registers_json_and_txt_and_rejects_duplicates() -> None:
    """
    GIVEN default_registry()
    WHEN formats are looked up or duplicate formats registered
    THEN json/txt resolve, unknown lookup raises KeyError, and duplicate registration raises ValueError.
    """
    registry = default_registry()
    assert isinstance(registry.get("json"), BudgetJsonParser)
    assert isinstance(registry.get("txt"), TranscriptTxtParser)

    with pytest.raises(KeyError, match=r"No parser registered"):
        registry.get("pdf")

    with pytest.raises(ValueError, match=r"already has a registered parser"):
        registry.register(BudgetJsonParser())

    assert isinstance(ParserRegistry(), ParserRegistry)


def test_budget_json_parser_renders_seed_budget_markdown() -> None:
    """
    GIVEN a real seed JSON budget file
    WHEN BudgetJsonParser.parse runs
    THEN it emits structured Markdown with headings and structured extra metadata.
    """
    seed_path = Path("data/seed/budgets/BUDGET-2024-0001.json")
    blob = LoadedBlob(relative_path="budgets/BUDGET-2024-0001.json", bytes_=seed_path.read_bytes())
    context = ParseContext(
        source=_make_source("historical_budgets", "json"),
        source_version="1.0.0",
        ingested_at=datetime.now(UTC),
    )

    docs = list(BudgetJsonParser().parse(blob, context))
    assert len(docs) == 1
    doc = docs[0]
    assert "# Budget BUDGET-2024-0001" in doc.text
    assert "## Client" in doc.text
    assert "## Total" in doc.text
    assert "## Phases" in doc.text
    assert "## Notes" in doc.text
    assert doc.metadata.extra["budget_id"] == "BUDGET-2024-0001"
    assert doc.metadata.extra["currency"] == "EUR"


def test_budget_json_parser_handles_missing_optional_fields() -> None:
    """
    GIVEN a minimal JSON budget without phases or notes
    WHEN BudgetJsonParser.parse runs
    THEN it renders fallback phase text without crashing.
    """
    blob = LoadedBlob(relative_path="budgets/min.json", bytes_=b'{"budget_id": "B-1"}')
    context = ParseContext(
        source=_make_source("historical_budgets", "json"),
        source_version="1.0.0",
        ingested_at=datetime.now(UTC),
    )

    docs = list(BudgetJsonParser().parse(blob, context))
    assert len(docs) == 1
    assert "(no phases declared)" in docs[0].text
    assert "## Notes" not in docs[0].text


def test_transcript_txt_parser_tagged_and_legacy_modes() -> None:
    """
    GIVEN tagged 2025 and legacy 2023 seed transcripts
    WHEN TranscriptTxtParser.parse runs
    THEN tagged emits per-turn Documents with speaker/timestamp and legacy emits block Documents.
    """
    context = ParseContext(
        source=_make_source("meeting_transcripts", "txt"),
        source_version="1.0.0",
        ingested_at=datetime.now(UTC),
    )
    parser = TranscriptTxtParser()

    tagged_path = Path("data/seed/transcripts/transcripcion_2025-01-15_acme.txt")
    tagged_docs = list(
        parser.parse(
            LoadedBlob(relative_path="transcripts/transcripcion_2025-01-15_acme.txt", bytes_=tagged_path.read_bytes()),
            context,
        )
    )
    assert len(tagged_docs) >= 3
    assert tagged_docs[0].metadata.extra["format_mode"] == "tagged"
    assert tagged_docs[0].metadata.extra["speaker"] == "Laura Miller"
    assert tagged_docs[0].metadata.extra["timestamp"] == "10:00:12"

    legacy_path = Path("data/seed/transcripts/transcripcion_2023-09-08_legacy.txt")
    legacy_docs = list(
        parser.parse(
            LoadedBlob(relative_path="transcripts/transcripcion_2023-09-08_legacy.txt", bytes_=legacy_path.read_bytes()),
            context,
        )
    )
    assert len(legacy_docs) >= 2
    assert legacy_docs[0].metadata.extra["format_mode"] == "legacy"
    assert "speaker" not in legacy_docs[0].metadata.extra
