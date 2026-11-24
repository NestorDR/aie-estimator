# tests/ingestion/cleaning/test_text.py

# --- Python Standard Library ---
from datetime import UTC, datetime

# --- App modules ---
from aie_estimator.ingestion.cleaning.text import clean_text_documents
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata


def _make_doc(doc_id: str, text: str) -> Document:
    return Document(
        id=doc_id,
        text=text,
        metadata=DocumentMetadata(
            source_name="meeting_transcripts",
            source_version="1.0.0",
            source_location="transcripts/sample.txt",
            ingested_at=datetime.now(UTC),
        ),
    )


def test_clean_text_documents_normalizes_and_filters_placeholders() -> None:
    """
    GIVEN non-tabular Documents with NFKC fullwidth text, control chars, disguised nulls, and short text
    WHEN clean_text_documents runs
    THEN valid text is normalized preserving line breaks while placeholders and short entries are filtered.
    """
    docs = [
        _make_doc("1", "Ｈｅｌｌｏ\x00 world   \nLine two   "),
        _make_doc("2", "TBD"),
        _make_doc("3", "short"),
    ]
    result = clean_text_documents(docs, min_length=10)

    assert len(result.documents) == 1
    assert result.documents[0].text == "Hello world\nLine two"
    assert result.filtered_count == 2
    assert result.report["valid_rows"] == 1
    assert result.report["discarded_rows"] == 2
    assert result.report["failures_by_check"] == {"disguised_null": 1, "min_length": 1}
