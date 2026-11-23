# tests/ingestion/test_documents.py

# --- Python Standard Library ---
from datetime import UTC, datetime

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata, compute_content_hash


def test_document_and_metadata_accept_valid_payload() -> None:
    """
    GIVEN valid DocumentMetadata and Document fields
    WHEN the models are constructed and mutated in place
    THEN validation succeeds and mutability is preserved without frozen errors.
    """
    now = datetime.now(UTC)
    metadata = DocumentMetadata(
        source_name="historical_budgets",
        source_version="1.0.0",
        source_location="budgets/BUDGET-2024-0001.json",
        ingested_at=now,
        lineage=("CRM",),
        contains_pii=True,
        pii_flags=(PiiFlag.PERSON, PiiFlag.EMAIL),
        access_level="confidential",
        extra={"budget_id": "BUDGET-2024-0001"},
    )
    doc = Document(
        id="doc-1",
        text="hello",
        content_hash=compute_content_hash("hello"),
        metadata=metadata,
    )
    doc.text = "updated"
    assert doc.text == "updated"
    assert len(doc.content_hash) == 64


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (
            DocumentMetadata,
            {
                "source_name": "historical_budgets",
                "source_version": "1.0.0",
                "source_location": "budgets/a.json",
                "ingested_at": datetime.now(UTC),
                "unexpected": True,
            },
        ),
        (
            Document,
            {
                "id": "doc-1",
                "text": "hello",
                "metadata": DocumentMetadata(
                    source_name="historical_budgets",
                    source_version="1.0.0",
                    source_location="budgets/a.json",
                    ingested_at=datetime.now(UTC),
                ),
                "unexpected": True,
            },
        ),
    ],
)
def test_document_models_forbid_extra_fields(model_cls: type[DocumentMetadata] | type[Document], kwargs: dict[str, object]) -> None:
    """
    GIVEN an undeclared field passed to Document or DocumentMetadata
    WHEN the model is instantiated
    THEN ValidationError is raised due to extra='forbid'.
    """
    with pytest.raises(ValidationError, match=r"Extra inputs are not permitted"):
        model_cls(**kwargs)  # type: ignore[arg-type]
