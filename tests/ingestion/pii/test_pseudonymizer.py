# tests/ingestion/pii/test_pseudonymizer.py

"""Unit tests for ConsistentPseudonymizer, AsyncMappingStore, and metadata scrubbing."""

# --- Python Standard Library ---
from datetime import UTC, datetime
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata, compute_content_hash
from aie_estimator.ingestion.pii.analyzer import PiiSpan
from aie_estimator.ingestion.pii.mapping_store import (
    AsyncMappingStore,
    InMemoryMappingStore,
    PostgresMappingStore,
)
from aie_estimator.ingestion.pii.pseudonymizer import (
    ConsistentPseudonymizer,
    compute_pii_hash,
    resolve_pii_salt,
)
from aie_estimator.persistence.repositories.mappings import Mapping
from aie_estimator.settings import Environment, Settings


class _StubAnalyzer:
    """Deterministic substring-matching PiiAnalyzer stub for offline unit tests."""

    def __init__(self, needles: dict[str, str]) -> None:
        self._needles = needles

    def analyze(self, text: str) -> list[PiiSpan]:
        spans: list[PiiSpan] = []
        for literal, entity_type in self._needles.items():
            start = 0
            while True:
                idx = text.find(literal, start)
                if idx == -1:
                    break
                spans.append(PiiSpan(start=idx, end=idx + len(literal), entity_type=entity_type, score=0.95))
                start = idx + len(literal)
        return spans


def _make_doc(doc_id: str, text: str, extra: dict[str, object] | None = None) -> Document:
    return Document(
        id=doc_id,
        text=text,
        content_hash=compute_content_hash(text),
        metadata=DocumentMetadata(
            source_name="historical_budgets",
            source_version="1.0.0",
            source_location="budgets/b1.json",
            ingested_at=datetime.now(UTC),
            contains_pii=True,
            pii_flags=(PiiFlag.PERSON, PiiFlag.EMAIL_ADDRESS, PiiFlag.BUDGET_ID),
            access_level="confidential",
            extra=extra or {},
        ),
    )


@pytest.mark.anyio
async def test_pseudonymizer_is_consistent_across_documents_and_preserves_offsets() -> None:
    """
    GIVEN two documents sharing the same person name, email, and budget ID
    WHEN ConsistentPseudonymizer.pseudonymize_documents is awaited
    THEN every occurrence maps to the same pseudonym, offsets remain intact, and plaintext never enters the store.
    """
    analyzer = _StubAnalyzer(
        {
            "Laura Miller": PiiFlag.PERSON.value,
            "laura.miller@acme.com": PiiFlag.EMAIL_ADDRESS.value,
            "BUDGET-2024-0001": PiiFlag.BUDGET_ID.value,
            "CLI-0042": PiiFlag.CLIENT_CODE.value,
        }
    )
    store = InMemoryMappingStore()
    assert isinstance(store, AsyncMappingStore)
    pseudonymizer = ConsistentPseudonymizer(analyzer=analyzer, mapping_store=store, salt="unit-salt")

    doc1 = _make_doc(
        "doc-1",
        "Contact Laura Miller (laura.miller@acme.com) for BUDGET-2024-0001 / CLI-0042.",
        extra={"speaker": "Laura Miller", "format_mode": "tagged", "tags": ["CLI-0042"]},
    )
    doc2 = _make_doc(
        "doc-2",
        "Follow-up with Laura Miller on BUDGET-2024-0001.",
        extra={"contact": "Laura Miller", "currency": "EUR"},
    )

    safe_docs = await pseudonymizer.pseudonymize_documents([doc1, doc2], source_name="historical_budgets")

    assert "Laura Miller" not in safe_docs[0].text
    assert "laura.miller@acme.com" not in safe_docs[0].text
    assert "BUDGET-2024-0001" not in safe_docs[0].text
    assert "CLI-0042" not in safe_docs[0].text

    # Recomputed content hash matches safe text.
    assert safe_docs[0].content_hash == compute_content_hash(safe_docs[0].text)
    assert safe_docs[0].content_hash != doc1.content_hash

    # Metadata.extra strings are scrubbed while technical keys stay untouched.
    fake_person = safe_docs[0].metadata.extra["speaker"]
    assert fake_person != "Laura Miller"
    assert safe_docs[1].metadata.extra["contact"] == fake_person
    assert safe_docs[0].metadata.extra["format_mode"] == "tagged"
    assert safe_docs[1].metadata.extra["currency"] == "EUR"
    assert safe_docs[0].metadata.extra["tags"][0] != "CLI-0042"
    assert fake_person in safe_docs[0].text
    assert fake_person in safe_docs[1].text

    # Mapping store keys are HMAC digests, never plaintext.
    expected_hash = compute_pii_hash("Laura Miller", "unit-salt")
    assert (PiiFlag.PERSON.value, expected_hash) in store._mappings
    for _, stored_hash in store._mappings:
        assert "Laura" not in stored_hash
        assert len(stored_hash) == 64


@pytest.mark.anyio
async def test_pseudonymizer_leaves_clean_document_unchanged() -> None:
    """
    GIVEN a document with no detected PII spans
    WHEN pseudonymize_document is awaited
    THEN text and content_hash remain identical.
    """
    pseudonymizer = ConsistentPseudonymizer(
        analyzer=_StubAnalyzer({}),
        mapping_store=InMemoryMappingStore(),
        salt="unit-salt",
    )
    doc = _make_doc("clean-1", "Standard architecture review notes.")

    safe_doc, applied = await pseudonymizer.pseudonymize_document(doc)

    assert safe_doc.text == doc.text
    assert safe_doc.content_hash == doc.content_hash
    assert applied == []


@pytest.mark.anyio
async def test_postgres_mapping_store_delegates_to_repository() -> None:
    """
    GIVEN a PostgresMappingStore wrapping a mocked MappingsRepository
    WHEN lookup_or_create and forget are awaited
    THEN the repository methods are called with source_name and their values returned.
    """
    repo = AsyncMock()
    repo.lookup_or_create.return_value = Mapping(
        entity_type=PiiFlag.PERSON.value,
        original_hash="abc123",
        pseudonym="Jordan Vance",
        source_names=("historical_budgets",),
        created_at=datetime.now(UTC),
    )
    repo.forget.return_value = True

    store = PostgresMappingStore(repo)
    pseudonym = await store.lookup_or_create(
        PiiFlag.PERSON.value,
        "abc123",
        lambda: "Unused",
        source_name="historical_budgets",
    )
    deleted = await store.forget(PiiFlag.PERSON.value, "abc123")

    assert pseudonym == "Jordan Vance"
    assert deleted is True
    repo.lookup_or_create.assert_awaited_once()
    repo.forget.assert_awaited_once_with(PiiFlag.PERSON.value, "abc123")


def test_salt_validation_rejects_missing_or_blank_salt() -> None:
    """
    GIVEN an empty salt or non-test Settings with pii_hash_salt=None
    WHEN compute_pii_hash, resolve_pii_salt, or ConsistentPseudonymizer is called
    THEN ValueError is raised.
    """
    with pytest.raises(ValueError, match=r"non-empty"):
        compute_pii_hash("Laura Miller", "   ")

    with pytest.raises(ValueError, match=r"non-empty"):
        ConsistentPseudonymizer(analyzer=_StubAnalyzer({}), mapping_store=InMemoryMappingStore(), salt="")

    prod_settings = Settings(app_env=Environment.PROD, pii_hash_salt=None)
    with pytest.raises(ValueError, match=r"PII_HASH_SALT"):
        resolve_pii_salt(prod_settings)
