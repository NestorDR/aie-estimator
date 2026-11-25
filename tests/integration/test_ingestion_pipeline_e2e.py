# tests/integration/test_ingestion_pipeline_e2e.py

"""
Offline end-to-end integration test for the Session 6 ingestion pipeline.

Exercises the complete corpus flow (catalog -> loader -> parser -> cleaning ->
validation -> normalization -> Presidio NER + custom recognizer pseudonymization ->
GDPR Article 17 erasure) against the real repository seed corpus without network
or PostgreSQL dependencies.
"""

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata, compute_content_hash
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.orchestrator import ingest_source
from aie_estimator.ingestion.parsers.budget_json import CleanedBudget
from aie_estimator.ingestion.parsers.registry import default_registry
from aie_estimator.ingestion.pii.analyzer import PresidioPiiAnalyzer
from aie_estimator.ingestion.pii.forget import forget_entity
from aie_estimator.ingestion.pii.mapping_store import InMemoryMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer, compute_pii_hash

_CATALOG_PATH: Final[Path] = Path("data/catalog/catalog.yaml")
_SEED_ROOT: Final[Path] = Path("data/seed")
_TEST_SALT: Final[str] = "offline-e2e-pii-salt"
_RAW_PII_VALUES: Final[tuple[str, ...]] = (
    "Acme Spain Ltd.",
    "BetaNorte Distribution",
    "Laura Miller",
    "laura.miller@acme.com",
    "CLI-0042",
    "James Reynolds",
    "j.reynolds@betanorte.com",
    "CLI-0078",
    "BUDGET-2024-0001",
    "BUDGET-2024-0002",
)


@pytest.fixture(scope="module")
def presidio_analyzer() -> PresidioPiiAnalyzer:
    """
    Load the real spaCy-backed Presidio analyzer once for this module.

    :return: Configured PresidioPiiAnalyzer instance.
    """
    return PresidioPiiAnalyzer(spacy_model="en_core_web_sm", language="en", score_threshold=0.7)


@pytest.mark.anyio
async def test_offline_ingestion_pipeline_pseudonymizes_seed_corpus(
    presidio_analyzer: PresidioPiiAnalyzer,
) -> None:
    """
    GIVEN the audited repository catalog and synthetic seed corpus
    WHEN ingest_source, ConsistentPseudonymizer, and forget_entity run end-to-end
    THEN valid budgets are pseudonymized with zero raw PII leakage, consistent pseudonyms,
         updated content hashes, divergent duplicates quarantined, and mappings erasable.

    :param presidio_analyzer: Shared module-scoped PresidioPiiAnalyzer fixture.
    """
    catalog = load_catalog(_CATALOG_PATH)
    ingestion_result = ingest_source(
        catalog=catalog,
        source_name="historical_budgets",
        loader=FileSystemLoader(_SEED_ROOT),
        registry=default_registry(),
    )

    assert len(ingestion_result.documents) == 2
    quarantined_keys = {record.record_key for record in ingestion_result.quarantined}
    assert "BUDGET-2024-0005" in quarantined_keys
    assert any("divergent_duplicate" in record.failure_reasons for record in ingestion_result.quarantined)

    raw_hashes = {doc.id: doc.content_hash for doc in ingestion_result.documents}
    mapping_store = InMemoryMappingStore()
    pseudonymizer = ConsistentPseudonymizer(
        analyzer=presidio_analyzer,
        mapping_store=mapping_store,
        salt=_TEST_SALT,
    )

    safe_docs = await pseudonymizer.pseudonymize_documents(
        ingestion_result.documents,
        source_name="historical_budgets",
    )

    assert len(safe_docs) == len(ingestion_result.documents)
    for safe_doc in safe_docs:
        cleaned_budget = CleanedBudget.model_validate(safe_doc.metadata.extra)
        assert len(cleaned_budget.phases) >= 3
        serialized_extra = json.dumps(safe_doc.metadata.extra)
        serialized_budget = cleaned_budget.model_dump_json()
        for raw_value in _RAW_PII_VALUES:
            assert raw_value not in safe_doc.text
            assert raw_value not in serialized_extra
            assert raw_value not in serialized_budget
        assert safe_doc.content_hash == compute_content_hash(safe_doc.text)
        assert safe_doc.content_hash != raw_hashes[safe_doc.id]

    # Probe the public AsyncMappingStore contract with a sentinel factory: a cache hit returns
    # the stored pseudonym without invoking the factory, avoiding private `_mappings` coupling.
    laura_pseudonym = await mapping_store.lookup_or_create(
        PiiFlag.PERSON.value,
        compute_pii_hash("Laura Miller", _TEST_SALT),
        lambda: "SENTINEL-NOT-CREATED",
    )
    assert laura_pseudonym != "SENTINEL-NOT-CREATED"
    assert laura_pseudonym in safe_docs[0].text

    repeat_doc = Document(
        id="synthetic:repeat",
        text="Follow-up call with Laura Miller (laura.miller@acme.com).",
        content_hash=compute_content_hash("Follow-up call with Laura Miller (laura.miller@acme.com)."),
        metadata=DocumentMetadata(
            source_name="historical_budgets",
            source_version=catalog.version,
            source_location="synthetic/repeat.txt",
            ingested_at=safe_docs[0].metadata.ingested_at,
        ),
    )
    repeat_safe_docs = await pseudonymizer.pseudonymize_documents([repeat_doc], source_name="historical_budgets")
    assert laura_pseudonym in repeat_safe_docs[0].text

    # Verify GDPR Article 17 erasure deletes the mapping and is idempotent on repeat calls.
    erased = await forget_entity(PiiFlag.PERSON, "Laura Miller", salt=_TEST_SALT, store=mapping_store)
    assert erased is True
    assert await forget_entity(PiiFlag.PERSON, "Laura Miller", salt=_TEST_SALT, store=mapping_store) is False
