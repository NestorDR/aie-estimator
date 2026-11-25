# GDPR PII Detection & Reversible Pseudonymization

## Overview

The PII subsystem protects cleaned, normalized offline ingestion [`Document`](../src/aie_estimator/ingestion/documents/models.py) records before Session 7 chunking and vectorization. It detects personal and commercial entities through a pluggable [`PiiAnalyzer`](../src/aie_estimator/ingestion/pii/analyzer.py) backed by Microsoft Presidio and `spaCy`, replaces spans consistently right-to-left via [`ConsistentPseudonymizer`](../src/aie_estimator/ingestion/pii/pseudonymizer.py), persists only salted HMAC-SHA256 digests in [`public.pseudonym_mappings`](../src/aie_estimator/persistence/migrations/0001_initial_schema.sql), and exposes GDPR Article 17 erasure through [`forget_entity`](../src/aie_estimator/ingestion/pii/forget.py).

## Usage Examples

### Pseudonymize canonical documents asynchronously

```python
from aie_estimator.ingestion.pii.analyzer import PresidioPiiAnalyzer
from aie_estimator.ingestion.pii.mapping_store import InMemoryMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer

pseudonymizer = ConsistentPseudonymizer(
    analyzer=PresidioPiiAnalyzer(spacy_model="en_core_web_sm", language="en", score_threshold=0.7),
    mapping_store=InMemoryMappingStore(),
    salt="your-secret-salt",
)
safe_docs = await pseudonymizer.pseudonymize_documents(documents, source_name="historical_budgets")
```

### Execute GDPR Article 17 erasure via CLI

```bash
uv run python -m aie_estimator.ingestion.pii.forget --entity-type PERSON --value "Laura Miller"
```

## Technical Considerations

- **Async/sync pipeline seam**: [`ingest_source()`](../src/aie_estimator/ingestion/orchestrator.py) remains synchronous and DB-free (offloaded via `asyncio.to_thread`). Pseudonymization runs asynchronously in [`_run_ingestion_job`](../src/aie_estimator/routers/ingestion_router.py) immediately after `ingest_source()` when `source.sensitivity.has_pii` and `Settings.pii_pseudonymization_enabled` are both `True`, offloading CPU-bound `spaCy` inference per text via `asyncio.to_thread` while awaiting PostgreSQL mapping lookups on the event loop.
- **Canonical entity alignment**: [`PiiFlag`](../src/aie_estimator/ingestion/catalog/models.py) uses Presidio canonical entity names (`EMAIL_ADDRESS`, `IBAN_CODE`, `CREDIT_CARD`, `NUMERIC_IDENTIFIER`, `PERSON`, `LOCATION`, `ORGANIZATION`, `CLIENT_CODE`, `BUDGET_ID`, `PHONE_NUMBER`). Domain recognizers built by [`build_domain_recognizers`](../src/aie_estimator/ingestion/pii/recognizers.py) (`BudgetIdRecognizer`, `ClientCodeRecognizer`, `KnownClientRecognizer`) share compiled regex patterns with online [`guardrails/input.py`](../src/aie_estimator/guardrails/input.py) via [`utils/constants.py`](../src/aie_estimator/utils/constants.py).
- **Document integrity**: `ConsistentPseudonymizer` resolves overlapping detections deterministically (higher score, then longer span), applies replacements right-to-left, recursively scrubs string leaves in `DocumentMetadata.extra` (including all 10 cleaned budget fields and nested `phases[].name` while skipping technical keys `format_mode`, `timestamp`, `currency`, and `signed_at`), and recomputes `Document.content_hash` on the safe text. Known `en_core_web_sm` behavior: uppercase technical acronyms such as `"MVP"` and `"ERP"` score `0.85` as `ORGANIZATION` and are pseudonymized identically in both `Document.text` and `DocumentMetadata.extra`.
- **Mapping store security & provenance**: Plaintext PII never reaches PostgreSQL. [`MappingsRepository`](../src/aie_estimator/persistence/repositories/mappings.py) keys on `(entity_type, original_hash)` and records multi-source appearances atomically in `source_names text[]`. `Settings.pii_hash_salt` (`SecretStr`) is required in `dev`/`prod` and defaults to a deterministic test-only secret in `APP_ENV=test`.
- **Observability**: Structured logs follow the lowercase `snake_case` taxonomy (`pseudonym_mapping_created`, `pseudonym_mapping_updated`, `pseudonym_mapping_deleted`, `ingestion_pii_completed`, `ingestion_pii_skipped`, `ingestion_pii_erasure_completed`, `ingestion_pii_erasure_not_found`) and never emit plaintext PII.
- **Session 7 handoff seams**: `Document.id` and `DocumentMetadata.source_location` remain unpseudonymized for deterministic incremental re-indexing, so downstream RAG citations must use pseudonymized `DocumentMetadata.extra` fields. `forget()` currently returns `bool` and can switch to `DELETE ... RETURNING` in Session 7 if vector-chunk deletion is coupled to mapping erasure.
