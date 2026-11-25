# Multi-Format Ingestion Pipeline

## Overview

The offline multi-format ingestion pipeline converts catalog-approved enterprise sources into canonical `Document` records through a modular `loaders -> parsers -> normalizers` flow under [`../backend/src`](../backend/src/aie_estimator/ingestion/__init__.py):

- **Canonical Contract**: [`documents/models.py`](../backend/src/aie_estimator/ingestion/documents/models.py) defines `Document`, `DocumentMetadata` (`extra="forbid"`), and `compute_content_hash()` (SHA-256 text fingerprint used for Topic 5 deduplication and Session 7 incremental re-indexing).
- **Loaders**: [`loaders/filesystem.py`](../backend/src/aie_estimator/ingestion/loaders/filesystem.py) provides `LoadedBlob` and `FileSystemLoader` rooted at `Settings.ingestion_data_root`.
- **Parsers**: [`parsers/protocol.py`](../backend/src/aie_estimator/ingestion/parsers/protocol.py) defines `ParseContext` and the `Parser` `Protocol`; [`parsers/registry.py`](../backend/src/aie_estimator/ingestion/parsers/registry.py) provides `ParserRegistry` and `default_registry()`, preloading [`BudgetJsonParser`](../backend/src/aie_estimator/ingestion/parsers/budget_json.py) (`json`, exposing `decode_record()`, `render_record()`, and strict `BudgetPhase`/`CleanedBudget` validation models over the 10 cleaned fields retained in `DocumentMetadata.extra`) and [`TranscriptTxtParser`](../backend/src/aie_estimator/ingestion/parsers/transcript_txt.py) (`txt`).
- **Cleaning, Normalizers & Orchestrator**: [`cleaning/`](../backend/src/aie_estimator/ingestion/cleaning/__init__.py) and [`normalizers/canonical.py`](../backend/src/aie_estimator/ingestion/normalizers/canonical.py) (`normalize_documents`) enforce data quality rules and authoritative catalog metadata; [`orchestrator.py`](../backend/src/aie_estimator/ingestion/orchestrator.py) (`ingest_source`, `IngestionResult`, `IngestionRejected`) gates runs on `CatalogDecision.INCLUDE` and returns canonical `documents`, `quarantined` records, and `report`.
- **HTTP Control Plane**: [`schemas/ingestion.py`](../backend/src/aie_estimator/schemas/ingestion.py) and [`routers/ingestion_router.py`](../backend/src/aie_estimator/routers/ingestion_router.py) expose `POST /api/v1/ingestion/runs` (`202 Accepted`) and `GET /api/v1/ingestion/jobs/{job_id}`, wired in [`dependencies.py`](../backend/src/aie_estimator/dependencies.py) and [`server.py`](../backend/src/aie_estimator/server.py).

---

## Usage Examples

Trigger an offline ingestion run and poll its job status:

```http
POST /api/v1/ingestion/runs
Content-Type: application/json

{"source_name": "historical_budgets"}
```

```http
GET /api/v1/ingestion/jobs/<job_id>
```

Programmatic execution against the repository catalog:

```python
from aie_estimator import get_catalog, get_filesystem_loader, get_parser_registry
from aie_estimator import ingest_source
from aie_estimator import get_settings

settings = get_settings()
result = ingest_source(
    catalog=get_catalog(settings),
    source_name="historical_budgets",
    loader=get_filesystem_loader(settings),
    registry=get_parser_registry(),
)
# result.documents, result.quarantined, result.report
```

Inspect cleaned, GDPR-pseudonymized budget JSON on `stdout` without writing to PostgreSQL or disk ([`inspect_budgets.py`](../backend/src/aie_estimator/ingestion/inspect_budgets.py)):

```bash
uv run python -m aie_estimator.ingestion.inspect_budgets
# Windows convenience wrapper (status banners routed to stderr so stdout stays valid JSON):
auto\inspect-budgets --no-pii
```

---

## Technical Considerations

1. **Isolation from Online Attachments**: [`attachments/extractor.py`](../backend/src/aie_estimator/attachments/extractor.py) remains dedicated to online single-turn chat attachments with character budgets and XML delimiters, while offline corpus parsing lives exclusively in `ingestion/`.
2. **Catalog-over-Parser Provenance Override**: `normalize_documents()` overwrites `source_name`, `source_version`, `source_location`, `ingested_at`, `lineage`, `contains_pii`, `pii_flags`, and `access_level` after `parser.parse()` returns, preventing format parsers from spoofing catalog metadata.
3. **Non-Blocking Async Dispatch & OTel Lifecycle**: `POST /api/v1/ingestion/runs` queues an `async def` background task that awaits [`JobsRepository`](../backend/src/aie_estimator/persistence/repositories/jobs.py) on the `SelectorEventLoop` while offloading synchronous `ingest_source()` via `asyncio.to_thread()`, then runs async PII pseudonymization via [`ConsistentPseudonymizer`](../backend/src/aie_estimator/ingestion/pii/pseudonymizer.py) when `source.sensitivity.has_pii` and `Settings.pii_pseudonymization_enabled` are `True`. Each background run opens an isolated OpenTelemetry root span linked to the triggering HTTP `SpanContext`, [`Server.lifespan`](../backend/src/aie_estimator/server.py) shuts down `TracerProvider` on exit, and `IngestionRejected` maps to HTTP `400`.
4. **Deferred Formats**: Only `json` and `txt` parsers are registered; `xlsx`, `docx`, and `pdf` parsers are deferred with zero new dependencies until audited sources require them.
