# Data Cleaning, Normalization & Schema Validation

## Overview

The cleaning and validation stage sits between format parsing and canonical `Document` normalization under [`../backend/src`](../backend/src/aie_estimator/ingestion/cleaning/__init__.py) to enforce data quality rules before downstream PII processing and indexing:

- **Tabular Budget Cleaning**: [`clean_budget_records()`](../backend/src/aie_estimator/ingestion/cleaning/budget_records.py) normalizes disguised nulls (`"TBD"`, `"N/A"`, `""`, etc. to `pd.NA`), currency aliases/casing (`"eur"`, `"€"` -> `"EUR"`), mixed ISO/`DD/MM/YYYY` dates (`dayfirst=True`), and numeric amounts, then computes deterministic SHA-256 `content_hash` digests per row.
- **Schema & Policy Routing**: [`BudgetRecord`](../backend/src/aie_estimator/ingestion/cleaning/schemas.py) (`pandera.pandas` `DataFrameSchema` with `strict=True`, `lazy=True`, and cross-column `phases_sum_matches_total` check) and [`validate_with_policy()`](../backend/src/aie_estimator/ingestion/cleaning/policy.py) partition records into `valid`, `quarantined`, and `discarded` outcomes with a JSON-serializable `report`.
- **Non-Tabular Text Cleaning**: [`clean_text_documents()`](../backend/src/aie_estimator/ingestion/cleaning/text.py) applies Unicode `NFKC`, strips control characters and trailing line whitespace while preserving line breaks, and filters disguised nulls or undersized texts.
- **Quarantine Persistence**: Migration [`0002_quarantine_and_validation_report.sql`](../backend/src/aie_estimator/persistence/migrations/0002_quarantine_and_validation_report.sql) adds `validation_report` (`jsonb`) to `ingestion_jobs` and creates `quarantined_records` managed idempotently by [`QuarantinedRecordsRepository`](../backend/src/aie_estimator/persistence/repositories/quarantine.py).

---

## Usage Examples

Run tabular cleaning and policy validation on decoded budget records:

```python
from aie_estimator import clean_budget_records
from aie_estimator import validate_with_policy

cleaned = clean_budget_records(raw_records)
result = validate_with_policy(cleaned)
# result.valid, result.quarantined, result.discarded, result.report
```

---

## Technical Considerations

1. **Pre-Render Tabular Interception**: [`BudgetJsonParser.decode_record()`](../backend/src/aie_estimator/ingestion/parsers/budget_json.py) decodes raw JSON payloads for batch cleaning and validation in [`ingest_source()`](../backend/src/aie_estimator/ingestion/orchestrator.py) *before* `render_record()` emits Markdown, ensuring normalized fields appear in both `Document.text` and `metadata.extra`.
2. **Divergent Duplicate Quarantine**: Exact `content_hash` duplicates collapse to one row; all rows sharing a `budget_id` with differing `content_hash` values (`BUDGET-2024-0005-v1` and `v2`) route to `quarantined` with reason `"divergent_duplicate"` until reviewed.
3. **Strict Schema with Auxiliary Rendering Fields**: `BudgetRecord` declares optional rendering columns (`phases`, `contact`, `contact_email`, `notes`, `content_hash`, `_source_location`) so `strict=True` rejects unexpected schema drift without dropping fields needed for Markdown rendering.
4. **Idempotent Quarantine Upsert**: [`QuarantinedRecordsRepository.upsert_many()`](../backend/src/aie_estimator/persistence/repositories/quarantine.py) conflicts on `(source_name, record_key, content_hash)` to refresh `job_id` and increment `seen_count` across repeated runs without duplicating rows.
