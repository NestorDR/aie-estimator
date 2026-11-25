# Versioned Data Catalog

## Overview

The versioned data catalog defines the audited source-of-truth contract for offline ingestion under [`../backend/src`](../backend/src/aie_estimator/ingestion/catalog/__init__.py):

- **Catalog Models**: [`models.py`](../backend/src/aie_estimator/ingestion/catalog/models.py) defines `extra="forbid"` Pydantic v2 models (`DataCatalog`, `CatalogSource`, `QualityScore`, `Sensitivity`, `CatalogDecision`, `PiiFlag`).
- **Loader**: [`loader.py`](../backend/src/aie_estimator/ingestion/catalog/loader.py) provides `load_catalog()` using `yaml.safe_load` and structured logging (`catalog_loaded`, `catalog_validation_failed`).
- **Census Inspector**: [`inspect.py`](../backend/src/aie_estimator/ingestion/catalog/inspect.py) provides `inspect_folder()` and `inspect_root()` returning `FolderFacts` without scoring or mutating YAML.
- **Audit Report**: [`report.py`](../backend/src/aie_estimator/ingestion/catalog/report.py) provides `generate_audit_report()` to render Markdown summaries for human or CI review.
- **Repository Corpus**: [`../backend/data`](../backend/data/catalog/catalog.yaml) and [`../backend/data`](../backend/data/seed/) store the audited seed catalog and synthetic fixtures wired through [`Settings`](../backend/src/aie_estimator/settings.py) (`ingestion_data_root`, `catalog_path`).

---

## Usage Examples

Load and validate the configured catalog and render its Markdown audit report:

```python
from aie_estimator import inspect_root
from aie_estimator import load_catalog
from aie_estimator import generate_audit_report
from aie_estimator import get_settings

settings = get_settings()
facts = inspect_root(settings.ingestion_data_root)
catalog = load_catalog(settings.catalog_path)
markdown = generate_audit_report(catalog)
```

---

## Technical Considerations

1. **Per-Run Reload in DI**: [`get_catalog`](../backend/src/aie_estimator/dependencies.py) reads and validates `settings.catalog_path` on each resolution without `@lru_cache`, so catalog edits take effect on the next offline ingestion run without restarting FastAPI.
2. **Advisory `is_rag_ready` vs. Explicit Gate**: `QualityScore.is_rag_ready` evaluates `all(score >= 3)` as an advisory indicator surfaced in the audit report, while `CatalogDecision.INCLUDE` (`included_sources()`) remains the explicit ingestion gate; `review` and `exclude` entries require a non-empty `decision_reason`.
3. **Closed Sensitivity Taxonomy**: `PiiFlag` (`StrEnum`) aligns Presidio canonical entity categories (`EMAIL_ADDRESS`, `IBAN_CODE`, `CREDIT_CARD`, `NUMERIC_IDENTIFIER`, `PERSON`, `LOCATION`, `ORGANIZATION`, `CLIENT_CODE`, `BUDGET_ID`, `PHONE_NUMBER`) across catalog declarations, online guardrails, and offline PII pseudonymization.
4. **Container Path Overrides**: Because [`../backend/Dockerfile`](../backend/Dockerfile) packages only `../backend/src` into the runtime wheel, [`docker/docker-compose.prod.yml`](../docker/docker-compose.prod.yml) bind-mounts `../data:/app/data:ro` on `estimator-prod` so default `INGESTION_DATA_ROOT` and `CATALOG_PATH` values resolve out of the box; custom deployments can still override both via environment variables.
