# PostgreSQL Persistence & Migrations

## Overview

The persistence layer provides asynchronous raw-SQL PostgreSQL storage for offline ingestion job tracking (`ingestion_jobs`), idempotent quarantine records (`quarantined_records`), and GDPR-auditable pseudonym mappings (`pseudonym_mappings`) under [`src/aie_estimator/persistence/`](../src/aie_estimator/persistence/__init__.py):

- **Pool Lifecycle**: [`database.py`](../src/aie_estimator/persistence/database.py) wraps `psycopg_pool.AsyncConnectionPool` (`open_pool`, `close_pool`) using single-source [`Settings`](../src/aie_estimator/settings.py) PostgreSQL fields (`postgres_host`, `postgres_port`, `postgres_user`, `postgres_password`, `postgres_db`, and `get_database_kwargs()`).
- **Lifespan & DI**: [`Server.lifespan`](../src/aie_estimator/server.py) stores the active pool on `app.state.db_pool` when `POSTGRES_PASSWORD` is configured, marks orphaned `pending`/`running` jobs as `failed`, skips initialization cleanly when unconfigured (`APP_ENV=test`), and maps `psycopg.Error` and `ConnectionError` to HTTP `503`. FastAPI dependencies [`get_db_pool`](../src/aie_estimator/dependencies.py), [`get_jobs_repository`](../src/aie_estimator/dependencies.py), [`get_mappings_repository`](../src/aie_estimator/dependencies.py), and [`get_quarantined_records_repository`](../src/aie_estimator/dependencies.py) resolve repositories from application state.
- **Repositories**: [`JobsRepository`](../src/aie_estimator/persistence/repositories/jobs.py), [`MappingsRepository`](../src/aie_estimator/persistence/repositories/mappings.py), and [`QuarantinedRecordsRepository`](../src/aie_estimator/persistence/repositories/quarantine.py) execute `psycopg.sql` queries returning frozen Pydantic models (`Job`, `Mapping`, `QuarantinedRecord`).
- **Versioned SQL Deltas**: [`migrate.py`](../src/aie_estimator/persistence/migrate.py) runs ordered `.sql` scripts from [`persistence/migrations/`](../src/aie_estimator/persistence/migrations/__init__.py) tracked in `schema_migrations` ([`0001_initial_schema.sql`](../src/aie_estimator/persistence/migrations/0001_initial_schema.sql), [`0002_quarantine_and_validation_report.sql`](../src/aie_estimator/persistence/migrations/0002_quarantine_and_validation_report.sql)).
- **Containers**: [`docker-compose.dev.yml`](../docker/docker-compose.dev.yml) and [`docker-compose.prod.yml`](../docker/docker-compose.prod.yml) provision `pgvector/pgvector:pg17` (deferring `CREATE EXTENSION vector` to Session 7).

---

## Usage Examples

Start PostgreSQL in development and apply pending SQL migrations:

```bash
docker compose -f docker/docker-compose.dev.yml up -d postgres
uv run python -m aie_estimator.persistence.migrate
```

Programmatic repository usage with an opened pool:

```python
job = await JobsRepository(pool).create("historical_budgets")
mapping = await MappingsRepository(pool).lookup_or_create(
    "PERSON",
    entity_hash,
    lambda: "Pseudonym Name",
)
```

---

## Technical Considerations

1. **Windows `SelectorEventLoop` Compatibility**: `psycopg` 3 async connections reject Python 3.14's default `ProactorEventLoop` on Windows. Dev Uvicorn `--reload` already uses `SelectorEventLoop`; [`__main__.py`](../src/aie_estimator/__main__.py) configures `loop="asyncio:SelectorEventLoop"` for non-reload Windows runs, and [`tests/conftest.py`](../tests/conftest.py) configures `anyio_backend` and `TestClient` `backend_options` with `loop_factory=asyncio.SelectorEventLoop` without deprecated event-loop policies.
2. **Wheel-Safe SQL Packaging**: [`migrate.py`](../src/aie_estimator/persistence/migrate.py) loads `.sql` files via `importlib.resources.files("aie_estimator.persistence.migrations")` so migrations resolve inside [`docker/Dockerfile`](../docker/Dockerfile)'s non-editable `uv_build` wheel. Migration scripts use pure PostgreSQL SQL without `psql` meta-commands (`\connect`).
3. **Zero DDL in Lifespan**: Schema migrations run explicitly via CLI rather than inside `Server.lifespan`, preventing multi-worker DDL lock contention and keeping app startup deterministic.
4. **Concurrency & Secret Hygiene**: `MappingsRepository.lookup_or_create` uses `INSERT ... ON CONFLICT (entity_type, original_hash) DO NOTHING RETURNING` with a `SELECT` fallback and atomic `source_names text[]` provenance appending to remain deterministic under parallel writers; neither pool startup nor repositories log `postgres_password` or plaintext PII.
