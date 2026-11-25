# AIE Estimator — Backend

Python 3.14+ FastAPI backend and Streamlit conversational UI for automated software engineering effort estimation.

See the [monorepo README](../README.md) and [documentation index](../docs/) for system architecture, Docker Compose infrastructure (`auto\dc.cmd`), API contracts, and feature guides.

## Setup

```bash
cd backend
uv sync
```

Create `src/aie_estimator/.env.dev` from [`src/aie_estimator/.env.example`](src/aie_estimator/.env.example) and configure provider API keys and infrastructure settings.

## Running Locally

All commands below assume `backend/` as the working directory:

```bash
# Apply pending PostgreSQL migrations
auto\migrate.cmd          # or: uv run python -m aie_estimator.foundation.persistence.migrate

# FastAPI backend (http://localhost:8000, Swagger UI at /docs)
auto\dev.cmd              # or: uv run aie-estimator

# Streamlit UI (http://localhost:8501)
auto\stl.cmd              # or: uv run streamlit run streamlit_app.py

# Inspect cleaned, pseudonymized historical budgets as JSON
auto\inspect-budgets.cmd  # or: uv run python -m aie_estimator.ingestion.inspect_budgets
```

## Code Quality & Tests

```bash
# Lint & format
auto\lint.cmd             # or: uv run ruff check .

# Full test suite with 85% branch coverage gate
auto\test.cmd             # or: uv run pytest --cov-fail-under=85
```
