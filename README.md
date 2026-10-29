# AIE Estimator

FastAPI backend service for automated software engineering effort estimation using Context-Augmented Generation (CAG) queryable via a conversational user interface based on Streamlit.

## About

This is my personal repository where I am coding the project while following the [LIDR AI Engineering program](https://lidr.notion.site/ai-engineering). It explores production Generative AI: Context-Augmented Generation (CAG), advanced RAG, multi-agent workflows, LLMOps, and evaluation with Python, FastAPI, and leading LLMs, starting with an automated software engineering effort estimator.

## Overview

AIE Estimator processes client meeting transcripts and generates structured project estimates—including task breakdowns, estimated hours, team compositions, and delivery timelines.

Rather than relying on retrieval-augmented generation (RAG) with vector databases, the service uses a Context-Augmented Generation (CAG) architecture. Historical reference estimations are directly injected into the LLM system prompt as few-shot context, grounding model output in established project patterns with minimal operational complexity.

## Architecture

The project enforces strict separation of concerns across dedicated application layers:

- **Schemas (`src/aie_estimator/schemas`)**: Defines domain data schemas and Pydantic models (`EstimationRequest`, `EstimationResponse`) for request payload validation and response contracts.
- **Routers (`src/aie_estimator/routers/`)**: Defines HTTP routes, endpoint documentation, status codes, and request/response handling.
- **Services (`src/aie_estimator/services/`)**: Encapsulates business logic and LLM interactions. Assembles the CAG prompt and invokes LLM APIs with retry mechanisms.
- **Context (`src/aie_estimator/context/`)**: Stores static reference datasets and few-shot estimation examples injected into model prompts.
- **Settings (`src/aie_estimator/settings.py`)**: Environment-aware configuration and settings management powered by `pydantic-settings`, supporting structured JSON logging and credential isolation via `SecretStr`.
- **Server (`src/aie_estimator/server.py`)**: Centralizes FastAPI application initialization, lifespan event handlers, router registration, and global exception mapping.

### Context-Augmented Generation (CAG) Workflow

```mermaid
flowchart TD
    Client(["Streamlit UI / Caller"]) -->|"POST /api/v1/estimate\n(EstimationRequest)"| Router["Router: EstimationsRouter\n(Pydantic Validation)"]

    subgraph ServiceLayer ["LLM Service Layer"]
        Router -->|"Valid transcription"| PromptBuilder["CAG Prompt Assembly"]

        subgraph ContextStore ["CAG Context"]
            Examples[("Reference Estimations\ncontext/examples.py")]
            Role["System Instructions\n(Senior Estimator Role)"]
        end

        Examples -->|"Inject few-shot examples"| PromptBuilder
        Role -->|"Inject role & guidelines"| PromptBuilder

        PromptBuilder -->|"Assemble payload"| CacheCheck{"Exact-Match Cache\n(Redis SHA-256)"}

        CacheCheck -->|"Cache Hit"| ServiceOutput["Result Dictionary"]
        CacheCheck -->|"Cache Miss"| Dispatcher["Multi-Provider Dispatcher\n(Retry & Backoff)"]
    end

    Dispatcher -->|"API call"| LLMProvider["LLM Provider\n(OpenAI & Anthropic APIs)"]
    LLMProvider -->|"Estimation text & usage"| CacheStore["Store in Redis Cache"]
    CacheStore --> ServiceOutput

    ServiceOutput -->|"Result dictionary"| Router
    Router -->|"HTTP 200: EstimationResponse\n(estimation, model, provider)"| Client
```

## Project Structure

```
aie-estimator/
├── auto/                               # Developer automation scripts (dev.cmd, stl.cmd, dc.cmd, test.cmd, lint.cmd, etc.)
├── docker/                             # Containerization and Compose configurations
├── docs/                               # Architecture and feature guides
├── src/
│   └── aie_estimator/
│       ├── __init__.py
│       ├── __main__.py                 # Application CLI entry point
│       ├── settings.py                 # Pydantic BaseSettings configuration
│       ├── server.py                   # FastAPI server setup and lifespan handling
│       ├── .env.example                # Example environment configuration
│       ├── context/
│       │   ├── __init__.py
│       │   └── examples.py             # Historical few-shot estimation examples
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── estimations.py          # EstimationRequest and EstimationResponse schemas
│       │   └── models.py               # Model catalog and specification schemas
│       ├── routers/
│       │   ├── __init__.py
│       │   ├── estimations_router.py   # POST /api/v1/estimate endpoint
│       │   └── health_router.py        # GET /health status endpoint
│       └── services/
│           ├── __init__.py
│           ├── cache.py                # Deterministic exact-match Redis cache
│           ├── evaluation.py           # Output evaluation and scoring
│           └── llm_service.py          # CAG prompt assembly and multi-provider integration
├── tests/                              # Unit and integration test suites
├── .dockerignore
├── .gitignore
├── .gitattributes
├── .pre-commit-config.yaml
├── pyproject.toml                      # Project metadata, dependencies, and tools config
├── README.md                           # Project documentation
├── streamlit_app.py                    # Streamlit conversational chat interface
└── uv.lock                             # Locked dependency graph
```

## Requirements

- Python >= 3.14
- [`uv`](https://docs.astral.sh/uv/) package manager
- Docker Desktop / Docker Compose (for Redis exact-match cache and containerized deployments)
- OpenAI API key (`OPENAI_API_KEY`) and/or Anthropic API key (`ANTHROPIC_API_KEY`)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd aie-estimator
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure environment variables:**
   Copy the example environment file to the target environment configuration:
   ```bash
   cp src/aie_estimator/.env.example src/aie_estimator/.env.dev
   ```

    Edit `src/aie_estimator/.env.dev` and configure your API keys:
   ```ini
   APP_ENV=dev
   LOG_LEVEL=INFO
   JSON_LOGGING=false
   ENABLE_FILE_LOGGING=false
   SERVER_HOST=localhost
   SERVER_PORT=8000
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   LLM_MODEL=gpt-4o-mini
   REDIS_HOST=localhost
   REDIS_PORT=6379
   CACHE_TTL=86400
   ```

## Running the Application

### 1. Docker Compose Environments

Start the detached Docker Compose services from the project root using `auto\dc.cmd <dev|prod>`:

- **Development (Redis Cache Service Only)**:
  Builds and launches the Redis cache data store on development environment using `docker/docker-compose.dev.yml` and `src/aie_estimator/.env.dev`:
  ```cmd
  auto\dc.cmd dev
  ```
- **Production (Containerized FastAPI + Redis Stack)**:
  Builds the multi-stage production image (`docker/Dockerfile`) and launches it along with the Redis cache data store on production environment using `docker/docker-compose.prod.yml` and `src/aie_estimator/.env.prod`:
  ```cmd
  auto\dc.cmd prod
  ```

### 2. Local Development Services

#### FastAPI Backend Service

Launch the FastAPI backend in the active console:

```cmd
auto\dev.cmd
```

Or directly via `uv`:

```bash
uv run aie-estimator
# or with auto-reload:
uv run uvicorn aie_estimator.server:server.app --reload --host localhost --port 8000
```

The API service starts on `http://localhost:8000`.

#### Streamlit Conversational Interface

With the FastAPI service running, launch the interactive chat interface in a separate terminal:

```cmd
auto\stl.cmd
```

Or directly via `uv`:

```bash
uv run streamlit run streamlit_app.py
```

The conversational interface opens in your browser at `http://localhost:8501`:

![Streamlit Conversational UI](docs/assets/session03_streamlit_ui.png)

## API Documentation & Endpoints

Interactive documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service status and version information |
| `GET` | `/health` | Health check endpoint returning API operational status |
| `POST` | `/api/v1/estimate` | Generates a project effort estimation from a meeting transcript |

---

### `POST /api/v1/estimate`

Analyzes meeting transcript text and returns an effort estimation breakdown.

#### Request Schema

```json
{
  "transcription": "Client requires a web platform for customer feedback collection with sentiment analysis and Slack notifications. Target release is 4 weeks."
}
```

#### Response Schema

```json
{
  "estimation": "## Estimation: Customer Feedback Platform\n\n### Task Breakdown:\n1. Backend API & Webhook Ingestion: 30 hours\n2. Sentiment Analysis Pipeline: 25 hours\n3. Slack Notification Integration: 15 hours\n4. Frontend Dashboard: 35 hours\n5. Testing & Deployment: 20 hours\n\n**Total Estimated Hours:** 125 hours\n**Recommended Team:** 2 Full-Stack Developers\n**Estimated Duration:** 3-4 weeks",
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

#### Example cURL

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "In the meeting with the marketing team, the client explained that they need a landing page with a contact form, integration with their existing CRM (HubSpot), and a blog section with a WYSIWYG editor. The target timeline is 4 weeks. Designs already exist in Figma."
  }'
```

---

### `GET /health`

Verifies that the service is running.

#### Example Response

```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

## Verification Checklist

Verify local setup against the baseline project criteria:

- [x] Application initializes with `uv run aie-estimator` or `uv run uvicorn aie_estimator.server:server.app --reload`
- [x] API keys and secrets load from `.env.<environment>` via `pydantic-settings` without hardcoded fallbacks
- [x] `GET /health` returns HTTP 200 with `status: healthy`
- [x] `POST /api/v1/estimate` processes transcriptions and returns structured estimations
- [x] Prompts include historical few-shot reference examples from `src/aie_estimator/context/examples.py`
- [x] Streamlit conversational UI accessible at `http://localhost:8501` via `streamlit run streamlit_app.py`
- [x] Redis exact-match cache short-circuits repeated estimation requests with offline fallback
- [x] Swagger documentation is accessible at `/docs`
- [x] `.env*` files are excluded in `.gitignore`

## Code Quality & Standards

- **Linter & Formatter**: Configured with `ruff` (line-length 140, Python 3.14 target).
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Test Suite**: Managed via `pytest`.
  ```bash
  uv run pytest
  ```
- **Pre-commit**: Git hooks configured via `.pre-commit-config.yaml`.
  ```bash
  uv run pre-commit run --all-files
  ```

## Documentation

Detailed architecture guides and technical documentation are maintained in the [`docs/`](docs/) directory:

- [LLM Service Architecture & Multi-Provider Integration](docs/llm-service.md): Deep dive into OpenAI Responses API, Anthropic Messages API, extended thinking, client lifecycle, retry backoff, and exception handling.
- [Model Catalog & Request Validation](docs/model-catalog.md): Centralized model catalog, supported OpenAI/Anthropic models, reasoning metadata, and boundary validation.
- [Conversational UI & Exact-Match Cache](docs/streamlit-conversational-ui.md): Streamlit chat interface, simulated progressive streaming, deterministic Redis exact-match caching, and graceful degradation.
- [Logging Architecture & Structlog](docs/logging.md): Configuration, formatting sinks, processor pipeline, and context variables binding.
- [Testing Guidelines](docs/testing.md): Test suite organization, pytest fixtures, and coverage expectations.

