# Structured Product Estimation Form and Exact-Match Caching

## Overview

This document specifies the structured product parameter estimation web interface and exact-match response caching layer implemented in `aie-estimator`.

The web interface is powered by Streamlit (`streamlit_app.py`) and transitions the client presentation layer from legacy conversational chat feeds into a structured parameter product interface. It enables software engineers to configure multi-dimensional project parameters (`description`, `project_type`, `detail_level`, `output_format`), validate inputs client-side, submit typed `EstimationRequest` payloads to `POST /api/v1/estimate`, observe structured rendering of the generated `EstimationResult` (summary alert, metric cards for total cost, duration, and confidence, and phase breakdown table) and active prompt template version, and inspect turn-anchored execution telemetry in an expandable container.

Complementing the UI, an exact-match caching layer (`src/aie_estimator/services/cache.py`) backed by Redis short-circuits repeated LLM invocations by returning cached estimations with low latency (~1–3 ms) when identical generation parameters and prompt contents are detected.

---

## Usage Examples

### 1. Launching the Services

#### Option A: Windows Automation Scripts

Start the detached Redis container, the FastAPI backend, and the Streamlit interface from the project root:

```cmd
:: Start the Redis cache container (estimator-redis)
auto\dc.cmd dev

:: Terminal 1: Start FastAPI backend in the current console
auto\dev.cmd

:: Terminal 2: Start Streamlit Product Form in the current console
auto\stl.cmd
```

#### Option B: Direct CLI Commands

Start the FastAPI estimation backend and the Streamlit interface in separate terminals via `uv`:

```bash
# Terminal 1: Start FastAPI backend
uv run aie-estimator
# or: uv run uvicorn aie_estimator.server:server.app --reload --host localhost --port 8000

# Terminal 2: Start Streamlit Product Form
uv run streamlit run streamlit_app.py
```

Streamlit automatically opens the application at `http://localhost:8501`.

### 2. Interactive Structured Form Submission

1. **Enter Project Description**: Input between 20 and 2000 characters in the `st.text_area` describing goals, key features, architecture, and constraints.
2. **Select Domain and Formatting Dimensions**:
   - **Project type**: Dropdown (`st.selectbox`) selecting among `web_saas`, `mobile_app`, `internal_tool`, or `data_pipeline`.
   - **Detail level**: Segmented control (`st.segmented_control`) selecting among `summary`, `medium`, or `detailed`.
   - **Output format**: Dropdown (`st.selectbox`) selecting among `phases_table`, `line_items`, or `narrative`.
3. **Submit Estimation**: Click the primary submit button (`Generate estimation`).
4. **View Rendered Structured Estimation**:
   - The active prompt template version is displayed prominently above the estimation, accompanied by a `Cached` indicator badge when served from Redis (`Prompt version: v1 :material/bolt: Cached`).
   - The project summary is rendered inside an informative alert (`st.info`), automatically transitioning to a warning indicator (`st.warning`) if confidence is below 30% or if prefixed with `"Out of scope:"`.
   - Primary estimation metrics render across three responsive columns: **Total cost** in EUR (`€XX,XXX`), **Total duration** in weeks (`X weeks`), and **Confidence** percentage (`XX%`).
   - The project phase breakdown is presented in a clean tabular view detailing phase names, durations in weeks, EUR costs, and scope summaries.

![Streamlit Product Estimation Form](assets/session04_streamlit_ui_1_of_2.png)
![Streamlit Product Estimation Telemetry and Output](assets/session04_streamlit_ui_2_of_2.png)

### 3. Client-Side Input Validation

The interface validates the project description bounds client-side prior to network dispatch:
- Descriptions under 20 characters display an immediate error alert: `The project description must be at least 20 characters long.` with the `:material/error:` icon.
- Descriptions exceeding 2000 characters display an immediate error alert: `The project description cannot exceed 2000 characters.` with the `:material/error:` icon.
- Invalid submissions suppress HTTP requests to the backend entirely.

### 4. Generation Parameters and Model Overrides

The Streamlit sidebar exposes real-time service telemetry and execution controls:
- **Service Status**:
  - Displays the active API endpoint address (`estimate_endpoint`).
  - Displays the primary configured model (`settings.llm_model`), fallback model, and cache TTL in seconds.
- **Model and Engine Overrides**:
  - **Target model**: Dropdown menu populated dynamically via `get_supported_models()` from the catalog, pre-selecting the primary model default.
  - **Execution engine**: Segmented control switching between `litellm` (multi-provider cost routing) and `native` (direct provider SDK dispatch).
- Selections made in the sidebar are automatically passed in the `model` and `engine` fields of the dispatched `EstimationRequest` payload.

### 5. Inspecting Execution Telemetry

Directly beneath each rendered estimation, an expandable telemetry card (`:material/insights: Execution telemetry`, collapsed by default) provides granular operational insights:
- **Model**: Physical model or routing profile utilized for inference.
- **Provider**: Upstream LLM provider (`openrouter`, `openai`, `anthropic`, `gemini`, etc.).
- **Cost**: Total monetary cost of execution in USD formatted to 5 decimal places (e.g. `$0.00350`).
- **Token Usage**: Token breakdown showing input tokens, output tokens, total tokens, and preprocessing tokens (when Phase-1 requirements extraction is executed).

### 6. Cache Hit Verification

When identical estimation parameters and description text are submitted again:
- The backend returns the cached estimation directly from Redis.
- The UI displays the cached response with low latency (~1–3 ms) and zero incremental token expenditure.
- The prompt version banner highlights a `Cached` indicator badge.

---

## Technical Considerations

### 1. Decoupled Asynchronous HTTP Client

`streamlit_app.py` acts strictly as an HTTP client communicating with `POST /api/v1/estimate` using `httpx.AsyncClient` wrapped in `asyncio.run()`. The backend router contract and Pydantic validation schemas remain decoupled from Streamlit presentation concerns:

```python
async def send_estimation_request(
    description: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    *,
    model: str | None = None,
    engine: str = "litellm",
    endpoint: str = estimate_endpoint,
) -> dict[str, Any]:
    ...
```

### 2. Structured Estimation Result Rendering

The legacy free-text markdown renderer has been replaced by structured `EstimationResult` formatting:
- Project summary is presented with conditional styling: warning alert (`st.warning`) for low-confidence or out-of-scope estimations, and informational alert (`st.info`) for standard estimations.
- High-level KPIs are rendered via native `st.metric` widgets (`Total cost`, `Total duration`, and `Confidence`).
- Phase-level allocations are rendered through an interactive `st.dataframe` detailing Phase name, duration, EUR cost, and summary.
- If legacy text responses are encountered, the interface gracefully falls back to `st.markdown`.

### 3. Session State Persistence

The active estimation response dictionary is cached in `st.session_state["latest_estimation"]`. When a user toggles sidebar settings or interacts with secondary widgets, the rendered Markdown estimation, prompt version tag, and execution telemetry persist across script rerun cycles without requiring form re-submission.

### 4. Runtime Isolation for Testability

UI elements and page configuration are guarded with `if runtime.exists():`:

```python
if runtime.exists():
    st.set_page_config(...)
    ...
```

This prevents bare-mode execution when `streamlit_app.py` is imported during unit test collection, protecting Streamlit's singleton `DeltaGenerator` stack from form context leakage while ensuring seamless execution under both `streamlit run` and headless `AppTest.from_file()`.

### 5. Deterministic SHA-256 Cache Keying

Cache keys are computed using a SHA-256 hash across all input parameters that affect output generation:

```python
payload = json.dumps(
    {
        "description": description,
        "detail_level": detail_level,
        "engine": engine,
        "model": model,
        "output_format": output_format,
        "project_type": project_type,
    },
    sort_keys=True,
)
cache_key = f"estimation:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
```

Any modification to prompt templates or generation parameters yields a distinct hash, guaranteeing cache consistency without manual invalidation.

### 6. Resilience and Graceful Degradation

Redis operations in `EstimationCache` are non-blocking asynchronous coroutines implemented with `redis.asyncio.Redis` and encapsulated in try/except blocks catching `redis.exceptions.RedisError`. If Redis is offline, unconfigured, or unreachable:
- Asynchronous cache lookups (`await cache.get(key)`) log a warning and return `None` (cache miss).
- Asynchronous cache storage attempts (`await cache.set(key, ...)`) log a warning and return silently.
- Estimation requests complete normally through direct LLM calls without throwing HTTP 500 errors.
- On shutdown, `await cache.close()` cleanly closes the connection pool.

### 7. Environment Configuration & Settings

The interface and cache layer read settings from `.env.<environment>` via [src/aie_estimator/settings.py](../src/aie_estimator/settings.py):
- `ESTIMATOR_API_BASE_URL`: Streamlit backend target address (default: `http://localhost:8000`).
- `REDIS_HOST`: Redis server host (`localhost` for host development, `redis` inside Docker Compose).
- `REDIS_PORT`: Redis server port (default: `6379`).
- `CACHE_TTL`: Cache entry time-to-live in seconds (default: `86400`).
