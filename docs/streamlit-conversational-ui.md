# Structured Product Estimation Form and Exact-Match Caching

## Overview

[`streamlit_app.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/streamlit_app.py) provides a structured parameter product web interface for `aie-estimator`. It allows users to configure project dimensions (`description`, `project_type`, `detail_level`, `output_format`), validates character boundaries client-side, submits typed requests to `POST /api/v1/estimate`, and renders the structured [`EstimationResult`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py) (summary alert, metric cards for total cost, duration, confidence, and HTML phase table) alongside turn-level execution telemetry.

---

## Usage Examples

### 1. Launching the Services

In separate terminal sessions:

```bash
# Terminal 1: Start FastAPI estimation backend
uv run aie-estimator

# Terminal 2: Start Streamlit Product Form
uv run streamlit run streamlit_app.py
```
The interface is available at `http://localhost:8501`.

### 2. Form Submission and Telemetry

1. **Input Dimensions**: Enter description (20–2000 chars), select project type, detail level, and output format.
2. **Submit Estimation**: Click `Generate estimation` to dispatch the request.
3. **Structured Rendering**:
   - Status badge indicating active prompt version and Redis cache status.
   - Summary alert (`st.info`, transitioning to `st.warning` when `confidence_pct < 30`).
   - Metric cards displaying Total Cost (EUR), Total Duration (weeks), and Confidence (%).
   - Phase breakdown table displaying durations, phase costs, and scope summaries.
4. **Execution Telemetry**: Expandable container showing model name, upstream provider, execution cost in USD, and input/output token counts.

---

## Technical Considerations

1. **Decoupled Asynchronous Client**: [`streamlit_app.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/streamlit_app.py) communicates with the backend exclusively via `httpx.AsyncClient` targeting `POST /api/v1/estimate`. Domain orchestration and LLM provider logic remain strictly in the backend service.
2. **Session State Persistence**: The active response payload is stored in `st.session_state["latest_estimation"]`, preserving rendered estimations and telemetry across sidebar interactions and script reruns.
3. **Headless Test Isolation**: Page configuration and interactive UI rendering are wrapped in `if runtime.exists():`. This allows `streamlit_app.py` to be imported and tested headlessly via Streamlit's `AppTest.from_file()` without global singleton context leakage.
4. **Deterministic Cache Verification**: Repeated submissions with identical parameters return cached estimations from Redis in ~1–3 ms, updating the UI badge to indicate a cache hit with zero incremental token expenditure.
5. **Guardrail Error Handling**: Semantic guardrail rejections (HTTP 400 with `InputGuardrailViolation`) are intercepted and displayed as actionable security alerts (`st.error`) with the violation reason (`moderation`, `prompt_injection`, or `pii`), preserving entered form fields for easy revision.
