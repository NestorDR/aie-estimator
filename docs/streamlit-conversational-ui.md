# Conversational Estimation UI

## Overview

[`streamlit_app.py`](../streamlit_app.py) provides a session-only conversational web interface for `aie-estimator` backed by [`sessions_router`](../src/aie_estimator/routers/sessions_router.py) (`/api/v1/sessions`). Users submit multi-turn project descriptions and supporting documents (`.pdf`, `.docx`, `.txt`, `.md`) through `st.chat_input`, configure execution engine (`litellm` or `native`), engine-filtered target model, stakeholder tier (`Auto` or explicit `developer`/`pm`/`executive` override), optional Actor-Critic-Boss review, and estimation dimensions in the sidebar, and inspect live [`SessionInfoResponse`](../src/aie_estimator/schemas/sessions.py) window counts, `last_tier`, and distilled [`ProjectMetadata`](../src/aie_estimator/sessions/models.py) alongside tier-adapted results, optional `BossTrace`, and execution telemetry.

---

## Usage Examples

### 1. Launching the Services

In separate terminal sessions:

```bash
# Terminal 1: Start FastAPI estimation backend
uv run aie-estimator

# Terminal 2: Start Streamlit conversational UI
uv run streamlit run streamlit_app.py
```

The interface is available at `http://localhost:8501`.

### 2. Multi-Turn Estimation and Attachment Flow

1. **Configure Parameters (Sidebar)**: Choose `Execution engine` (`litellm` or `native`), `Target model` (filtered via [`get_models_for_engine`](../src/aie_estimator/model_catalog.py)), `Stakeholder tier` (`Auto`, `Developer`, `Pm`, or `Executive`), optional `Actor-Critic-Boss review` toggle (default off), and estimation dimensions (`Project type`, `Detail level`, `Output format`), plus up to 5 supported attachments (`.pdf`, `.docx`, `.txt`, `.md`).
2. **Submit Turn (`st.chat_input`)**: Enter a 20–2000 character transcript in the chat input; attached files are submitted with the turn and cleared automatically afterward.
3. **Inspect Output & Telemetry**: Each assistant turn renders the active `prompt_version`, resolved `tier` and `tier_rule`, summary alert (`st.info`, or `st.warning` when `confidence_pct < 30` or prefixed with `"Out of scope:"`), tier-specific body (`DeveloperEstimation`, `PmEstimation`, or `ExecutiveEstimation`), collapsed `Boss audit trail` expander (when ACB is enabled), and `Execution telemetry` panel.
4. **Monitor Memory & Reset**: Inspect live sliding-window counts (`message_count / max_turns * 2`), `last_tier`, and extracted `ProjectMetadata` in the sidebar `Session state` section, or click `New conversation` to reset the client session.

---

## Technical Considerations

1. **Lazy Session Lifecycle & 404 Recovery**: `st.session_state.session_id` starts as `None` and is provisioned lazily via [`create_session`](../streamlit_app.py) on the first valid turn. If [`send_session_turn`](../streamlit_app.py) receives HTTP `404` (backend restart or TTL expiration), the client creates a replacement session, clears the prior thread, displays a warning, and retries the turn once.
2. **Client-Side Thread vs. Backend Window**: Because `GET /api/v1/sessions/{session_id}` returns counts and `ProjectMetadata` without message bodies, the UI retains the full turn list in `st.session_state.thread` across reruns and displays a caption stating the backend `max_turns` sliding-window limit.
3. **Multipart Client & Shared Extension Source**: [`send_session_turn`](../streamlit_app.py) dispatches `multipart/form-data` requests via `httpx.AsyncClient`, and the sidebar `st.file_uploader(accept_multiple_files=True)` derives allowed extensions directly from [`Settings.supported_attachment_extensions`](../src/aie_estimator/settings.py), resetting via `upload_nonce` after each turn with attachments.
4. **Unified HTTP Error Mapping**: [`format_http_error`](../streamlit_app.py) translates backend status codes into structured alerts: `400` input guardrail violations (`moderation`, `prompt_injection`, `pii`), `415` unsupported attachment extensions, `422` attachment extraction or validation errors, and `500`/other failures.
5. **Headless Test Isolation**: Page configuration and interactive UI rendering are guarded by `if runtime.exists():`, allowing [`test_streamlit_app.py`](../tests/test_streamlit_app.py) to test pure client/error helpers with `pytest` and UI flows with Streamlit's `AppTest.from_file()`.
6. **Tier Override & `_TIER_RENDERERS` Dispatch**: When `Stakeholder tier` is `Auto`, the UI omits `tier` so the backend resolves it; explicit selections send the `tier` form field and never `X-User-Tier` (reserved for trusted gateways). [`render_estimation`](../streamlit_app.py) renders shared header/summary/telemetry once and dispatches the body via `_TIER_RENDERERS` (defaulting to `_render_pm_body`).
7. **ACB Routing & Audit Trail**: Enabling `Actor-Critic-Boss review` routes [`send_session_turn`](../streamlit_app.py) to `/estimate-acb` with the same form fields and 404 recovery. [`render_boss_trace`](../streamlit_app.py) displays `ACBIteration.issue_summary` rather than raw `CriticIssue` objects, matching the `BossTrace` API contract.
