# CAG-RAG Pipeline Boundary

## Overview

The CAG-RAG pipeline boundary separates the latency-bound online estimation path from the throughput-oriented offline ingestion path under [`src/aie_estimator/ingestion/`](../src/aie_estimator/ingestion/__init__.py).

### 1. CAG-vs-RAG Architectural Decision

Grounded in [`evals/stress/REPORT.md`](../evals/stress/REPORT.md), pure Context-Augmented Generation (CAG) fails production latency and recall thresholds as corpus and conversation sizes grow (`61/65` turns exceeded `8,000 ms`, P50 reached `17,442 ms` at `>=10000` tokens, attachment truncation produced `0.0%` canary recall, and Turn-1 `ProjectMetadata` recall dropped to `0.0%` from Turn 2 onward in `pivot` and `contradiction`).

The architecture adopts a **hybrid CAG + RAG** target evaluated across four constraints and four axes without shipping dead diagnostic code inside `src/aie_estimator/`:

| Dimension | Assessment Summary | Architectural Consequence |
| :--- | :--- | :--- |
| **Viability Constraints** | Context window fit, cost per query (`$0.1149` / 65 turns), latency SLA (`61/65` > `8,000 ms`), and long-context recall (`0.0%` in `pivot`/`contradiction` and truncated attachments) | Latency and recall fail before token pricing; one failing constraint kills pure CAG |
| **Corpus Volume & Update Frequency** | Historical budgets, meeting transcripts, PDF contracts, and client rate cards grow linearly and change weekly/monthly | Exceeds safe static context margin and invalidates prefix caching |
| **Traceability & Access Control** | Estimates require precedent citations and client-confidential isolation | Pure CAG cannot cite chunk provenance or enforce per-user filtering prior to prompting |

### 2. Four-Bucket Context and Corpus Partition

| Bucket | Assets | Lifecycle |
| :--- | :--- | :--- |
| **Permanent static CAG** | `v3` tier templates ([`developer.j2`](../src/aie_estimator/prompts/estimation/v3/developer.j2), [`pm.j2`](../src/aie_estimator/prompts/estimation/v3/pm.j2), [`executive.j2`](../src/aie_estimator/prompts/estimation/v3/executive.j2)), [`TIER_CONFIG`](../src/aie_estimator/prompts/tier_config.py), `<output_schema>`, and [`_rules.j2`](../src/aie_estimator/prompts/estimation/v3/_rules.j2) | Stays permanently in [`prompts/loader.py`](../src/aie_estimator/prompts/loader.py) |
| **Session 7 complement/replacement candidates** | Baseline rates ([`_scope_rates.j2`](../src/aie_estimator/prompts/estimation/v3/_scope_rates.j2)) and calibration tables ([`_examples.j2`](../src/aie_estimator/prompts/estimation/v3/_examples.j2) -> [`v1/examples.j2`](../src/aie_estimator/prompts/estimation/v1/examples.j2)) | Retained in `v3` for Session 6; eligible for clean replacement or supplementation in Session 7 |
| **Online session memory** | [`SessionStore`](../src/aie_estimator/sessions/store.py) sliding window, [`CompressionPolicy`](../src/aie_estimator/sessions/compression/policy.py), [`_project_metadata.j2`](../src/aie_estimator/prompts/estimation/v3/_project_metadata.j2), [`_anchored_turns.j2`](../src/aie_estimator/prompts/estimation/v3/_anchored_turns.j2), [`_conversation_summary.j2`](../src/aie_estimator/prompts/estimation/v3/_conversation_summary.j2) | Per-session runtime state; outside both static CAG and offline RAG corpora |
| **Offline RAG corpus** | Historical budgets, meeting transcripts, signed PDF contracts/proposals, and client/versioned rate cards | Enters exclusively via [`src/aie_estimator/ingestion/`](../src/aie_estimator/ingestion/__init__.py) |

---

## Usage Examples

Target HTTP control contract for offline ingestion (to be wired once job persistence and orchestration land in Topics 2–4):

```http
POST /api/v1/ingestion/runs
Content-Type: application/json

{"source_name": "historical_budgets"}
```

Expected immediate dispatch response (`202 Accepted`) and polling endpoint:

```http
HTTP/1.1 202 Accepted
{"job_id": "<uuid>", "source_name": "historical_budgets", "status": "pending"}

GET /api/v1/ingestion/jobs/<uuid>
```

---

## Technical Considerations

1. **Async Event-Loop & Windows `ProactorEventLoop` Isolation**: All online routers in [`src/aie_estimator/routers/`](../src/aie_estimator/routers/__init__.py) are `async def` executing on Python 3.14's `ProactorEventLoop` on Windows. CPU-bound ingestion stages (PDF/DOCX parsing, tabular validation, NLP/PII detection) scheduled via FastAPI `BackgroundTasks` must run off the event loop—either via a synchronous `BackgroundTasks` target in AnyIO's worker threadpool or explicit `asyncio.to_thread(...)` calls—so concurrent `/api/v1/estimate` and `/health` requests never stall.
2. **OpenTelemetry Span Lifecycle & Structured Logging**: Under [`telemetry.py`](../src/aie_estimator/telemetry.py), `FastAPIInstrumentor` closes the `POST /api/v1/ingestion/runs` span when `202` is sent, before `BackgroundTasks` runs, and worker threads do not automatically inherit `contextvars`. Background ingestion runs must open their own root span (optionally linked to the triggering HTTP `SpanContext`), flush/shut down `TracerProvider` in [`Server.lifespan`](../src/aie_estimator/server.py), and emit lowercase snake_case `<entity>_<action>_<phase>` events (`ingestion_run_requested`, `ingestion_job_started`, `ingestion_job_completed`, `ingestion_job_failed`).
3. **Attachment Budget Baseline vs. Current Settings**: [`evals/stress/REPORT.md`](../evals/stress/REPORT.md) recorded truncation at `12,000` characters during the stress baseline run, whereas [`Settings`](../src/aie_estimator/settings.py) currently configures `max_attachment_chars = 60_000` and `max_total_attachment_chars = 60_000`.
4. **Clean-Stage Sequencing & Package Rules**: [`src/aie_estimator/ingestion/`](../src/aie_estimator/ingestion/__init__.py) is a peer domain package to `attachments/`, `guardrails/`, `prompts/`, and `sessions/`. Every `__init__.py` remains empty (module path comment only) unless an explicit facade is required, and no unbacked `/api/v1/ingestion/*` router stub or `CAGViability` module is introduced prior to Topic 2 persistence and Topic 4 orchestration.
5. **Dev Auto-Reload Caveat**: [`__main__.py`](../src/aie_estimator/__main__.py) enables Uvicorn `reload=True` outside `prod`; in-process `BackgroundTasks` interrupted by reload or shutdown require startup stale-job recovery in Topics 2–4.
