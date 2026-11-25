# Observability: Structured Logging, Distributed Tracing, and LLM Telemetry

## Overview

The observability architecture in `aie-estimator` combines structured logging, standardized lifecycle events, and distributed tracing:

1. **Structured Logging**: Powered by [structlog](https://www.structlog.org/) and bridged to the Python standard library `logging` subsystem in [`../backend/src`](../backend/src/aie_estimator/logging.py), providing formatted ANSI console output for development and structured JSON output for production.
2. **Lifecycle Event Taxonomy**: Standardizes application and domain event names under the `<entity>_<action>_<phase>` convention.
3. **Distributed Tracing & LLM Observability**: Uses OpenTelemetry and OpenInference in [`../backend/src`](../backend/src/aie_estimator/telemetry.py) to instrument FastAPI and LiteLLM, exporting spans over gRPC (`:4317`) to an [Arize Phoenix](https://docs.arize.com/phoenix) container backed by embedded DuckDB.

---

## Usage Examples

### 1. Declaring a Module Logger and Emitting Events

```python
from typing import Final
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

# Standardized lifecycle events: started, completed, failed
log.info("estimation_generation_started", engine=opts.engine, model=opts.model)
log.info("estimation_generation_completed", latency_ms=elapsed_ms, cost_usd=cost, usage=usage)
```

### 2. Running Arize Phoenix via Docker Compose

```bash
docker compose -f docker/docker-compose.dev.yml up -d phoenix redis
```
Access the Phoenix web console at `http://localhost:6006` to inspect execution traces, latency distributions, and token usage.

### 3. Enabling OpenTelemetry Tracing

Enable tracing in `.env.dev` or the deployment environment:

```ini
ENABLE_OTEL_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## Technical Considerations

1. **Event Naming Taxonomy**: Event names use `<entity>_<action>_<phase>` snake_case format (e.g. `estimation_request_received`, `exact_cache_entry_retrieved`, `semantic_cache_lookup_evaluated`, `litellm_router_call_failed`, `session_turn_observed`). Consult individual modules in [`../backend/src`](../backend/src/aie_estimator/) for domain-specific event tokens.
2. **Unified Per-Turn Session Telemetry (`session_turn_observed`)**: [`EstimationService._finalize_session_turn`](../backend/src/aie_estimator/domain/estimations_service.py) emits a single `session_turn_observed` event (and stores the matching [`TurnObservation`](../backend/src/aie_estimator/generation/conversation/models.py) on `Session.last_turn_observed`) for both standard and ACB conversational turns, aggregating `turn_index`, `session_id`, `enriched_transcript_chars`, `attachments_total_chars`, `attachments_truncated_chars`, `messages_in_window`, `anchors_count`, `summary_chars`, full-turn `tokens_in` / `tokens_out` (Actor/Critic plus compression and metadata extraction), `cost_usd` (preserved as `None` whenever primary generation lacks pricing), full-turn `latency_ms`, `cache_hit_kind="none"`, and `last_resolved_tier`. `estimation_generation_completed` logs per-dispatch single-call latency, while `estimation_session_completed` and `estimation_acb_completed` emit lifecycle/governance attributes (`session_id`, `tier`, and for ACB `final_decision`, `iterations_run`) without duplicating per-turn telemetry.
3. **Trace-Log Correlation**: `inject_opentelemetry_context` in [`../backend/src`](../backend/src/aie_estimator/telemetry.py) injects active `trace_id` and `span_id` hex strings into structlog event dictionaries when tracing is active.
4. **Storage Architecture**: Arize Phoenix uses embedded DuckDB inside the container at `/root/.phoenix` (persisted to `./volumes/phoenix_data`), performing vectorized columnar queries over trace spans without requiring an external database cluster.
5. **Third-Party Log Suppression**: `configure_logging()` in [`../backend/src`](../backend/src/aie_estimator/logging.py) suppresses noisy third-party loggers (`asyncio`, `psycopg`, `uvicorn.error`) to `WARNING`, while preserving HTTP access logs in `uvicorn.access`.
6. **Lifecycle Isolation**: Telemetry initialization in [`../backend/src`](../backend/src/aie_estimator/telemetry.py) short-circuits immediately when `ENABLE_OTEL_TRACING=false`, eliminating background network polling and unmocked loopback socket overhead during testing.
7. **Offline Ingestion Tracing & PII Hygiene**: [`_run_ingestion_job`](../backend/src/aie_estimator/api/ingestion_router.py) opens an isolated `ingestion.run` OTel root span linked to the triggering HTTP `SpanContext` via `Link` and emits `<entity>_<action>_<phase>` events (`ingestion_job_started`, `ingestion_job_completed`, `ingestion_job_failed`, `ingestion_pii_completed`, `ingestion_pii_skipped`, `ingestion_pii_erasure_completed`, `ingestion_inspection_started`, `ingestion_inspection_completed`, `ingestion_inspection_failed`, `pseudonym_mapping_created`). Structured logs never emit `postgres_password` or plaintext PII.
8. **RAG Embedding Batch Telemetry**: [`OpenAIEmbedder`](../backend/src/aie_estimator/generation/rag/embedding/embedder.py) emits `embedding_batch_completed` (`chunks`, `tokens`, `latency_ms`, `model`) per API batch and `embedding_batch_rate_limited` (`attempt`, `retry_in_s`, `model`) when scheduling backoff retries, without logging raw input texts or API keys.
