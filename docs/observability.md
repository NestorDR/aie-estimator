# Observability: Structured Logging, Distributed Tracing, and LLM Telemetry

## Overview

The observability architecture in `aie-estimator` combines structured logging, standardized lifecycle events, and distributed tracing:

1. **Structured Logging**: Powered by [structlog](https://www.structlog.org/) and bridged to the Python standard library `logging` subsystem in [`src/aie_estimator/logging.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/logging.py), providing formatted ANSI console output for development and structured JSON output for production.
2. **Lifecycle Event Taxonomy**: Standardizes application and domain event names under the `<entity>_<action>_<phase>` convention.
3. **Distributed Tracing & LLM Observability**: Uses OpenTelemetry and OpenInference in [`src/aie_estimator/telemetry.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/telemetry.py) to instrument FastAPI and LiteLLM, exporting spans over gRPC (`:4317`) to an [Arize Phoenix](https://docs.arize.com/phoenix) container backed by embedded DuckDB.

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

1. **Event Naming Taxonomy**: Event names use `<entity>_<action>_<phase>` snake_case format (e.g. `estimation_request_received`, `exact_cache_entry_retrieved`, `semantic_cache_lookup_evaluated`, `litellm_router_call_failed`). Consult individual modules in [`src/aie_estimator/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/) for domain-specific event tokens.
2. **Trace-Log Correlation**: `inject_opentelemetry_context` in [`src/aie_estimator/telemetry.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/telemetry.py) injects active `trace_id` and `span_id` hex strings into structlog event dictionaries when tracing is active.
3. **Storage Architecture**: Arize Phoenix uses embedded DuckDB inside the container at `/root/.phoenix` (persisted to `./volumes/phoenix_data`), performing vectorized columnar queries over trace spans without requiring an external database cluster.
4. **Third-Party Log Suppression**: `configure_logging()` in [`src/aie_estimator/logging.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/logging.py) suppresses noisy third-party loggers (`asyncio`, `psycopg`, `uvicorn.error`) to `WARNING`, while preserving HTTP access logs in `uvicorn.access`.
5. **Lifecycle Isolation**: Telemetry initialization in [`src/aie_estimator/telemetry.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/telemetry.py) short-circuits immediately when `ENABLE_OTEL_TRACING=false`, eliminating background network polling and unmocked loopback socket overhead during testing.
