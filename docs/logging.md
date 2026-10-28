# Logging Guidelines and Structlog Configuration

## Overview

This document outlines the logging architecture, conventions, and configuration options used across the `aie-estimator` application.

The application uses [structlog](https://www.structlog.org/) integrated with the Python standard library `logging` framework via `structlog.stdlib.ProcessorFormatter`. This bridge enables both direct `structlog` loggers and standard library loggers (such as Uvicorn HTTP access logs and FastAPI framework logs) to flow through a unified processor pipeline and emit consistent output across all destinations.

### Destination Sinks and Formatters

| Environment | Flag | Console Handler (stdout) | File Handler (`logs/`) |
| :--- | :--- | :--- | :--- |
| **Development** | `JSON_LOGGING=false` (default) | `ConsoleRenderer(colors=True)` with ANSI colors | `ConsoleRenderer(colors=False)` (clean plain text, no ANSI codes) |
| **Development** | `JSON_LOGGING=true` | `JSONRenderer()` | `JSONRenderer()` |
| **Production** | Any | `JSONRenderer()` | `JSONRenderer()` |
| **Test** | `JSON_LOGGING=false` | `ConsoleRenderer(colors=True)` | `ConsoleRenderer(colors=False)` |

---

## Usage Examples

### 1. Declaring a Module Logger

Declare a module-level logger named `log` typed with `Final[structlog.stdlib.BoundLogger]` at the top of the module:

```python
# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)
```

### 2. Emitting Structured Events

Pass a short, snake_case event name as the first argument, followed by contextual key-value pairs. Avoid manual string formatting or f-strings in event names:

```python
# Informational event with structured metadata
log.info(
    "dispatching_estimation_request",
    provider="openai",
    model=settings.llm_model,
    transcript_length=len(transcription),
)

# Warning event
if opts.thinking_budget is not None:
    log.warning("thinking_budget_ignored_for_provider", provider="openai")

# Exception logging with traceback
try:
    result = llm_service.generate_estimation(transcription)
except Exception as e:
    log.exception("estimation_generation_failed", error=str(e), exc_info=e)
    raise
```

### 3. Context Variables Binding

Context variables bound via `structlog.contextvars` automatically attach to all subsequent log events within the same asynchronous task:

```python
import structlog

structlog.contextvars.bind_contextvars(request_id="req-12345")
log.info("processing_request")  # Automatically includes request_id="req-12345"
```

### 4. Environment Configuration

Logging behavior is configured via environment variables defined in `.env.<environment>`:

```ini
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Force JSON logging in development to verify production format locally: true / false
JSON_LOGGING=false

# Write logs to rotating file on disk: true / false
ENABLE_FILE_LOGGING=true
```

---

## Technical Considerations

### 1. Standard Library Bridge and Uvicorn Hand-off

Uvicorn defaults to overwriting the Python root logger configuration with its internal `LOGGING_CONFIG` dictionary. In `src/aie_estimator/__main__.py`, `uvicorn.run(...)` is invoked with:

```python
uvicorn.run(
    ...,
    log_config=None,
)
```

Passing `log_config=None` prevents Uvicorn from clearing existing handlers. Standard library log records emitted by Uvicorn (`uvicorn`, `uvicorn.access`, `uvicorn.error`) route into the root logger's `StreamHandler` and `RotatingFileHandler`, where `structlog.stdlib.ProcessorFormatter` formats them using the active structlog renderer.

### 2. Renderer Selection Logic

Renderer selection is evaluated dynamically in `Settings.configure_logging()`:

```python
use_json = self.json_logging or (self.app_env == Environment.PROD)
```

When `use_json` is `True`, both console and file formatters instantiate `structlog.processors.JSONRenderer()`. This enables local validation of JSON log structures before deploying to production.

### 3. Clean File Output for JetBrains IDEs

During development, `ConsoleRenderer(colors=True)` sends ANSI escape sequences to the terminal. If ANSI codes are written to `logs/dev.log`, viewing log files inside JetBrains IDEs (PyCharm / WebStorm / DataGrip) displays raw escape codes (`\033[32m`, etc.). To prevent log corruption on disk, the file handler explicitly uses `ConsoleRenderer(colors=False)`.

### 4. Logger Cache Dynamics Across Environments

In production, `cache_logger_on_first_use=True` caches bound loggers for maximum throughput. In development and test environments, `cache_logger_on_first_use=False` is configured so that test fixtures reloading settings via `get_settings.cache_clear()` immediately propagate new log configurations to existing and new loggers.

### 5. Third-Party Namespace Suppression

Noisy third-party loggers (`asyncio`, `psycopg`) are explicitly pinned to `logging.WARNING` in `Settings.configure_logging()` to keep development and production logs signal-focused.

### 6. Callsite Location Injection

To maintain location traceability during debugging and production error triage, `shared_processors` includes `structlog.processors.CallsiteParameterAdder`:

```python
CallsiteParameterAdder(
    {
        CallsiteParameter.FILENAME,
        CallsiteParameter.LINENO,
        CallsiteParameter.FUNC_NAME,
    }
)
```

This injects `filename`, `lineno`, and `func_name` into every log event across both direct structlog calls and standard library records bridged by `ProcessorFormatter`.

### 7. Test Isolation and Mocked File Logging

In automated unit tests (`pytest`), tests must never write files to disk. Tests verifying file handler configuration (such as renderer selection and argument verification in `test_settings.py`) patch `logging.handlers.RotatingFileHandler` with `autospec=True` and `pathlib.Path.mkdir`. This validates that:
- Target filenames (`app.log` in PROD, `dev.log` in DEV), `maxBytes`, `backupCount`, and `encoding` are configured correctly.
- `structlog.stdlib.ProcessorFormatter` is assigned with the appropriate renderer.
- Zero files are written to the workspace during test runs, preventing concurrency conflicts and CI environment pollution.

Writing to `logs/dev.log` for full stack error inspection is reserved exclusively for local application runtime, enabled via `ENABLE_FILE_LOGGING=true` in `.env.development`.


