# Testing Guidelines and Shared Fixtures

## Overview

Testing architecture, shared fixtures, and execution conventions for the `aie-estimator` test suite.

### Structure and Configuration

- **Test Directory**: `tests/` (`conftest.py`, `test_settings.py`, `test_models_catalog.py`, `test_streamlit_app.py`, `routers/`, `services/`)
- **Import Resolution**: `tool.pytest.ini_options.pythonpath = [".", "src"]` in `pyproject.toml`
- **Execution Target**: `pytest tests/ --cache-clear --basetemp="%TEMP%\pytest_runner"`

### Global Fixtures (`tests/conftest.py`)

| Fixture | Scope | Autouse | Description |
| :--- | :--- | :--- | :--- |
| `clean_settings_cache` | function | Yes | Clears `get_settings.cache_clear()` before and after each test for state isolation. |
| `test_settings` | function | No | Returns an isolated `Settings` instance configured for `Environment.TEST`. |
| `app` | function | No | Yields `Server().app` and resets `dependency_overrides.clear()` on teardown. |
| `client` | function | No | Wraps `app` in `TestClient(app)` lifespan context manager for startup/shutdown events. |
| `mock_openai_response` | function | No | Returns a `MagicMock` matching OpenAI Responses schema (`status`, `output_text`, `model`). |
| `sample_transcription` | function | No | Multi-line transcript string for estimation tests. |
| `sample_estimation_payload` | function | No | Dictionary matching `EstimationRequest` schema (`{"transcription": ...}`). |

---

## Usage Examples

### 1. Parametric API Testing (`app`, `client`, `ids`)

Deduplicate symmetric test variants into a single parameterized function using explicit `ids`:

```python
# tests/routers/test_health_router.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.settings import Settings, get_settings


@pytest.mark.parametrize(
    ("override_version", "expected_version"),
    [
        (None, None),
        ("9.9.9-test", "9.9.9-test"),
    ],
    ids=["default_settings", "dependency_override"],
)
def test_get_health(
    app: FastAPI,
    client: TestClient,
    override_version: str | None,
    expected_version: str | None,
) -> None:
    """
    GIVEN an application instance and an optional settings dependency override
    WHEN a GET request is sent to the /health endpoint
    THEN a 200 OK status is returned with status healthy and the matching version.

    :param app: FastAPI application fixture.
    :param client: Shared TestClient fixture.
    :param override_version: Custom version string for dependency override, or None for default.
    :param expected_version: Expected version string in response body, or None to assert present.
    """
    if override_version is not None:
        custom_settings = Settings()
        custom_settings.app_version = override_version
        app.dependency_overrides[get_settings] = lambda: custom_settings

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    if expected_version is not None:
        assert data["version"] == expected_version
    else:
        assert "version" in data
```

### 2. Concise Exception Assertions (`match=r'...'`)

Target concise keywords rather than literal error messages with `pytest.raises`:

```python
# tests/test_settings.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.settings import Settings


@pytest.mark.parametrize(
    ("kwargs", "expected_exception", "match_pattern"),
    [
        ({"app_env": "invalid_environment_name"}, ValueError, r"is not a valid Environment"),
        ({"server_port": "invalid_port"}, ValidationError, r"valid integer"),
    ],
    ids=["invalid_env_string", "invalid_port_type"],
)
def test_settings_initialization_validation_errors(
    kwargs: dict[str, Any],
    expected_exception: type[Exception],
    match_pattern: str,
) -> None:
    """
    GIVEN invalid configuration arguments
    WHEN Settings is instantiated
    THEN the expected exception is raised matching concise keywords.

    :param kwargs: Keyword arguments to pass to Settings constructor.
    :param expected_exception: Expected exception class.
    :param match_pattern: Concise regex pattern targeting distinctive error keywords.
    """
    with pytest.raises(expected_exception, match=match_pattern):
        Settings(**kwargs)
```

### 3. Isolated Environment Overrides

`clean_settings_cache` isolates `monkeypatch` modifications across test runs:

```python
# tests/test_settings.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.settings import Environment, get_settings


def test_get_settings_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an environment variable override for APP_ENV
    WHEN get_settings.cache_clear() is executed and get_settings is called
    THEN a reloaded Settings instance reflecting the new environment is returned.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()

    reloaded_settings = get_settings()
    assert reloaded_settings.app_env == Environment.TEST
```

### 4. Mocking External LLM Invocations

Mock LLM provider calls using `mock_openai_response` and `sample_estimation_payload`:

```python
# tests/routers/test_estimations_router.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
from fastapi.testclient import TestClient

# --- App modules ---


def test_create_estimation(
    client: TestClient,
    sample_estimation_payload: dict[str, Any],
    mock_openai_response: MagicMock,
) -> None:
    """
    GIVEN an active client and mocked OpenAI completion response
    WHEN a POST request is sent to /api/v1/estimate with a transcription payload
    THEN a 200 OK status is returned containing estimation data.

    :param client: Shared TestClient fixture.
    :param sample_estimation_payload: Request payload dictionary matching schema.
    :param mock_openai_response: Preconfigured MagicMock simulating LLM response.
    """
    with patch("aie_estimator.services.llm_service._call_openai", return_value=mock_openai_response):
        response = client.post("/api/v1/estimate", json=sample_estimation_payload)
        assert response.status_code == 200
        assert "estimation" in response.json()
```

### 5. Mocking File Logging Handlers

When validating logging configurations with file logging enabled, patch `RotatingFileHandler` with `autospec=True` and `pathlib.Path.mkdir` to prevent disk writes:

```python
# tests/test_settings.py

# --- Python Standard Library ---
import logging
import logging.handlers
from unittest.mock import patch

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.settings import Environment, Settings


def test_file_handler_mocking() -> None:
    """
    GIVEN Settings with enable_file_logging=True
    WHEN configure_logging is executed under mocked handlers
    THEN RotatingFileHandler is instantiated with valid parameters without writing to disk.
    """
    with (
        patch("logging.handlers.RotatingFileHandler", autospec=True) as mock_file_handler_cls,
        patch("pathlib.Path.mkdir"),
    ):
        mock_file_handler = mock_file_handler_cls.return_value
        mock_file_handler.level = logging.NOTSET

        try:
            settings = Settings(app_env=Environment.DEV, enable_file_logging=True)
            settings.configure_logging()

            mock_file_handler_cls.assert_called()
            assert mock_file_handler_cls.call_args.kwargs["filename"].name == "dev.log"
        finally:
            logging.getLogger().handlers.clear()
```

### 6. CLI Execution Patterns

```bash
# Run entire test suite
pytest --cache-clear --basetemp="%TEMP%\pytest_runner"

# Run single test module
pytest tests/routers/test_health_router.py

# Target specific parameterized case by id
pytest -k "dependency_override"
pytest tests/routers/test_health_router.py::test_get_health[dependency_override]
```

---

## Technical Considerations

### 1. Test Design Rules

- **GIVEN-WHEN-THEN**: Every test function docstring must explicitly follow GIVEN (preconditions), WHEN (action), THEN (assertions) followed by Sphinx `:param` definitions.
- **Parametric Deduplication**: Consolidate symmetric branches with `@pytest.mark.parametrize`. Always specify `ids` to prevent unreadable `[None-None]` auto-generated identifiers, enable `-k` filtering, and clarify CI failure diagnostics.
- **Concise Exception Regexes**: Assert exceptions with `pytest.raises(Exc, match=r'...')` using minimal distinctive keywords (e.g., `match=r'valid integer'`) to prevent brittleness against minor formatting revisions.
- **Type Annotations**: All fixtures, test parameters, and return types require explicit type hints (`client: TestClient`, `-> None`).

### 2. Runtime Isolation Mechanics

- **Lifespan Execution**: The `client` fixture uses `with TestClient(app)` to ensure FastAPI async startup/shutdown lifespan context handlers execute.
- **Cache Isolation**: The autouse `clean_settings_cache` fixture clears `@lru_cache` on `get_settings` before and after each test, preventing cross-test settings contamination.
- **Override Teardown**: The `app` fixture invokes `app.dependency_overrides.clear()` on fixture teardown, eliminating manual `finally` cleanup in individual tests.
- **Path Resolution**: `pyproject.toml` configures `pythonpath = ["src"]`, enabling standard absolute imports without requiring editable package installs.

### 3. Filesystem Isolation and Zero File Writes

- **Zero Disk Writes**: Unit tests executed by `pytest` must never create files or directories within the workspace (`logs/`, etc.). Any file handler initialization must be intercepted with `unittest.mock.patch(..., autospec=True)`.
- **Runtime File Logging**: Generating `logs/dev.log` for debugging and inspecting full stack error traces is an application runtime feature, activated via `ENABLE_FILE_LOGGING=true` in `src/aie_estimator/.env.development`, never by test fixtures.
- **Handler Cleanup**: Tests adding mocked handlers to standard library loggers must clear `logging.getLogger().handlers.clear()` in a `finally` block to prevent mock leaks into subsequent test cases.

