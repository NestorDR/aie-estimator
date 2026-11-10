# Testing Guidelines and Shared Fixtures

## Overview

[`pyproject.toml`](file:///c:/Development/Repos/ai-engineering/aie-estimator/pyproject.toml) configures pytest for `aie-estimator` (`testpaths = ["tests"]`, `pythonpath = [".", "src"]`). Test modules mirror the source tree under [`tests/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/tests/).

Core shared fixtures are defined in [`tests/conftest.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/tests/conftest.py):
- `app`: Instantiates the FastAPI application via `create_app()` and clears dependency overrides on teardown.
- `client`: `TestClient` executing application startup and shutdown lifespan handlers.
- `clean_settings_cache` (autouse): Clears application `@lru_cache` singletons before and after each test.
- `auto_sync_settings_on_env_mutation` (autouse): Flushes `get_settings` and dependent singletons whenever environment variables are mutated via `monkeypatch`.
- `default_tiered_cache_isolation` (autouse): Sets Tier 1 exact cache and Tier 2 semantic cache to default miss states across unit and router tests.

---

## Usage Examples

### 1. Parametrization with Explicit `ids`

```python
@pytest.mark.parametrize(
    ("override_version", "expected_version"),
    [(None, None), ("9.9.9-test", "9.9.9-test")],
    ids=["default_settings", "dependency_override"],
)
def test_get_health(app: FastAPI, client: TestClient, override_version: str | None, expected_version: str | None) -> None:
    """
    GIVEN an application instance and an optional settings dependency override
    WHEN a GET request is sent to the /health endpoint
    THEN a 200 OK status is returned with status healthy and the matching version.

    :param app: FastAPI application fixture.
    :param client: Shared TestClient fixture.
    :param override_version: Custom version string for dependency override, or None for default.
    :param expected_version: Expected version string in response body, or None to assert present.
    """
```

### 2. Concise Exception Assertions

```python
with pytest.raises(ValidationError, match=r"valid integer"):
    Settings(server_port="invalid_port")
```

### 3. Environment Overrides

```python
def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    settings = Settings()
    assert settings.app_env == Environment.TEST
```

### 4. Mocking at the Boundary via Dependency Injection

```python
from unittest.mock import AsyncMock
from aie_estimator.dependencies import get_estimation_service
from aie_estimator.services.estimations_service import EstimationService

mock_service = AsyncMock(spec=EstimationService)
mock_service.generate_estimation.return_value = sample_response
app.dependency_overrides[get_estimation_service] = lambda: mock_service
```

### 5. Async Tests with AnyIO

```python
@pytest.mark.anyio
async def test_cache_miss_returns_none() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    assert await EstimationExactCache(mock_redis).get("estimation:missing") is None
```

### 6. CLI Execution

```powershell
rtk uv run pytest --cov-fail-under=85   # Full regression suite with coverage gate
rtk uv run pytest tests/routers/test_health_router.py  # Single test file with local coverage
rtk uv run pytest --no-cov             # Zero-overhead iterative feedback
```

---

## Technical Considerations

1. **Test Design Conventions**:
   - Every test docstring follows GIVEN-WHEN-THEN structure with Sphinx `:param` definitions.
   - Assert exceptions with `pytest.raises(Exc, match=r'...')` using concise, distinctive keywords.
   - Tests never make real network or I/O calls to LLM providers or Redis.
2. **Runtime Isolation Mechanics**:
   - `clean_settings_cache` and `auto_sync_settings_on_env_mutation` in [`tests/conftest.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/tests/conftest.py) ensure zero state leakage between tests.
   - `app` fixture neutralizes `EstimationSemanticCache.initialize_index` so lifespan startup does not block on physical Redis sockets in offline environments.
3. **Environment Isolation**: The suite runs under `APP_ENV=test` configured in `[tool.pytest_env]` in [`pyproject.toml`](file:///c:/Development/Repos/ai-engineering/aie-estimator/pyproject.toml). Tests must never produce `logs/dev.log`.
4. **Coverage Policy**: Minimum 85% line and branch coverage across `aie_estimator` and `streamlit_app`. `--cov-fail-under` is enforced via CLI during full regression runs and must not be placed in `pyproject.toml` `addopts`.
5. **Windows 11 & Python 3.14 Async Execution Policy**: Python 3.14's `ProactorEventLoop` uses Windows I/O Completion Ports (IOCP) and local loopback sockets restricted by standard process sandboxes. When executing test suites or async scripts on Windows, always pass `BypassSandbox: true`.
