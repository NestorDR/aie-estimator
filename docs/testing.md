# Testing Guidelines and Shared Fixtures

## Overview

Guide for writing, organizing, and validating tests in `aie-estimator`. The main sections state the conventions new tests must follow. Measured figures and the cleanup backlog live in the dated [Appendix](#appendix-current-state-snapshot).

When a convention changes, update the rule here first, then bring existing tests into line.

### Structure and Configuration

- **Configuration**: `[tool.pytest.ini_options]` in `pyproject.toml` (`testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `pythonpath = [".", "src"]`). No editable install is required.
- **Execution**: `rtk uv run pytest` with `APP_ENV=test` set (omit `rtk` if it is unavailable). See [Environment Isolation](#3-environment-isolation).

### Test Layout

Test modules mirror the source tree, one test module per source module. For example, `src/aie_estimator/services/cache.py` → `tests/services/test_cache.py`, and `streamlit_app.py` → `tests/test_streamlit_app.py`.

- A new source module ships with its test module in the same change.
- Test each behavior at the lowest layer that owns it: exhaustive validation in schema tests, one representative case per error mapping in router tests.
- Name tests after unit, condition, and outcome (e.g. `test_cache_get_miss_returns_none`).

### Fixtures

- **Placement**: fixtures shared by several test modules go in `tests/conftest.py`; otherwise keep them module-local or in a package-level `conftest.py`.
- **No orphans**: every shared fixture must have at least one consumer.
- **Factories over repetition**: when the same payload or mock shape repeats across tests, provide it through a factory fixture (see [Pattern 6](#6-factory-fixtures)) instead of inline dictionaries or module constants.
- **Autouse only for isolation**: e.g. `clean_settings_cache` in `tests/conftest.py` clears application `lru_cache` singletons around each test, and `mock_default_cache` in `test_llm_service.py` keeps tests off Redis. Register every new `lru_cache` singleton in `_clear_all_caches()`.

Core shared fixtures are `app` (FastAPI app with override teardown), `client` (`TestClient` running lifespan handlers), and `sample_transcription`. See `tests/conftest.py` for the full list.

---

## Usage Examples

Excerpts from existing tests showing the reference style. Imports and docstrings are trimmed except where they illustrate the convention.

### 1. Parametrization with Explicit `ids`

```python
@pytest.mark.parametrize(
    ("override_version", "expected_version"),
    [(None, None), ("9.9.9-test", "9.9.9-test")],
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
    ...
```

### 2. Concise Exception Assertions

```python
with pytest.raises(ValidationError, match=r"valid integer"):
    Settings(server_port="invalid_port")
```

### 3. Environment Overrides

Use `monkeypatch`, never direct `os.environ` assignment. `clean_settings_cache` guarantees the next `get_settings()` re-reads the environment:

```python
def test_get_settings_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()

    assert get_settings().app_env == Environment.TEST
```

### 4. Mocking at the Boundary

Replace collaborators with `AsyncMock` and patch the name where it is looked up, not where it is defined:

```python
monkeypatch.setattr(
    "aie_estimator.routers.estimations_router.llm_service.generate_estimation",
    AsyncMock(return_value=result),
)
response = client.post("/api/v1/estimate", json={"transcription": sample_transcription, "evaluate": False})
assert response.status_code == 200
```

### 5. Async Tests

Write coroutine tests as `async def` with `@pytest.mark.anyio` (asyncio backend, hence `[asyncio]` in test ids). Avoid `asyncio.run(...)` inside sync tests.

```python
@pytest.mark.anyio
async def test_cache_get_miss_returns_none() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    assert await EstimationCache(mock_redis).get("estimation:missing") is None
```

### 6. Factory Fixtures

Target pattern for repeated payloads: defaults plus keyword overrides, so each test states only the fields it asserts on.

```python
@pytest.fixture
def make_estimation_result() -> Callable[..., dict[str, Any]]:
    def _factory(**overrides: Any) -> dict[str, Any]:
        return {"estimation": "Total: 40h", "model": DEFAULT_GPT_LLM, "provider": "openai", **overrides}

    return _factory
```

### 7. Logging Without File Writes

Tests that enable file logging patch the file handler and directory creation, and clear root handlers afterwards:

```python
with (
    patch("logging.handlers.RotatingFileHandler", autospec=True),
    patch("pathlib.Path.mkdir"),
):
    try:
        Settings(app_env=Environment.DEV, enable_file_logging=True).configure_logging()
        ...
    finally:
        logging.getLogger().handlers.clear()
```

### 8. CLI Execution

```powershell
$env:APP_ENV = "test"; rtk uv run pytest --cache-clear --basetemp="$env:TEMP\pytest_runner"
rtk uv run pytest tests/routers/test_health_router.py
rtk uv run pytest -k "dependency_override"
rtk uv run --with pytest-cov pytest --cov=aie_estimator --cov=streamlit_app --cov-branch --cov-report=term-missing
```

`auto\test.cmd` wraps the suite run and cleans up test artifacts (manual use only).

Prefix supported terminal commands with `rtk`; if it is unavailable, run the base command. Use the context-mode MCP tools or Antigravity native search for codebase discovery, not `grep` or `Select-String`.

---

## Coverage

- **Target**: 85% combined line + branch coverage for `aie_estimator` and `streamlit_app`. Always measure with `--cov-branch`.
- **No regressions**: a change must not lower total coverage; new modules ship with tests for their public functions and error branches.
- **Bug fixes** include a regression test that fails before the fix.
- **Exclusions** (`# pragma: no cover`) are limited to entry-point glue and carry a reason comment.
- **Priorities**: business logic first, then error branches and exception handlers, then UI code (extract testable functions rather than testing Streamlit rendering).
- **Enforcement** (`pytest-cov` as a dev dependency and `--cov-fail-under=85`) is planned once the baseline reaches the target.

---

## Quality Gates

| Gate | Status |
| :--- | :--- |
| Ruff lint (`E,F,B,R,S,N`) on tests | Active |
| Ruff test-oriented rules (`PT`, `ANN`, `ARG`, `SIM`, `I`, `UP`) | Planned |
| `APP_ENV=test` pinned for the suite (`pytest-env` or autouse fixture) plus a `.env.test` | Active |
| Coverage threshold (`--cov-fail-under=85`) | Active |
| Random test order (`pytest-randomly`) to expose state leaks | Planned |
| Property-based tests (Hypothesis) for functions with invariants, e.g. cache-key determinism | Planned |
| Mutation testing pilot (Cosmic Ray on Windows; mutmut needs WSL) | Planned |

Update the status when a gate is adopted and add any resulting rule to the sections above. Keep the pre-commit ruff `rev` aligned with the ruff version in `uv.lock`.

---

## Technical Considerations

### 1. Test Design Rules

- **GIVEN-WHEN-THEN**: Every test function docstring must explicitly follow GIVEN (preconditions), WHEN (action), THEN (assertions) followed by Sphinx `:param` definitions.
- **Parametric Deduplication**: Consolidate symmetric branches with `@pytest.mark.parametrize`. Always specify `ids` to prevent unreadable `[None-None]` auto-generated identifiers, enable `-k` filtering, and clarify CI failure diagnostics.
- **Concise Exception Regexes**: Assert exceptions with `pytest.raises(Exc, match=r'...')` using minimal distinctive keywords to prevent brittleness against minor formatting revisions.
- **Type Annotations**: All fixtures, test parameters, and return types require explicit type hints.
- **One Behavior per Test**: split tests whose THEN clause joins unrelated outcomes.
- **No Real I/O**: tests never call real LLM providers, Redis, or the network.

### 2. Runtime Isolation Mechanics

- **Lifespan Execution**: the `client` fixture uses `with TestClient(app)` so startup/shutdown handlers run.
- **Cache Isolation**: `clean_settings_cache` clears application `lru_cache` singletons before and after each test.
- **Override Teardown**: the `app` fixture clears `dependency_overrides` on teardown.

### 3. Environment Isolation

`Settings` resolves the environment from its `app_env` argument, then `APP_ENV`, then defaults to `dev`. It loads `src/aie_estimator/.env.<env>`, configures logging, and mirrors API keys into `os.environ`. Without `APP_ENV=test`, tests therefore pick up the developer's `.env.dev`: real keys enter the test process and file logging, if enabled there, writes to `logs/dev.log`.

- Run the suite with `APP_ENV=test` until the pinning gate is active.
- Tests that construct `Settings` for `dev` or `prod` disable or patch file logging (Pattern 7).
- Use `tmp_path` for any file the code under test creates.
- Tests must never produce `logs/dev.log`.
