# Dependency Injection & Application Factory

## Overview

The application implements the Application Factory pattern in [`src/aie_estimator/server.py`](../src/aie_estimator/server.py) and a centralized Composition Root in [`src/aie_estimator/dependencies.py`](../src/aie_estimator/dependencies.py). This architecture decouples the HTTP transport layer from domain orchestration, eliminates shared mutable singletons across test executions, and enables deterministic mocking via FastAPI's `app.dependency_overrides`.

Layer flow:
`create_app()` ([`server.py`](../src/aie_estimator/server.py)) → Route Handlers ([`src/aie_estimator/routers/`](../src/aie_estimator/routers/)) → `Depends(get_estimation_service)` ([`dependencies.py`](../src/aie_estimator/dependencies.py)) → [`EstimationService`](../src/aie_estimator/services/estimations_service.py).

---

## Usage Examples

### 1. Application Factory Instantiation

```python
from aie_estimator.server import create_app
from aie_estimator.settings import Settings

# Default configuration from environment variables
app = create_app()

# Custom settings for isolated environments or tests
custom_app = create_app(settings=Settings(server_port=9000))
```

### 2. Injecting `EstimationService` into Route Handlers

```python
from fastapi import APIRouter, Depends
from aie_estimator.dependencies import get_estimation_service
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services.estimations_service import EstimationService

router = APIRouter(prefix="/api/v1")


@router.post("/estimate", response_model=EstimationResponse)
async def create_estimation(
    request: EstimationRequest,
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    return await service.generate_estimation(request)
```

### 3. Overriding Dependencies in Tests

```python
from unittest.mock import AsyncMock
from aie_estimator.dependencies import get_estimation_service
from aie_estimator.services.estimations_service import EstimationService

mock_service = AsyncMock(spec=EstimationService)
client.app.dependency_overrides[get_estimation_service] = lambda: mock_service
```

---

## Technical Considerations

1. **Separation of Concerns**: [`Settings`](../src/aie_estimator/settings.py) is consumed only at the factory (`create_app`) and Composition Root ([`dependencies.py`](../src/aie_estimator/dependencies.py)). Domain services receive concrete collaborators (`exact_cache`, `semantic_cache`, `moderation_client`) rather than entire settings instances.
2. **Factory Invocation**: ASGI entry points ([`src/aie_estimator/__main__.py`](../src/aie_estimator/__main__.py)) invoke Uvicorn using the factory string (`"aie_estimator.server:create_app"`, `factory=True`), eliminating module-level singleton state.
3. **Test Isolation**: Factories in [`dependencies.py`](../src/aie_estimator/dependencies.py) use `@lru_cache(maxsize=1)`. The test harness in [`tests/conftest.py`](../tests/conftest.py) clears cached factories before and after each test via `_clear_all_caches()`.
