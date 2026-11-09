# Semantic Response Caching

## Overview

The `aie_estimator` service implements a two-tier caching architecture to optimize latency and minimize LLM token costs:
1. **Tier 1 (Exact-Match Cache)**: `EstimationExactCache` ([exact.py](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/cache/exact.py)) computes a deterministic SHA-256 hash across all input generation parameters. Cache lookups execute in ~1 ms with zero embedding overhead.
2. **Tier 2 (Semantic Vector Cache)**: `EstimationSemanticCache` ([semantic.py](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/cache/semantic.py)) evaluates cosine vector distance in Redis using RediSearch when exact matching misses. Natural language descriptions asking for identical project scopes return cached results in ~80 ms without invoking upstream LLM providers.

### Request Pipeline

```
                    ┌────────────────────────┐
                    │   EstimationRequest    │
                    └───────────┬────────────┘
                                │
                                v
               [ 1. Input Guardrails (check_input) ]
                                │
                                v
               [ 2. Tier 1: Exact-Match Cache ] ──── (HIT, ~1ms) ───► Return (cached=True)
                                │ (MISS)
                                v
              [ Compute Embedding (OpenAI, ~50ms) ]
                                │
                                v
               [ 3. Tier 2: Semantic Vector Cache ] ── (HIT, ~80ms) ──► Return (cached=True)
                   Similarity >= Threshold (0.92)?
                                │ (MISS)
                                v
               [ 4. LLM Generation (call_llm) ]
                                │
                                v
               [ 5. Output Guardrail (Filter) ]
                                │
                                v
               [ 6. Dual-Write to Both Caches ] ─────────────────────► Return (cached=False)
```

---

## Usage Examples

### 1. Direct Lookup and Storage with `EstimationSemanticCache`

```python
from aie_estimator.schemas.estimations import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from aie_estimator.services.cache import get_semantic_cache

# Initialize request and semantic cache singleton
cache = get_semantic_cache()
request = EstimationRequest(
    description="Mobile application with user authentication, live chat, and push notifications.",
    project_type=ProjectType.MOBILE_APP,
    detail_level=DetailLevel.MEDIUM,
    output_format=OutputFormat.PHASES_TABLE,
)

# Tier 2 lookup against bucket partition "v1:mobile_app:medium:phases_table"
cached_result = await cache.lookup(request, prompt_version="v1")
if cached_result:
    print(f"Cache Hit! Total Cost: {cached_result.total_cost_eur} EUR")
else:
    print("Cache Miss. Proceeding to LLM provider dispatch.")
```

### 2. Lifespan Index Initialization in FastAPI

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from aie_estimator.services.cache import get_semantic_cache
from aie_estimator.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.enable_semantic_cache:
        semantic_cache = get_semantic_cache()
        # Idempotent schema and index registration
        await semantic_cache.initialize_index()

    yield

    if settings.enable_semantic_cache:
        await semantic_cache.close()
```

---

## Technical Considerations

### 1. Composite Bucket Keys
To prevent cross-configuration cache pollution (such as returning a narrative summary when the user requested a phases table), semantic similarity is strictly partitioned using RediSearch tag filters:
```python
bucket = f"{prompt_version}:{request.project_type.value}:{request.detail_level.value}:{request.output_format.value}"
# Example: "v1:mobile_app:medium:phases_table"
```
The search query restricts comparison exclusively to documents within the matching bucket before calculating vector distances:
```
(@bucket:{v1\:mobile_app\:medium\:phases_table})=>[KNN 1 @embedding $vec AS vector_distance]
```
When `prompt_version` increments (e.g., from `v1` to `v2`), all previous bucket entries become inactive immediately, yielding natural zero-downtime cache invalidation without requiring manual Redis purging.

### 2. Native `redis.asyncio` RediSearch Integration
The implementation avoids third-party abstraction dependencies (such as `redisvl` or `numpy`):
- `redis.commands.search.field`: `TagField("bucket")`, `TextField("result_json")`, and `VectorField("embedding", "FLAT", ...)`.
- Standard library `struct.pack`: Converts 1,536 float values into contiguous 32-bit float bytes (`struct.pack(f"{len(vec)}f", *vec)`).
- Dialect 2: Queries enable modern vector syntax and score aliases via `.dialect(2)`.

### 3. Threshold Calibration via Shadow Mode (`semantic_cache_log_only`)
Setting `SEMANTIC_CACHE_LOG_ONLY=true` in `.env` enables shadow mode:
- Generates embeddings and performs the RediSearch query.
- Emits structured telemetry logs: `semantic_cache_hit_log_only`, recording the calculated similarity score.
- Returns `None` to ensure the request proceeds to the LLM without bypassing inference.
- Allows inspecting similarity scores on real user traffic to tune `SEMANTIC_CACHE_THRESHOLD` (default: `0.92`).

### 4. Fail-Open Graceful Degradation
If Redis is offline, OpenAI embeddings fail, or RediSearch raises a connection timeout, the cache catches exceptions, emits a warning log, and falls open to upstream LLM generation without failing user requests.

### 5. Lazy Prompt Template Rendering
Jinja2 prompt rendering is deferred until both cache tiers miss. By passing `request: EstimationRequest` directly into `generate_estimation(request, opts, prompt_version)`, template rendering CPU cycles are completely eliminated during both Tier 1 exact hits and Tier 2 semantic hits.

### 6. Test Suite Isolation & Offline Startup
In automated test suites, the test harness in [tests/conftest.py](file:///c:/Development/Repos/ai-engineering/aie-estimator/tests/conftest.py) isolates Tier 2 caching from physical infrastructure:
- **Lifespan Neutralization**: The `app` fixture patches `EstimationSemanticCache.initialize_index` to an async no-op, preventing blocking 1.02s TCP socket connection attempts when running offline without Redis.
- **Default Miss State**: The `default_tiered_cache_isolation` autouse fixture configures Tier 1 and Tier 2 caches to return safe miss states (`lookup -> None`, `store -> no-op`), eliminating accidental network calls to OpenAI embedding endpoints and Redis sockets across all unit and router tests.


