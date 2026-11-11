# Semantic Response Caching

## Overview

The estimation service implements a two-tier caching architecture in [`src/aie_estimator/services/cache/`](../src/aie_estimator/services/cache/) to optimize latency and minimize LLM token expenditure:

1. **Tier 1 (Exact-Match Cache)**: [`EstimationExactCache`](../src/aie_estimator/services/cache/exact.py) computes a deterministic SHA-256 hash across all generation parameters. Lookups execute in ~1 ms with zero embedding overhead.
2. **Tier 2 (Semantic Vector Cache)**: [`EstimationSemanticCache`](../src/aie_estimator/services/cache/semantic.py) evaluates cosine vector distance in Redis using RediSearch on exact-match misses. Queries with equivalent intent return cached estimations in ~80 ms without invoking upstream LLM providers.

Pipeline flow:
`check_input` → Tier 1 exact cache miss → OpenAI text embedding (`text-embedding-3-small`) → Tier 2 semantic cache query → (Miss) `call_llm` → `enforce_scope_response` → Dual-write persist to both caches.

---

## Usage Examples

### 1. Direct Lookup and Storage with `EstimationSemanticCache`

```python
from aie_estimator.schemas.estimations import EstimationRequest
from aie_estimator.services.cache import get_semantic_cache

cache = get_semantic_cache()
request = EstimationRequest(
    description="Mobile application with user authentication, live chat, and push notifications.",
    project_type="mobile_app",
    detail_level="medium",
    output_format="phases_table",
)

cached_result = await cache.lookup(request, prompt_version="v1")
if not cached_result:
    # On cache miss, generate and persist
    await cache.store(request, result, prompt_version="v1")
```

### 2. Lifespan Index Initialization

```python
# Managed automatically in src/aie_estimator/server.py:
if settings.enable_semantic_cache:
    await semantic_cache.initialize_index()
```

---

## Technical Considerations

1. **Composite Bucket Partitioning**: Requests are partitioned using bucket tags combining `prompt_version`, `project_type`, `detail_level`, and `output_format` via `bucket_for()`. The RediSearch query restricts comparison to documents matching the escaped bucket tag (`escape_tag(bucket)`) before computing vector distances, eliminating cross-format cache pollution.
2. **Native `redis.asyncio` RediSearch Integration**: Avoids third-party ORMs by using `redis.commands.search.field` (`TagField`, `TextField`, `VectorField`) and Python's standard library `struct.pack` for contiguous 32-bit float vector serialization under `dialect(2)`.
3. **Shadow Mode (`SEMANTIC_CACHE_LOG_ONLY=true`)**: Evaluates RediSearch vector queries, logs similarity scores via `semantic_cache_hit_log_only`, but returns `None` to let requests proceed to the LLM for threshold calibration.
4. **Fail-Open Policy**: If Redis or OpenAI embedding generation encounters network timeouts or exceptions, the cache logs a warning and gracefully falls open to direct LLM generation without failing the user request.
5. **Test Isolation**: The test harness in [`tests/conftest.py`](../tests/conftest.py) patches index initialization to an async no-op and defaults caches to safe miss states, eliminating physical socket connection attempts during test runs.
