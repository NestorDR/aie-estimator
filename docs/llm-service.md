# LLM Service Architecture and Multi-Provider Routing

## Overview

`src/aie_estimator/services/llm_service.py` orchestrates software effort estimations using Context-Augmented Generation (CAG), configurable input preprocessing, deterministic exact-match Redis caching, and a modular dual-engine architecture delegating execution to **`src/aie_estimator/services/litellm_provider.py`** (LiteLLM Router, default) and **`src/aie_estimator/services/native_providers.py`** (legacy **OpenAI Responses API** and **Anthropic Messages API** callers).

The service is driven by configuration externalized in `src/aie_estimator/llm_profiles.yml`:

| Engine | Routing Target | Dynamic Failover | Cost Tracking (`cost_usd`) | Telemetry Policy |
|:-------|:---------------|:-----------------|:---------------------------|:-----------------|
| **LiteLLM Router** (`engine="litellm"`) | Cost-based routing across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius | Supported cross-provider and cross-tier | Calculated automatically from provider token pricing (or estimated via fallback) | Strict zero-telemetry (`litellm.telemetry=False`) |
| **Native SDK** (`engine="native"`) | Direct vendor SDK calls to OpenAI or Anthropic | None (single provider) | Estimated via fallback pricing (`services/pricing.py`) | Native vendor SDK defaults |

### Logical Routing Profiles

The service decouples business estimation requirements from specific vendor model IDs by providing logical profiles:
- **`reasoner-fast` (Tier 1)**: Cost-optimized reasoning tier for Phase-1 requirement extraction, standard estimations, and high-throughput workloads (routes across GPT-6 Luna low effort, Gemini Flash Lite low reasoning, Claude Haiku 5.5 adaptive medium, Mistral Small, Scaleway DeepSeek Flash).
- **`reasoner-deep` (Tier 2)**: Deep analytical reasoning tier for complex, contradictory transcripts and architecture risk matrices (routes across GPT-6 Luna high effort, Gemini Flash high reasoning, Claude Sonnet 5.5 adaptive medium, Nebius DeepSeek R2).

For details on parameter normalization and provider mappings, see [docs/reasoning-profiles.md](file:///c:/Development/Repos/ai-engineering/aie-estimator/docs/reasoning-profiles.md).

---

## Usage Examples

### 1. Generate Estimation with LiteLLM Router (Default)

```python
from aie_estimator.services.llm_service import GenerationOptions, generate_estimation

# Default options route through LiteLLM Router using reasoner-fast or default model
result = await generate_estimation("Client needs an e-commerce backend with Stripe integration...")

print(result["estimation"])
print(result["cost_usd"])    # e.g. 0.00034
print(result["provider"])    # "litellm" or resolved provider
```

### 2. Explicit Engine and Logical Profile Selection

```python
opts = GenerationOptions(
    engine="litellm",
    model="reasoner-deep",
    preprocessing="two_phase",
    max_tokens=4000,
)
result = await generate_estimation("Complex enterprise transcript...", opts=opts)
```

### 3. Fallback to Legacy Native SDK Execution

```python
# Direct OpenAI Responses API or Anthropic Messages API without LiteLLM Router
opts = GenerationOptions(
    engine="native",
    model="gpt-6-luna",
    max_tokens=2048,
)
result = await generate_estimation("Standard transcript...", opts=opts)
```

### 4. Router Singleton and Direct Completion

```python
from aie_estimator.services.litellm_provider import call_litellm_router, get_litellm_router

# Singleton Router access (@lru_cache) with zero-telemetry and missing-credential filtering
router = get_litellm_router()

# Direct async router completion (required for cost-based-routing)
response = await call_litellm_router(
    messages="Estimate this backend API project...",
    system_prompt="You are an expert estimator.",
    model="reasoner-fast",
    max_tokens=2000,
)
print(response["cost_usd"])
```

---

## Technical Considerations

### 1. Startup Credential Filtering
When `get_litellm_router()` bootstraps the router singleton from `llm_profiles.yml`, each model's `api_key` environment variable reference (e.g. `os.environ/SCALEWAY_API_KEY`) is inspected. If the environment variable is missing or empty, that model is omitted from the active candidate list. This allows the application to run seamlessly in partial-credential environments (e.g. testing with only `OPENAI_API_KEY`).

### 2. Zero-Telemetry and Parameter Normalization Guardrails
Zero-telemetry is enforced programmatically during router instantiation based on `general_settings` in `llm_profiles.yml`:
```python
litellm.telemetry = False
litellm.drop_params = True
litellm.modify_params = True
```
This guarantees GDPR compliance and ensures unsupported parameters (e.g., passing temperature to reasoning models) are stripped before sending upstream.

### 3. Execution Pipeline & Cache Key Integration
1. **Preprocessing (`opts.preprocessing`)**: Supports `"none"`, `"inline_cleaning"` (prompt injection), and `"two_phase"` (Phase-1 requirement extraction via `extract_requirements`, routed based on `opts.engine`).
2. **Deterministic Cache Key**: `EstimationCache.make_key` computes a SHA-256 digest incorporating `engine` alongside `system_prompt`, `user_message`, `model`, `max_tokens`, `reasoning_effort`, and `preprocessing`. This prevents cache collisions between `litellm` and `native` executions as well as between different reasoning intensities.
3. **Cost Extraction**: LiteLLM completion responses provide execution cost via `response._hidden_params["response_cost"]`, normalized as float `cost_usd` in the service output.

### 4. Asynchronous Non-Blocking I/O Architecture
All service I/O boundaries operate as asynchronous coroutines to ensure FastAPI's single-threaded event loop is never blocked:
- **Native Providers (`services/native_providers.py`)**: Implements client singletons using `AsyncOpenAI` and `AsyncAnthropic`. Transient retry logic (`call_with_retry`) uses `await asyncio.sleep(...)` with jitter, and all provider invocations (`call_openai`, `call_anthropic`) are asynchronous coroutines.
- **Exact-Match Caching (`services/cache.py`)**: Utilizes `redis.asyncio.Redis` with asynchronous `get`, `set`, and `close` methods. Cache lookups and mutations are awaited within `generate_estimation`.
- **Application Lifespan Integration (`server.py`)**: The application shutdown sequence triggers `await cache.close()` to cleanly terminate Redis client connections without dangling socket warnings.

### 5. Cost Tracking and Fallback Pricing Engine
`format_provider_response` (`src/aie_estimator/services/__init__.py`) handles cost resolution:
- **Upstream Priority**: Dynamic costs returned by LiteLLM Router (`response._hidden_params["response_cost"]`) take precedence.
- **Fallback Calculation**: When `cost_usd is None` (such as direct native SDK invocations or unmetered routes), the service delegates to `estimate_cost(model, input_tokens, output_tokens)` from `src/aie_estimator/services/pricing.py`.
- **Model Identifier Resolution**: `estimate_cost` normalizes vendor-prefixed identifiers (e.g. "openai/gpt-6-luna") matching against `MODEL_COSTS` keys.
- **Unconfigured Models**: If a model is not found in `MODEL_COSTS`, `estimate_cost` returns `None`. The resulting response omits `cost_usd`, and the Streamlit UI displays `Cost: N/A`.

