# Estimations Service Architecture and Provider Subpackage

## Overview

`src/aie_estimator/services/estimations_service.py` orchestrates software effort estimations using Context-Augmented Generation (CAG), prompt loading from versioned Jinja2 templates (`prompts/loader.py`), deterministic exact-match Redis caching, and execution delegation to an isolated provider subpackage at `src/aie_estimator/services/providers/`.

Provider execution is divided into two engines:
- **LiteLLM Router (`engine="litellm"`, default)** via `src/aie_estimator/services/providers/litellm_provider.py`: dynamic multi-provider routing across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius.
- **Native SDKs (`engine="native"`, legacy)** via `src/aie_estimator/services/providers/native_providers.py`: direct vendor SDK calls to the **OpenAI Responses API** and **Anthropic Messages API**.

Unified dispatch across both engines is exposed via `call_llm(user_prompt, system_prompt, opts)` in `src/aie_estimator/services/providers/__init__.py`.

The service architecture is driven by configuration externalized in `src/aie_estimator/llm_profiles.yml`:

| Engine | Routing Target | Dynamic Failover | Cost Tracking (`cost_usd`) | Telemetry Policy |
|:-------|:---------------|:-----------------|:---------------------------|:-----------------|
| **LiteLLM Router** (`engine="litellm"`) | Cost-based routing across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius | Supported cross-provider and cross-tier | Calculated automatically from provider token pricing (or estimated via fallback) | Strict zero-telemetry (`litellm.telemetry=False`) |
| **Native SDK** (`engine="native"`) | Direct vendor SDK calls to OpenAI or Anthropic | None (single provider) | Estimated via fallback pricing (`services/providers/pricing.py`) | Native vendor SDK defaults |

### Logical Routing Profiles

The service decouples business estimation requirements from specific vendor model IDs by providing logical profiles:
- **`reasoner-fast` (Tier 1)**: Cost-optimized reasoning tier for Phase-1 requirement extraction, standard estimations, and high-throughput workloads (routes across GPT-6 Luna low effort, Gemini Flash Lite low reasoning, Claude Haiku 5.5 adaptive medium, Mistral Small, Scaleway DeepSeek Flash).
- **`reasoner-deep` (Tier 2)**: Deep analytical reasoning tier for complex, contradictory transcripts and architecture risk matrices (routes across GPT-6 Luna high effort, Gemini Flash high reasoning, Claude Sonnet 5.5 adaptive medium, Nebius DeepSeek R2).

For details on parameter normalization and provider mappings, see [docs/reasoning-profiles.md](reasoning-profiles.md).

---

## Usage Examples

### 1. Generate Estimation with Estimations Service (Default)

```python
from aie_estimator.services.estimations_service import generate_estimation
from aie_estimator.services.providers import GenerationOptions

# Default options route through LiteLLM Router using reasoner-fast or configured default model
opts = GenerationOptions()
result = await generate_estimation(
    system_prompt="You are an expert estimator.",
    user_prompt="Client needs an e-commerce backend with Stripe integration...",
    opts=opts,
)

print(result["estimation"])
print(result["cost_usd"])  # e.g. 0.00034
print(result["provider"])  # "litellm" or resolved provider
```

### 2. Explicit Engine and Logical Profile Selection

```python
opts = GenerationOptions(
    engine="litellm",
    model="reasoner-deep",
    max_tokens=4000,
    temperature=0.3,
    max_retries=3,
)
result = await generate_estimation(system_prompt, user_prompt, opts=opts)
```

### 3. Fallback to Legacy Native SDK Execution

```python
# Direct OpenAI Responses API or Anthropic Messages API without LiteLLM Router
opts = GenerationOptions(
    engine="native",
    model="gpt-6-luna",
    max_tokens=2048,
)
result = await generate_estimation(system_prompt, user_prompt, opts=opts)
```

### 4. Direct Provider Dispatch via `call_llm`

```python
from aie_estimator.services.providers import GenerationOptions, call_llm

opts = GenerationOptions(model="reasoner-fast", engine="litellm")
response = await call_llm(
    user_prompt="Estimate this backend API project...",
    system_prompt="You are an expert estimator.",
    opts=opts,
)
print(response["cost_usd"])
```

---

## Technical Considerations

### 1. Isolated Provider Subpackage Architecture
The `src/aie_estimator/services/providers/` subpackage encapsulates all vendor SDK and router integrations:
- `generation_options.py`: Defines `GenerationOptions` with `temperature: float = 0.3`, `max_retries: int = 3`, `effective_model`, and `provider` properties.
- `pricing.py`: Defines static rates in `MODEL_COSTS` and computes cost in USD via `estimate_cost`.
- `model_catalog.py`: Parses and caches catalog metadata with multi-location fallback path resolution.
- `formatting.py`: Provides `format_provider_response` normalizing response dictionaries across all engines.
- `litellm_provider.py`: Manages LiteLLM Router initialization, missing-credential filtering, and completion dispatch.
- `native_providers.py`: Dispatches direct SDK requests to OpenAI or Anthropic with jittered exponential retries.
- `__init__.py`: Unifies engine dispatch under `call_llm(user_prompt, system_prompt, opts)`.

### 2. Startup Credential Filtering
When `get_litellm_router()` bootstraps the router singleton from `llm_profiles.yml`, each model's `api_key` environment variable reference (e.g. `os.environ/SCALEWAY_API_KEY`) is inspected. If the environment variable is missing or empty, that model is omitted from the active candidate list. This allows the application to run seamlessly in partial-credential environments (e.g. testing with only `OPENAI_API_KEY`).

### 3. Zero-Telemetry and Parameter Normalization Guardrails
Zero-telemetry is enforced programmatically during router instantiation based on `general_settings` in `llm_profiles.yml`:
```python
litellm.telemetry = False
litellm.drop_params = True
litellm.modify_params = True
```
This guarantees GDPR compliance and ensures unsupported parameters (e.g., passing temperature to reasoning models) are stripped before sending upstream.

### 4. Deterministic Cache Key Integration
`EstimationCache.make_key` computes a SHA-256 digest incorporating `system_prompt`, `user_message`, `model`, `max_tokens`, `reasoning_effort`, and `engine`. This prevents cache collisions between `litellm` and `native` executions as well as between different reasoning intensities.

### 5. Cost Tracking and Fallback Pricing Engine
`format_provider_response` (`src/aie_estimator/services/providers/formatting.py`) handles cost resolution:
- **Upstream Priority**: Dynamic costs returned by LiteLLM Router (`response._hidden_params["response_cost"]`) take precedence.
- **Fallback Calculation**: When `cost_usd is None` (such as direct native SDK invocations or unmetered routes), the service delegates to `estimate_cost(model, input_tokens, output_tokens)` from `src/aie_estimator/services/providers/pricing.py`.
- **Model Identifier Resolution**: `estimate_cost` normalizes vendor-prefixed identifiers (e.g. "openai/gpt-6-luna") matching against `MODEL_COSTS` keys.
- **Unconfigured Models**: If a model is not found in `MODEL_COSTS`, `estimate_cost` returns `None`. The resulting response omits `cost_usd`, and the Streamlit UI displays `Cost: N/A`.


