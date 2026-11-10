# Estimations Service Architecture and Provider Subpackage

## Overview

[`src/aie_estimator/services/estimations_service.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/estimations_service.py) orchestrates software effort estimation using Context-Augmented Generation (CAG). It coordinates prompt rendering ([`src/aie_estimator/prompts/loader.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/prompts/loader.py)), pre-cache input semantic validation ([`src/aie_estimator/guardrails/input.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/guardrails/input.py)), two-tier response caching ([`src/aie_estimator/services/cache/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/cache/)), LLM provider dispatch ([`src/aie_estimator/services/providers/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/)), and post-generation output filtering ([`src/aie_estimator/guardrails/output.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/guardrails/output.py)).

Provider execution supports two engines configured via [`src/aie_estimator/llm_profiles.yml`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/llm_profiles.yml):
- **LiteLLM Router (`engine="litellm"`, default)**: Dynamic multi-provider routing and cross-tier failover across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius via [`litellm_provider.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/litellm_provider.py).
- **Native SDKs (`engine="native"`)**: Direct vendor SDK dispatch to OpenAI or Anthropic via [`native_providers.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/native_providers.py).

Unified dispatch is exposed through `call_llm(user_prompt, system_prompt, opts)` in [`src/aie_estimator/services/providers/dispatcher.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/dispatcher.py).

---

## Usage Examples

### 1. Generating an Estimation with `EstimationService`

```python
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services.cache import get_exact_cache, get_semantic_cache
from aie_estimator.services.estimations_service import EstimationService

service = EstimationService(
    exact_cache=get_exact_cache(),
    semantic_cache=get_semantic_cache(),
)

request = EstimationRequest(
    description="Scalable notification microservice with email, SMS, and webhooks.",
    project_type="web_saas",
    detail_level="medium",
    output_format="phases_table",
    model="reasoner-fast",
    engine="litellm",
)

response: EstimationResponse = await service.generate_estimation(request)
```

### 2. Direct Provider Dispatch via `call_llm`

```python
from aie_estimator.services.providers import GenerationOptions, call_llm

opts = GenerationOptions(model="reasoner-fast", engine="litellm")
response = await call_llm(
    user_prompt="Estimate this backend API project...",
    system_prompt="You are an expert software estimation consultant.",
    opts=opts,
)
```

---

## Technical Considerations

1. **Provider Subpackage Structure**: [`src/aie_estimator/services/providers/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/) isolates engine implementations:
   - [`dispatcher.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/dispatcher.py): Single entry point routing between `litellm` and `native`.
   - [`litellm_provider.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/litellm_provider.py): Router client wrapped with `instructor.from_litellm` and zero-telemetry configuration (`litellm.telemetry = False`).
   - [`native_providers.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/native_providers.py): Vendor SDK callers wrapped with Instructor, featuring 2-attempt parameter recovery on `BadRequestError` (stripping unsupported `temperature` or remapping `max_tokens` to `max_completion_tokens`).
   - [`formatting.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/formatting.py): Normalizes output dictionaries across engines into `EstimationResult`.
   - [`pricing.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/providers/pricing.py): Fallback token pricing estimation via `MODEL_COSTS` when upstream routers omit cost metadata.
2. **Structured Extraction with Instructor**: The service enforces typed extraction into [`EstimationResult`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py). If validation fails (such as phase cost mismatch), Instructor feeds errors back to the LLM up to `max_retries`. Exception unwrapping via `unwrap_exception` ([`exceptions.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/services/exceptions.py)) inspects root causes before raising `LLMProviderError`.
3. **Multi-Layer Resilience Hierarchy**:
   - **Layer 1 (Transport)**: HTTP client retries and exponential backoff (`httpx` / vendor SDKs).
   - **Layer 2 (Schema Adherence)**: Instructor validation re-prompting loop.
   - **Layer 3 (Parameter Recovery)**: Automatic parameter stripping on vendor rejections.
   - **Layer 4 (Dynamic Failover)**: LiteLLM intra-tier model rotation and cross-tier escalation (`reasoner-fast` → `reasoner-deep`). See [`docs/reasoning-profiles.md`](file:///c:/Development/Repos/ai-engineering/aie-estimator/docs/reasoning-profiles.md).
4. **Execution Pipeline Order**: `check_input` runs *before* cache lookups to prevent invalid prompts or PII from querying or polluting cache stores. Dual-write persistence occurs only after output filtering passes. See [`docs/guardrails.md`](file:///c:/Development/Repos/ai-engineering/aie-estimator/docs/guardrails.md) and [`docs/semantic-response-caching.md`](file:///c:/Development/Repos/ai-engineering/aie-estimator/docs/semantic-response-caching.md).
