# Estimations Service Architecture and Provider Subpackage

## Overview

[`../backend/src`](../backend/src/aie_estimator/services/estimations_service.py) orchestrates software effort estimation using Context-Augmented Generation (CAG). It coordinates prompt rendering ([`../backend/src`](../backend/src/aie_estimator/prompts/loader.py)), pre-cache input semantic validation ([`../backend/src`](../backend/src/aie_estimator/guardrails/input.py)), two-tier response caching ([`../backend/src`](../backend/src/aie_estimator/services/cache/)), LLM provider dispatch ([`../backend/src`](../backend/src/aie_estimator/services/providers/)), and post-generation output filtering ([`../backend/src`](../backend/src/aie_estimator/guardrails/output.py)).

Provider execution supports two engines configured via [`../backend/src`](../backend/src/aie_estimator/llm_profiles.yml):
- **LiteLLM Router (`engine="litellm"`, default)**: Dynamic multi-provider routing and cross-tier failover across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius via [`litellm_provider.py`](../backend/src/aie_estimator/services/providers/litellm_provider.py).
- **Native SDKs (`engine="native"`)**: Direct vendor SDK dispatch to OpenAI or Anthropic via [`native_providers.py`](../backend/src/aie_estimator/services/providers/native_providers.py).

Unified dispatch is exposed through `call_llm(user_prompt, system_prompt, opts, *, messages=None, response_model=PmEstimation)` in [`../backend/src`](../backend/src/aie_estimator/services/providers/dispatcher.py).

---

## Usage Examples

### 1. Generating an Estimation with `EstimationService`

```python
from aie_estimator import EstimationRequest, EstimationResponse
from aie_estimator import get_exact_cache, get_semantic_cache
from aie_estimator import EstimationService

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
from aie_estimator import GenerationOptions, call_llm

opts = GenerationOptions(model="reasoner-fast", engine="litellm")
response = await call_llm(
    user_prompt="Estimate this backend API project...",
    system_prompt="You are an expert software estimation consultant.",
    opts=opts,
)
```

---

## Technical Considerations

1. **Provider Subpackage Structure**: [`../backend/src`](../backend/src/aie_estimator/services/providers/) isolates engine implementations:
   - [`dispatcher.py`](../backend/src/aie_estimator/services/providers/dispatcher.py): Single entry point routing between `litellm` and `native`.
   - [`litellm_provider.py`](../backend/src/aie_estimator/services/providers/litellm_provider.py): Router client wrapped with `instructor.from_litellm` and zero-telemetry configuration (`litellm.telemetry = False`). Raises `LLMProviderError` when given a non-profile model instead of silently falling back.
   - [`native_providers.py`](../backend/src/aie_estimator/services/providers/native_providers.py): Vendor SDK callers wrapped with Instructor, featuring 2-attempt parameter recovery on `BadRequestError` (remapping `max_tokens` to `max_completion_tokens`).
   - [`formatting.py`](../backend/src/aie_estimator/services/providers/formatting.py): Normalizes output dictionaries across engines for any validated Pydantic `BaseModel` (`PmEstimation` by default).
   - [`pricing.py`](../backend/src/aie_estimator/services/providers/pricing.py): Fallback token pricing estimation via `MODEL_COSTS` when upstream routers omit cost metadata.
2. **Structured Extraction with Instructor**: The service enforces typed extraction into `response_model` (defaulting to [`PmEstimation`](../backend/src/aie_estimator/schemas/estimations.py), and parameterized for tier schemas or auxiliary calls such as [`MetadataExtractionResult`](../backend/src/aie_estimator/sessions/models.py)). If validation fails (such as phase cost mismatch), Instructor feeds errors back to the LLM up to `max_retries`. Exception unwrapping via `unwrap_exception` ([`exceptions.py`](../backend/src/aie_estimator/services/exceptions.py)) inspects root causes before raising `LLMProviderError`.
3. **Multi-Layer Resilience Hierarchy**:
   - **Layer 1 (Transport)**: HTTP client retries and exponential backoff (`httpx` / vendor SDKs).
   - **Layer 2 (Schema Adherence)**: Instructor validation re-prompting loop.
   - **Layer 3 (Parameter Recovery)**: Automatic parameter stripping on vendor rejections.
   - **Layer 4 (Dynamic Failover)**: LiteLLM intra-tier model rotation and cross-tier escalation (`reasoner-fast` → `reasoner-deep`). See [`docs/reasoning-profiles.md`](reasoning-profiles.md).
4. **Execution Pipeline Order**: `check_input` runs *before* cache lookups to prevent invalid prompts or PII from querying or polluting cache stores. Dual-write persistence occurs only after output filtering passes. See [`docs/guardrails.md`](guardrails.md) and [`docs/semantic-response-caching.md`](semantic-response-caching.md).
5. **Conversational Attachment & Post-Turn Memory Refresh**: [`EstimationService.generate_session_estimation`](../backend/src/aie_estimator/services/estimations_service.py) validates attachment counts, runs strict `check_input` on the raw user transcript, extracts and sanitizes optional `(filename, bytes)` attachments via [`extract_text`](../backend/src/aie_estimator/attachments/extractor.py) and [`check_attachment_input`](../backend/src/aie_estimator/guardrails/input.py) (redacting PII inline while blocking moderation/prompt-injection violations), and wraps non-empty attachments in `<attachments>` XML blocks via `enrich_transcript`. It bypasses Tier 1 and Tier 2 caches, appends the enriched turn to `session.history`, and invokes [`update_metadata`](../backend/src/aie_estimator/sessions/metadata_extractor.py). Compression and extractor token usage and cost aggregate into `session.last_turn_observed` ([`TurnObservation`](../backend/src/aie_estimator/sessions/models.py)), preserving `cost_usd=None` whenever primary generation is unpriced, while [`EstimationResponse`](../backend/src/aie_estimator/schemas/estimations.py) retains generation-only `usage` and `cost_usd`.
