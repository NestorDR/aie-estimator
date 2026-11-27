# Estimations Service Architecture and Provider Subpackage

## Overview

[`estimations_service.py`](../backend/src/aie_estimator/domain/estimations_service.py) orchestrates software effort estimation using Context-Augmented Generation (CAG). It coordinates prompt rendering ([`loader.py`](../backend/src/aie_estimator/foundation/prompts/loader.py)), pre-cache input semantic validation ([`input.py`](../backend/src/aie_estimator/foundation/guardrails/input.py)), two-tier response caching ([`cag/`](../backend/src/aie_estimator/generation/cag/)), LLM provider dispatch ([`llm/`](../backend/src/aie_estimator/foundation/llm/)), and post-generation output filtering ([`output.py`](../backend/src/aie_estimator/foundation/guardrails/output.py)).

Provider execution supports two engines configured via [`llm_profiles.yml`](../backend/src/aie_estimator/llm_profiles.yml):
- **LiteLLM Router (`engine="litellm"`, default)**: Dynamic multi-provider routing and cross-tier failover across OpenAI, Anthropic, Gemini, Mistral, Scaleway, and Nebius via [`litellm_provider.py`](../backend/src/aie_estimator/foundation/llm/litellm_provider.py).
- **Native SDKs (`engine="native"`)**: Direct vendor SDK dispatch to OpenAI or Anthropic via [`native_providers.py`](../backend/src/aie_estimator/foundation/llm/native_providers.py).

Unified dispatch is exposed through `call_llm(user_prompt, system_prompt, opts, *, messages=None, response_model=PmEstimation)` in [`dispatcher.py`](../backend/src/aie_estimator/foundation/llm/dispatcher.py).

---

## Usage Examples

### 1. Generating an Estimation with `EstimationService`

```python
from aie_estimator.domain.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.dependencies import get_exact_cache, get_semantic_cache
from aie_estimator.domain.estimations_service import EstimationService

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
from aie_estimator.foundation.llm import GenerationOptions, call_llm

opts = GenerationOptions(model="reasoner-fast", engine="litellm")
response = await call_llm(
    user_prompt="Estimate this backend API project...",
    system_prompt="You are an expert software estimation consultant.",
    opts=opts,
)
```

---

## Technical Considerations

1. **Provider Subpackage Structure**: [`llm/`](../backend/src/aie_estimator/foundation/llm/) isolates engine implementations:
   - [`dispatcher.py`](../backend/src/aie_estimator/foundation/llm/dispatcher.py): Single entry point routing between `litellm` and `native`.
   - [`litellm_provider.py`](../backend/src/aie_estimator/foundation/llm/litellm_provider.py): Router client wrapped with `instructor.from_litellm` and zero-telemetry configuration (`litellm.telemetry = False`). Raises `LLMProviderError` when given a non-profile model instead of silently falling back.
   - [`native_providers.py`](../backend/src/aie_estimator/foundation/llm/native_providers.py): Vendor SDK callers wrapped with Instructor, featuring 2-attempt parameter recovery on `BadRequestError` (remapping `max_tokens` to `max_completion_tokens`).
   - [`formatting.py`](../backend/src/aie_estimator/foundation/llm/formatting.py): Normalizes output dictionaries across engines for any validated Pydantic `BaseModel` (`PmEstimation` by default).
   - [`pricing.py`](../backend/src/aie_estimator/foundation/llm/pricing.py): Fallback token pricing estimation via `MODEL_COSTS` when upstream routers omit cost metadata.
2. **Structured Extraction with Instructor**: The service enforces typed extraction into `response_model` (defaulting to [`PmEstimation`](../backend/src/aie_estimator/domain/schemas/estimations.py), and parameterized for tier schemas or auxiliary calls such as [`MetadataExtractionResult`](../backend/src/aie_estimator/generation/conversation/models.py)). If validation fails (such as phase cost mismatch), Instructor feeds errors back to the LLM up to `max_retries`. Exception unwrapping via `unwrap_exception` ([`exceptions.py`](../backend/src/aie_estimator/domain/exceptions.py)) inspects root causes before raising `LLMProviderError`.
3. **Multi-Layer Resilience Hierarchy**:
   - **Layer 1 (Transport)**: HTTP client retries and exponential backoff (`httpx` / vendor SDKs).
   - **Layer 2 (Schema Adherence)**: Instructor validation re-prompting loop.
   - **Layer 3 (Parameter Recovery)**: Automatic parameter stripping on vendor rejections.
   - **Layer 4 (Dynamic Failover)**: LiteLLM intra-tier model rotation and cross-tier escalation (`reasoner-fast` → `reasoner-deep`). See [`docs/reasoning-profiles.md`](reasoning-profiles.md).
4. **Execution Pipeline Order**: `sanitize_input` runs *before* cache lookups to prevent invalid prompts or PII from querying or polluting cache stores. Dual-write persistence occurs only after output filtering passes. See [`docs/guardrails.md`](guardrails.md) and [`docs/semantic-response-caching.md`](semantic-response-caching.md).
5. **Conversational Attachment & Post-Turn Memory Refresh**: [`EstimationService.generate_session_estimation`](../backend/src/aie_estimator/domain/estimations_service.py) validates attachment counts, runs strict `sanitize_input` on the raw user transcript, extracts and sanitizes optional `(filename, bytes)` attachments via [`extract_text`](../backend/src/aie_estimator/foundation/attachments/extractor.py) and [`sanitize_attachment_input`](../backend/src/aie_estimator/foundation/guardrails/input.py) (redacting PII inline while blocking moderation/prompt-injection violations), and wraps non-empty attachments in `<attachments>` XML blocks via `enrich_transcript`. It bypasses Tier 1 and Tier 2 caches, appends the enriched turn to `session.history`, and invokes [`update_metadata`](../backend/src/aie_estimator/generation/conversation/metadata_extractor.py). Compression and extractor token usage and cost aggregate into `session.last_turn_observed` ([`TurnObservation`](../backend/src/aie_estimator/generation/conversation/models.py)), preserving `cost_usd=None` whenever primary generation is unpriced, while [`EstimationResponse`](../backend/src/aie_estimator/domain/schemas/estimations.py) retains generation-only `usage` and `cost_usd`.
