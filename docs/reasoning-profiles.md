# LLM Reasoning Parameter Normalization and Profiles

## Overview

Modern generative LLM providers (Anthropic Claude 4.5/5.5, Google Gemini 3+, and OpenAI GPT-6/o-series) employ distinct APIs for controlling thinking depth and reasoning deliberation. Previously, divergent parameter formats—such as integer token budgets (`budget_tokens`) and nested configurations (`extra_body.thinkingConfig`)—triggered upstream HTTP 400 parameter rejections and broke automated router fallback chains.

The reasoning parameter normalization subsystem in `aie-estimator` provides a unified, provider-agnostic abstraction:
1. **Qualitative Reasoning Effort**: Standardizes reasoning intensity into three tiers: `"low"`, `"medium"`, and `"high"`.
2. **Provider-Specific Adaptation**:
   - **Anthropic Claude**: Formats requests using adaptive thinking (`{"type": "adaptive"}`) paired with `output_config.effort` (`"medium"` by default), strictly suppressing sampling `temperature`.
   - **Google Gemini**: Uses native `reasoning_effort` (`"low"`, `"high"`) routed via LiteLLM directly into `generationConfig.thinkingConfig.thinkingLevel`, avoiding invalid root-level `thinkingConfig` payloads.
   - **OpenAI / Router Models**: Employs top-level `reasoning_effort` (`"low"`, `"medium"`, `"high"`).
3. **Deterministic Cache Hashing**: `EstimationCache.make_key()` incorporates `reasoning_effort` into Redis cache keys to prevent collisions across distinct reasoning tiers.

---

## Usage Examples

### 1. Request with Qualitative Reasoning Effort

Clients specify the desired logical profile (such as `reasoner-fast` or `reasoner-deep`) in `POST /api/v1/estimate`:

```json
{
  "description": "We need a scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "model": "reasoner-deep",
  "engine": "litellm"
}
```

### 2. Programmatic Generation with GenerationOptions

```python
from aie_estimator.services.estimations_service import generate_estimation
from aie_estimator.services.providers.generation_options import GenerationOptions

# Explicit reasoning effort
opts = GenerationOptions(
    model="reasoner-deep",
    engine="litellm",
    reasoning_effort="high",
)
result = await generate_estimation(
    system_prompt="System instructions with reference examples...",
    user_prompt="<project_description>Project details...</project_description>",
    opts=opts,
)
```

---

## Technical Considerations

### 1. Provider Payload Schemas

- **Anthropic Messages API**:
  ```json
  {
    "model": "claude-haiku-5-5",
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "medium"},
    "max_tokens": 4000
  }
  ```
  Note: When `thinking` is active, Anthropic forbids the `temperature` parameter. The service automatically omits `temperature`.

- **LiteLLM Router & Gemini Models**:
  LiteLLM translates top-level `reasoning_effort` to provider-native schemas:
  - For Gemini: `generationConfig: {thinkingConfig: {thinkingLevel: "low" | "high", includeThoughts: true}}`.
  - For OpenAI: `reasoning_effort: "low" | "medium" | "high"`.

### 2. Cache Invalidation and Key Isolation

`EstimationCache.make_key()` computes SHA-256 digests over all request parameters affecting completion quality:
```python
cache_key = EstimationCache.make_key(
    system_prompt=system_prompt,
    user_message=user_prompt,
    model=model,
    max_tokens=opts.max_tokens,
    reasoning_effort=opts.reasoning_effort,
    engine=opts.engine,
)
```
Requests configured with `"low"` reasoning do not hit cached responses generated with `"high"` reasoning, preserving SLA accuracy and cost predictability.

