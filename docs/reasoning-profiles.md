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

---

## Two-Level Resilience & Runtime Failover

The LiteLLM Router (`src/aie_estimator/services/providers/litellm_provider.py`) operates a two-level resilience strategy configured declaratively in `src/aie_estimator/llm_profiles.yml`:

```
+---------------------------------------------------------------------------------+
|                         TWO-LEVEL RUNTIME FAILOVER                              |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  [LEVEL 1: Intra-Tier Multi-Provider Rotation]                                  |
|  Profile: "reasoner-fast" (Tier 1)                                              |
|  Routing Strategy: cost-based-routing | Timeout: 30.0s | Retries: 2             |
|                                                                                 |
|  Available Candidates (pruned at startup if API key is missing):                |
|    1. OpenAI 6 (openai/gpt-6-luna)                                              |
|    2. Google Gemini (gemini/gemini-3.5-flash-lite)                              |
|    3. Anthropic (claude-haiku-5-5)                                              |
|    4. Mistral AI (mistral/mistral-small-3.2-24b-instruct)                       |
|    5. Scaleway DeepSeek (openai/deepseek-ai/DeepSeek-V4-Flash)                  |
|                                                                                 |
|  Failover Trigger: Rate limit (429), server error (5xx), network timeout.       |
|  Action: LiteLLM Router rotates to the next lowest-cost healthy candidate.      |
|                                                                                 |
+---------------------------------------------------------------------------------+
                                       |
                   (All Tier 1 candidates exhausted / failed)
                                       v
+---------------------------------------------------------------------------------+
|  [LEVEL 2: Cross-Tier Failover Escalation]                                      |
|  Policy: router_settings.fallbacks: - reasoner-fast: ["reasoner-deep"]          |
|                                                                                 |
|  Target Profile: "reasoner-deep" (Tier 2 Frontier Reasoning)                    |
|    1. OpenAI 6 Deep (openai/gpt-6-luna, reasoning_effort: high)                 |
|    2. Google Gemini (gemini/gemini-3.8-flash, reasoning_effort: medium)         |
|    3. Anthropic Claude (claude-sonnet-5-5, effort: medium)                      |
|    4. Nebius DeepSeek (openai/deepseek-ai/DeepSeek-R2)                          |
+---------------------------------------------------------------------------------+
```

### 1. Level 1: Intra-Tier Multi-Provider Rotation

Within an active profile (e.g. `reasoner-fast`), LiteLLM dynamically routes traffic to the lowest-cost healthy provider with valid credentials.
If an upstream provider returns an HTTP 429 (Rate Limit), HTTP 5xx (Server Error), or fails to respond within `timeout: 30.0` seconds, LiteLLM automatically retries against the next available candidate provider in the tier up to `num_retries: 2` without failing the user request.

### 2. Level 2: Cross-Tier Fallback Escalation

If all candidate providers in Tier 1 (`reasoner-fast`) fail or are rate-limited, LiteLLM executes the cross-tier fallback policy defined in `llm_profiles.yml`:
```yaml
router_settings:
  fallbacks:
    - reasoner-fast: ["reasoner-deep"]
```
The router transparently escalates execution to the Tier 2 frontier reasoning pool (`reasoner-deep`).

### 3. Dynamic Credential Pruning

At router initialization, `get_litellm_router()` checks each model's required `api_key` environment variable (e.g. `os.environ/SCALEWAY_API_KEY`).
Any deployment lacking valid credentials in the environment is omitted from the active candidate list. As long as at least one provider has valid API credentials configured, the router initializes successfully and runs without runtime errors.


