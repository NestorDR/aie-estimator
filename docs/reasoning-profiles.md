# LLM Reasoning Parameter Normalization and Profiles

## Overview

Modern LLM providers (Anthropic Claude, Google Gemini, OpenAI GPT-6) use divergent APIs for controlling thinking depth and reasoning budgets. The reasoning subsystem provides a provider-agnostic normalization layer:

1. **Qualitative Reasoning Effort**: Standardizes reasoning intensity into `"low"`, `"medium"`, and `"high"` via [`GenerationOptions`](../src/aie_estimator/services/providers/generation_options.py).
2. **Provider Translation**:
   - **Anthropic Claude**: Adaptive thinking (`{"type": "adaptive"}`, `output_config.effort: "medium"`), strictly suppressing sampling `temperature`.
   - **Google Gemini**: Maps `reasoning_effort` to `generationConfig.thinkingConfig.thinkingLevel` via LiteLLM.
   - **OpenAI / Router Models**: Employs top-level `reasoning_effort` (`"low"`, `"medium"`, `"high"`).
3. **Logical Profiles**: Externalized in [`src/aie_estimator/llm_profiles.yml`](../src/aie_estimator/llm_profiles.yml): `reasoner-fast` (Tier 1 cost-optimized reasoning) and `reasoner-deep` (Tier 2 frontier reasoning).

---

## Usage Examples

### 1. Request with Logical Profile (`POST /api/v1/estimate`)

```json
{
  "description": "Scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "model": "reasoner-deep",
  "engine": "litellm"
}
```

### 2. Programmatic Dispatch with `GenerationOptions`

```python
from aie_estimator.services.providers import GenerationOptions, call_llm

opts = GenerationOptions(
    model="reasoner-deep",
    engine="litellm",
    reasoning_effort="high",
)
response = await call_llm(
    user_prompt="<project_description>Details...</project_description>",
    system_prompt="Estimation instructions...",
    opts=opts,
)
```

---

## Technical Considerations

1. **Cache Key Isolation**: `EstimationExactCache.make_key()` incorporates `reasoning_effort` into its SHA-256 digest, preventing `"low"` effort requests from colliding with `"high"` effort cache entries.
2. **Two-Level Resilience & Runtime Failover** (configured in [`src/aie_estimator/llm_profiles.yml`](../src/aie_estimator/llm_profiles.yml)):
   - **Level 1 (Intra-Tier Provider Rotation)**: LiteLLM rotates traffic to the lowest-cost healthy candidate within `reasoner-fast`. On HTTP 429, 5xx, or 30s timeouts, it retries against the next candidate provider up to `num_retries: 2`.
   - **Level 2 (Cross-Tier Escalation)**: If all Tier 1 candidates are exhausted, LiteLLM triggers `router_settings.fallbacks` (`reasoner-fast` → `reasoner-deep`), escalating execution to the frontier reasoning pool.
3. **Startup Credential Pruning**: `get_litellm_router()` in [`src/aie_estimator/services/providers/litellm_provider.py`](../src/aie_estimator/services/providers/litellm_provider.py) inspects required API key environment variables, omitting uncredentialed models so the service boots successfully with partial API credentials.
