# Model Catalog and Request Validation

## Overview

[`src/aie_estimator/model_catalog.py`](../src/aie_estimator/model_catalog.py) manages model specifications, logical routing profiles, and request boundary validation based on configuration defined in [`src/aie_estimator/llm_profiles.yml`](../src/aie_estimator/llm_profiles.yml).

| Model / Profile Identifier | Provider | Reasoning Capable | Logical Profile | Scope |
| :--- | :--- | :---: | :---: | :--- |
| `reasoner-fast` | `litellm` | Yes | Yes | Tier-1 cost-optimized reasoning (GPT-6 Luna, Gemini Flash Lite, Claude Haiku 5.5, Mistral Small, Scaleway DeepSeek). |
| `reasoner-deep` | `litellm` | Yes | Yes | Tier-2 deep reasoning (GPT-6 Luna High, Gemini Flash, Claude Sonnet 5.5, Nebius DeepSeek R2). |
| `gpt-6-luna` | `openai` | Yes | No | OpenAI reasoning model with qualitative reasoning effort. |
| `claude-haiku-5-5` | `anthropic` | Yes | No | Anthropic reasoning model with adaptive thinking. |

---

## Usage Examples

### 1. Specifying Model or Profile in Request Payload

```json
{
  "description": "Scalable notification service supporting email and SMS.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "model": "reasoner-fast",
  "engine": "litellm"
}
```
*Note*: If `model` is omitted, the validator defaults to `settings.llm_model`. Invalid models return HTTP 422 with supported model options.

### 2. Inspecting Catalog Programmatically

```python
from aie_estimator.model_catalog import get_model_catalog, get_supported_models

catalog = get_model_catalog()
supported_models = get_supported_models()
spec = catalog["reasoner-fast"]
print(spec.provider, spec.is_reasoning, spec.is_profile)
```

---

## Technical Considerations

1. **Decoupled Metadata Layer**: Situated in [`model_catalog.py`](../src/aie_estimator/model_catalog.py), separating domain schemas ([`schemas/estimations.py`](../src/aie_estimator/schemas/estimations.py)) from provider infrastructure ([`services/providers/`](../src/aie_estimator/services/providers/)) to eliminate circular imports.
2. **Leaf Constants**: Canonical defaults (`DEFAULT_FAST_PROFILE`, `DEFAULT_DEEP_PROFILE`, `LOGICAL_PROFILE_NAMES`) reside in [`src/aie_estimator/utils/model_constants.py`](../src/aie_estimator/utils/model_constants.py), a leaf module with zero internal imports.
3. **Single Cache Architecture**: `get_model_catalog()` uses `@lru_cache(maxsize=4)`. `get_supported_models()` derives directly from the cached catalog to prevent cache drift.
4. **OpenAPI Schema Enrichment**: `Server._setup_openapi()` in [`src/aie_estimator/server.py`](../src/aie_estimator/server.py) dynamically injects supported models into the OpenAPI schema for interactive dropdown selection in Swagger UI (`/docs`).
