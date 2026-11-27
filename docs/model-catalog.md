# Model Catalog and Request Validation

## Overview

[`model_catalog.py`](../backend/src/aie_estimator/model_catalog.py) manages model specifications, logical routing profiles, and request boundary validation based on configuration defined in [`llm_profiles.yml`](../backend/src/aie_estimator/llm_profiles.yml).

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
*Note*: If `model` is omitted under `litellm`, it defaults to `settings.llm_model`; under `native`, it defaults to the first native model (`gpt-6-luna`). Incompatible engine/model pairs (such as `engine="litellm"` with `"gpt-6-luna"` or `engine="native"` with `"reasoner-fast"`) return HTTP 422.

### 2. Inspecting Catalog Programmatically

```python
from aie_estimator.model_catalog import get_model_catalog, get_models_for_engine, get_supported_models

catalog = get_model_catalog()
supported_models = get_supported_models()
litellm_profiles = get_models_for_engine("litellm")
native_models = get_models_for_engine("native")
spec = catalog["reasoner-fast"]
print(spec.provider, spec.is_reasoning, spec.is_profile)
```

---

## Technical Considerations

1. **Decoupled Metadata Layer**: Situated in [`model_catalog.py`](../backend/src/aie_estimator/model_catalog.py), separating domain schemas ([`domain/schemas/estimations.py`](../backend/src/aie_estimator/domain/schemas/estimations.py)) from provider infrastructure ([`foundation/llm/`](../backend/src/aie_estimator/foundation/llm/)) to eliminate circular imports.
2. **Leaf Constants**: Canonical defaults (`DEFAULT_FAST_PROFILE`, `DEFAULT_DEEP_PROFILE`, `LOGICAL_PROFILE_NAMES`) reside in [`constants.py`](../backend/src/aie_estimator/utils/constants.py), a leaf module with zero internal imports.
3. **Engine-Scoped Filtering & Validation**: `get_models_for_engine(engine)` returns logical profiles (`is_profile=True`) for `"litellm"` and physical models (`is_profile=False`) for `"native"`. `resolve_engine_model(engine, model)` in [`model_catalog.py`](../backend/src/aie_estimator/model_catalog.py) validates compatibility and resolves omitted defaults, serving as the single source of truth for [`domain/schemas/estimations.py`](../backend/src/aie_estimator/domain/schemas/estimations.py), [`sessions_router.py`](../backend/src/aie_estimator/api/sessions_router.py), and [`../backend/streamlit_app.py`](../backend/streamlit_app.py).
4. **Single Cache Architecture**: `get_model_catalog()` uses `@lru_cache(maxsize=4)`. `get_supported_models()` and `get_models_for_engine()` derive directly from the cached catalog to prevent cache drift.
5. **OpenAPI Schema Enrichment**: `Server._setup_openapi()` in [`server.py`](../backend/src/aie_estimator/server.py) dynamically injects supported models into the OpenAPI schema for interactive dropdown selection in Swagger UI (`/docs`).
