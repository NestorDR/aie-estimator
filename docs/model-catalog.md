# Model Catalog and Request Validation

## Overview

The `aie-estimator` service utilizes an externalized, dynamic model catalog defined in `src/aie_estimator/llm_profiles.yml` and managed via `src/aie_estimator/model_catalog.py`, which co-locates the catalog registry, vendor model defaults, and the `ModelSpec` dataclass.

The catalog serves four primary architectural purposes:
1. **Request Boundary Validation**: Validates client request payloads against `get_supported_models()`, accepting both logical routing profiles (`reasoner-fast`, `reasoner-deep`) and direct physical models (`gpt-6-luna`, `claude-haiku-5-5`), rejecting invalid or placeholder inputs with HTTP 422 Unprocessable Entity.
2. **Dynamic Configuration Externalization**: Defines models, provider parameters, reasoning effort, token budgets, and fallback tiers in YAML rather than hardcoded Python dictionaries.
3. **Logical Abstraction Over Physical Models**: Provides abstract profiles (`reasoner-fast`, `reasoner-deep`) that allow callers to express reasoning intent while the router dynamically balances cost and latency across upstream providers.
4. **Structured Capability Mapping**: Each catalog entry is represented by a `ModelSpec` dataclass specifying `provider`, `is_reasoning`, and `is_profile`.

---

## Supported Models and Logical Profiles

| Model Identifier | Provider | Reasoning Capable | Logical Profile | Purpose / Target Scope |
| :--- | :--- | :---: | :---: | :--- |
| `reasoner-fast` | `litellm` | Yes | Yes | Tier-1 cost-optimized reasoning (GPT-6 Luna, Gemini 3.5 Flash Lite, Claude Haiku 5.5, Mistral Small, Scaleway DeepSeek V4 Flash). |
| `reasoner-deep` | `litellm` | Yes | Yes | Tier-2 deep synthesis reasoning (GPT-6 Luna High, Gemini 3.8 Flash, Claude Sonnet 5.5, Nebius DeepSeek R2). |
| `gpt-6-luna` | `openai` | Yes | No | Default OpenAI reasoning model with qualitative reasoning effort. |
| `claude-haiku-5-5` | `anthropic` | Yes | No | Anthropic reasoning model with adaptive thinking and qualitative effort support. |

---

## Usage Examples

### 1. Specifying Model or Profile in API Payload

Clients can pass either a logical profile or a physical model identifier in `POST /api/v1/estimate`:

```json
{
  "description": "We need a scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "model": "reasoner-fast",
  "engine": "litellm"
}
```

If `model` is omitted or passed as `null`, the request validator defaults it to `settings.llm_model` (e.g. `"reasoner-fast"`).

If an unsupported model string is provided:

```json
{
  "description": "We need a scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "model": "unsupported-model-name"
}
```

The endpoint immediately returns HTTP 422:

```json
{
  "detail": "Validation Error",
  "errors": [
    {
      "type": "value_error",
      "loc": ["body", "model"],
      "msg": "Value error, Input should be one of: reasoner-fast, reasoner-deep, gpt-6-luna, claude-haiku-5-5"
    }
  ]
}
```

### 2. Inspecting Catalog Programmatically

```python
from aie_estimator.model_catalog import (
    ModelSpec,
    get_model_catalog,
    get_model_provider,
    get_supported_models,
)
from aie_estimator.utils.model_constants import DEFAULT_FAST_PROFILE, LOGICAL_PROFILE_NAMES

# Dynamic accessor functions
catalog = get_model_catalog()
supported = get_supported_models()

spec = catalog[DEFAULT_FAST_PROFILE]
print(spec.is_profile)  # True
print(spec.provider)  # "litellm"
print(spec.is_reasoning)  # True

provider = get_model_provider("gpt-6-luna")
print(provider)  # "openai"
```

---

## Technical Considerations

### Architectural Layering and Decoupling
The model catalog is situated in `src/aie_estimator/model_catalog.py`, establishing it as a domain-level metadata service. This decouples domain schemas (`src/aie_estimator/schemas/estimations.py`) from provider infrastructure (`src/aie_estimator/services/providers/`), eliminating circular dependency cycles during module initialization.

### Leaf Constants
Low-level model constants (`DEFAULT_FAST_PROFILE`, `DEFAULT_DEEP_PROFILE`, `DEFAULT_GPT_LLM`, `DEFAULT_CLAUDE_LLM`, `LOGICAL_PROFILE_NAMES`, and `PROVIDER_PREFIXES`) reside in `src/aie_estimator/utils/model_constants.py`. This module functions as an independent leaf node with zero internal imports, enabling `settings.py`, `model_catalog.py`, and provider modules to import constants safely without mutual coupling.

### Settings Configuration and Path Injection
- **Configuration Path**: `Settings.llm_profiles_path` in `src/aie_estimator/settings.py` resolves the default path to `llm_profiles.yml` and can be overridden via the `LLM_PROFILES_PATH` environment variable.
- **Dependency Injection**: Accessor functions (`get_model_catalog`, `get_supported_models`, `get_model_provider`) accept an optional `profiles_path: Path | str | None = None` parameter, enabling test suites and dynamic callers to inject custom configuration files directly without monkeypatching internal filesystem helpers.

### Single-Cache Architecture
- `get_model_catalog()` is decorated with `@lru_cache(maxsize=4)` keyed on the resolved profiles path.
- `get_supported_models()` derives directly from `get_model_catalog()` without an independent cache, preventing dual-cache desynchronization when cache invalidation occurs.

### OpenAPI Schema Enrichment
FastAPI documentation for `EstimationRequest.model` is decoupled from Pydantic schema field definitions. `Server._setup_openapi()` in `src/aie_estimator/server.py` customizes `app.openapi` to dynamically inject the supported catalog list along with `None` into the OpenAPI schema definition, rendering an interactive `<select>` dropdown in Swagger UI (`/docs`).

### Configuration Fallback
If `llm_profiles.yml` is missing or temporarily corrupted, `get_model_catalog()` logs a warning and returns `_DEFAULT_CATALOG` to prevent startup panics or cascading service crashes.
