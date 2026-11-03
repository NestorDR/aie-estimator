# Model Catalog and Request Validation

## Overview

The `aie-estimator` service utilizes an externalized, dynamic model catalog defined in `src/aie_estimator/llm_profiles.yml` and managed via `src/aie_estimator/services/model_catalog.py`, while `src/aie_estimator/schemas/models.py` defines pure data structures (`ModelSpec`).

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
  "transcription": "Client: We need a scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "model": "reasoner-fast",
  "engine": "litellm"
}
```

If `model` is omitted or passed as `null`, the request validator defaults it to `settings.llm_model` (e.g. `"gpt-6-luna"`).

If an unsupported model string is provided:

```json
{
  "transcription": "Valid transcription text...",
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
from aie_estimator.schemas.models import DEFAULT_FAST_PROFILE, ModelSpec
from aie_estimator.services.model_catalog import (
    PROVIDER_PREFIXES,
    get_model_catalog,
    get_model_provider,
    get_supported_models,
)

# Dynamic accessor functions
catalog = get_model_catalog()
supported = get_supported_models()

spec = catalog[DEFAULT_FAST_PROFILE]
print(spec.is_profile)   # True
print(spec.provider)     # "litellm"
print(spec.is_reasoning) # True

provider = get_model_provider("gpt-6-luna")
print(provider)          # "openai"
```

---

## Technical Considerations

### Separation of Concerns
Data structures live strictly in `src/aie_estimator/schemas/models.py`. File reading, YAML parsing, caching, and provider mapping reside in `src/aie_estimator/services/model_catalog.py`, eliminating I/O side effects from schema definitions.

### Singleton Caching
- **Model Catalog**: `get_model_catalog()` and `get_supported_models()` are decorated with `@lru_cache(maxsize=1)` to avoid repeated filesystem and YAML parsing overhead during request processing.
- **LiteLLM Router**: `get_litellm_router()` in `src/aie_estimator/services/litellm_provider.py` caches the initialized `Router` instance with `@lru_cache(maxsize=1)`.
- **Application Settings**: `get_settings()` in `src/aie_estimator/settings.py` caches the loaded `Settings` instance with `@lru_cache`.

### Configuration Fallback
If `src/aie_estimator/llm_profiles.yml` is missing or temporarily unreadable, `get_model_catalog()` gracefully returns `_DEFAULT_CATALOG` to prevent startup panics or cascading service crashes.
