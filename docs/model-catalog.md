# Model Catalog and Request Validation

## Overview

The `aie-estimator` service utilizes a centralized, typed model catalog to manage supported Large Language Models across multiple upstream providers (**OpenAI** and **Anthropic**).

The catalog serves four primary architectural purposes:
1. **Request Boundary Validation**: Enforces strict payload validation via Pydantic and `@field_validator` against `SUPPORTED_MODELS`, rejecting unsupported or placeholder values (such as Swagger's default `"string"`) with HTTP 422 Unprocessable Entity before any service execution or external API calls occur.
2. **OpenAPI / Swagger UI Hardening**: Automatically renders an interactive `<select>` dropdown menu in Swagger UI (`/docs`), preventing invalid free-text placeholders.
3. **Model-Driven Provider Dispatch**: Determines the target upstream provider dynamically based on model metadata (`ModelSpec.provider`), completely eliminating the need for a separate `LLM_PROVIDER` environment configuration.
4. **Structured Capability Mapping**: Replaces brittle substring heuristics (`startswith(...)` or `"-6-" in model`) with an explicit `is_reasoning` capability flag across both OpenAI and Anthropic models.

---

## Supported Models Catalog

| Model Identifier            | Provider    | Reasoning Capable | Notes                                                                            |
|:----------------------------|:------------|:-----------------:|:---------------------------------------------------------------------------------|
| `gpt-4o-mini`               | `openai`    |        No         | Default low-latency OpenAI model for fast estimates.                             |
| `gpt-4o`                    | `openai`    |        No         | Standard high-accuracy multimodal OpenAI model.                                  |
| `o3-mini`                   | `openai`    |        Yes        | Efficient reasoning model.                                                       |
| `gpt-5-nano`                | `openai`    |        Yes        | Advanced OpenAI reasoning model.                                                 |
| `gpt-6-luna`                | `openai`    |        Yes        | Reasoning model; receives `reasoning={"effort": effort}` instead of temperature. |
| `claude-haiku-4-5-20251001` | `anthropic` |        Yes        | Fast Anthropic model with extended thinking budget support.                      |

---

## Usage Examples

### 1. Specifying Model in API Payload

Clients can optionally override the default model in `POST /api/v1/estimate`:

```json
{
  "transcription": "Client: We need a scalable notification service supporting email and SMS. Launch target is within 3 weeks.",
  "model": "gpt-5-nano",
  "thinking_budget": 2048
}
```

If `model` is omitted or passed as `null`, the request validator automatically defaults it to `settings.llm_model` (e.g. `"gpt-4o-mini"`).

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
      "msg": "Value error, Input should be one of: gpt-4o-mini, gpt-4o, o3-mini, gpt-5-nano, gpt-6-luna, claude-haiku-4-5-20251001"
    }
  ]
}
```

### 2. Inspecting Catalog Specs Programmatically

```python
from aie_estimator.schemas.models import MODEL_CATALOG, SUPPORTED_MODELS, ModelSpec

print(SUPPORTED_MODELS)  # ("gpt-4o-mini", "gpt-4o", ...)
spec: ModelSpec = MODEL_CATALOG["gpt-6-luna"]
print(spec.provider)  # "openai"
print(spec.is_reasoning)  # True
```

### 3. Adding a New Model

To register a new model in the catalog:
1. Open `src/aie_estimator/schemas/models.py`.
2. Add a corresponding `ModelSpec` entry to `MODEL_CATALOG`. `SUPPORTED_MODELS`, request schema validation, and Swagger documentation update automatically with zero redundancy.

---

## Technical Considerations

### Boundary Validation vs Upstream 500s
Previously, unvalidated model inputs were forwarded directly to upstream vendor endpoints, causing OpenAI `NotFoundError (404)` exceptions that were masked as internal server errors (HTTP 500). Constraining the request model field using `SUPPORTED_MODELS` with `json_schema_extra` and `@field_validator("model", mode="after")` guarantees that client payload errors fail immediately at the HTTP perimeter while keeping interactive Swagger UI `<select>` menus intact.

### Model-Driven Provider Dispatch
Provider selection is entirely derived from `MODEL_CATALOG`:

```python
model = opts.model or settings.llm_model
spec = MODEL_CATALOG.get(model)
provider = spec.provider if spec is not None else settings.llm_provider
```

The system no longer requires a separate `LLM_PROVIDER` environment variable. Instead, `Settings.llm_provider` is a read-only computed property resolving to `MODEL_CATALOG[settings.llm_model].provider`. Both `extract_requirements` and `generate_estimation` route dynamically according to the requested model.

### Reasoning Parameter Exclusivity
When dispatching to OpenAI's Responses API (`_call_openai`), reasoning models (`spec.is_reasoning == True`) receive `reasoning={"effort": effort}` while `temperature` is omitted to prevent `BadRequestError`.
