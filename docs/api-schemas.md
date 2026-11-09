# API Schemas and Data Contracts

## Overview

`src/aie_estimator/schemas/estimations.py` defines the domain data models and validation perimeter for the estimation service. Built with Pydantic v2, these schemas enforce strict request validation before execution and guarantee deterministic, schema-enforced response contracts across all LLM providers.

The schemas govern the core phases of the estimation lifecycle:
1. **Request Intake (`EstimationRequest`)**: Ingestion, constraint validation, domain classifications (`project_type`, `detail_level`, `output_format`), engine selection (`engine="litellm"` vs `engine="native"`), and model catalog validation.
2. **Execution & Instrumentation (`TokenUsage` and Execution Cost)**: Token consumption accounting across input and output tokens, plus execution cost tracking (`cost_usd`).
3. **Structured Response Contract (`Phase` and `EstimationResult`)**: Rich, schema-validated estimation breakdown enforcing business invariants (cost sum parity and low-confidence refusal prefix).
4. **Response Delivery (`EstimationResponse`)**: Standardized delivery of the validated `EstimationResult`, prompt template version tracking, cache status, provider attribution, and token telemetry.

---

## Schema Reference

### 1. `EstimationRequest`

Payload schema for `POST /api/v1/estimate`:

| Field | Type | Default | Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `description` | `str` | *(required)* | `min_length=20`, `max_length=2000` | Free-text description or transcription of the project to estimate. |
| `project_type` | `ProjectType` | *(required)* | `"mobile_app"`, `"web_saas"`, `"internal_tool"`, `"data_pipeline"` | Project domain category used to contextualize estimation rules. |
| `detail_level` | `DetailLevel` | *(required)* | `"summary"`, `"medium"`, `"detailed"` | Granularity and depth of the task breakdown. |
| `output_format` | `OutputFormat` | *(required)* | `"phases_table"`, `"line_items"`, `"narrative"` | Target presentation layout for the rendered estimation. |
| `model` | `str \| None` | `None` | Must exist in `get_supported_models()` | Target LLM model or logical profile override (`reasoner-fast`, `cost-saver`, etc.). Defaults to `settings.llm_model`. |
| `engine` | `ProviderMode` | `"litellm"` | `"litellm"` or `"native"` | Provider dispatch mode: LiteLLM Router dynamic routing or direct vendor SDK calls. |

### 2. Domain Enums

```python
class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"
```

### 3. `Phase` and `EstimationResult`

Structured estimation data contract emitted by the LLM and validated via Pydantic:

#### `Phase`
Represents an individual phase in the estimation breakdown:
- `name` (`str`, `min_length=1`, `max_length=64`): Phase title or milestone identifier.
- `duration_weeks` (`int`, `ge=1`, `le=52`): Estimated duration in calendar weeks.
- `cost_eur` (`int`, `ge=0`, `le=1_000_000`): Estimated phase cost in EUR.
- `summary` (`str`, `min_length=10`, `max_length=600`): Deliverables and scope summary.

#### `EstimationResult`
Root structured estimation result:
- `summary` (`str`, `min_length=10`, `max_length=1200`): High-level project scope summary or refusal narrative.
- `confidence_pct` (`int`, `ge=0`, `le=100`): Qualitative estimation certainty percentage.
- `phases` (`list[Phase]`, `min_length=1`, `max_length=8`): Ordered delivery phase items.
- `total_duration_weeks` (`int`, `ge=1`, `le=104`): Total delivery duration in calendar weeks.
- `total_cost_eur` (`int`, `ge=0`, `le=2_000_000`): Total delivery cost in EUR.

**Business Invariant Validators:**
1. `phases_sum_matches_total`: Asserts that `sum(phase.cost_eur for phase in phases) == total_cost_eur`.
2. `low_confidence_requires_out_of_scope_prefix`: Asserts that if `confidence_pct < 30`, `summary` must begin with `"Out of scope:"`.

### 4. `EstimationResponse`

Unified response schema returned by `POST /api/v1/estimate`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `result` | `EstimationResult` | Validated, typed estimation breakdown and totals. |
| `prompt_version` | `str` | Prompt template version identifier resolved at runtime (e.g. `"v1"`). |
| `cached` | `bool` | Indicator if the estimation was retrieved from exact-match cache (`False` by default). |
| `model` | `str` | Identifier of the model or profile that produced the estimation. |
| `provider` | `str` | Upstream provider or routing framework (`"litellm"`, `"openai"`, `"anthropic"`). |
| `cost_usd` | `float \| None` | Calculated execution cost in USD reported by the routing engine, or `null`. |
| `usage` | `TokenUsage` | Breakdown of input, output, and sub-call token consumption. |

### 5. Supporting Sub-Schemas

#### `TokenUsage`
Tracks end-to-end token consumption across LLM invocations:
- `input_tokens` (`int`): Main prompt input tokens.
- `output_tokens` (`int`): Estimation completion tokens.
- `total_tokens` (`int`): Sum of main call input and output tokens.
- `preprocessing_input_tokens` (`int`, default `0`): Upstream or preprocessing input tokens.
- `preprocessing_output_tokens` (`int`, default `0`): Upstream or preprocessing completion tokens.

---

## Usage Examples

### 1. Estimation Request with LiteLLM Router

```json
{
  "description": "We need a scalable notification service supporting email and SMS with webhooks and AWS SNS/SES integration. Launch target is within 3 weeks.",
  "project_type": "web_saas",
  "detail_level": "detailed",
  "output_format": "phases_table",
  "model": "reasoner-fast",
  "engine": "litellm"
}
```

Corresponding HTTP 200 response payload:

```json
{
  "result": {
    "summary": "Implementation of a scalable notification service with multi-channel dispatch.",
    "confidence_pct": 85,
    "phases": [
      {
        "name": "Architecture & Ingestion",
        "duration_weeks": 1,
        "cost_eur": 2500,
        "summary": "Webhook receiver endpoints, auth verification, and AWS SNS/SES client setup."
      },
      {
        "name": "Dispatch Pipeline & Templates",
        "duration_weeks": 2,
        "cost_eur": 5000,
        "summary": "Asynchronous job queue, retry backoff, template rendering, and delivery status tracking."
      }
    ],
    "total_duration_weeks": 3,
    "total_cost_eur": 7500
  },
  "prompt_version": "v1",
  "cached": false,
  "model": "openai/gpt-6-luna",
  "provider": "litellm",
  "cost_usd": 0.00034,
  "usage": {
    "input_tokens": 820,
    "output_tokens": 240,
    "total_tokens": 1060,
    "preprocessing_input_tokens": 0,
    "preprocessing_output_tokens": 0
  }
}
```

---

## Technical Considerations

### 1. Autoregressive Field Ordering
In `EstimationResult`, `phases` is declared before `total_duration_weeks` and `total_cost_eur`. In autoregressive generation, this sequence forces the model to commit to individual phase breakdowns first before computing the overall totals, eliminating arithmetic hallucinations on smaller models.

### 2. Post-Validation Re-Prompting via Instructor
When an LLM response violates a model validator (such as phase cost sum mismatch), Instructor captures the `ValueError` and automatically re-prompts the model up to the configured retry threshold (`max_retries`).

### 3. Boundary Validation and OpenAPI Documentation
`EstimationRequest` uses `@field_validator("model", mode="after")` importing `get_supported_models()` from `src/aie_estimator/model_catalog.py` to validate requested model identifiers at the API boundary, accepting physical model IDs and logical profiles while rejecting placeholders and unknown models with HTTP 422. OpenAPI documentation dropdowns are enriched decoupled from the DTO via FastAPI's `app.openapi` customization in `src/aie_estimator/server.py`.

### 4. Input Guardrail Rejection Contract (HTTP 400)
When incoming requests pass syntactic Pydantic validation but violate semantic input guardrails (`check_input`), the application server intercepts `InputGuardrailViolation` and returns an HTTP 400 Bad Request error with structured payload:
```json
{
  "detail": {
    "reason": "prompt_injection",
    "message": "Suspicious instruction-like text detected: 'ignore previous instructions'"
  }
}
```
Possible values for `reason` are `"moderation"`, `"prompt_injection"`, and `"pii"`. See [docs/guardrails.md](guardrails.md) for guardrail layer specifications.
