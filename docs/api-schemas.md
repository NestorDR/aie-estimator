# API Schemas and Data Contracts

## Overview

[`src/aie_estimator/schemas/estimations.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py) defines the domain data models and validation perimeter for the estimation service using Pydantic v2.

The schema contract governs the core phases of the estimation lifecycle:
1. **Request Intake ([`EstimationRequest`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py))**: Validates project descriptions (20–2000 characters), domain classifications (`ProjectType`, `DetailLevel`, `OutputFormat`), provider mode (`engine="litellm"` or `"native"`), and target models validated against the catalog.
2. **Structured Response Contract ([`EstimationResult`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py) & [`Phase`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py))**: Enforces phase breakdowns, duration, cost, and invariants (`phases_sum_matches_total`, `low_confidence_requires_out_of_scope_prefix`).
3. **Execution Instrumentation ([`TokenUsage`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py))**: Tracks input, output, and preprocessing tokens along with calculated execution cost (`cost_usd`).
4. **Response Delivery ([`EstimationResponse`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py))**: Standardized envelope containing the structured result, active prompt version, cache status, resolved model, provider attribution, and token telemetry.

Consult [`src/aie_estimator/schemas/estimations.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py) for complete field constraints, type annotations, and validator logic.

---

## Usage Examples

### 1. Estimation Request Payload (`POST /api/v1/estimate`)

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

### 2. Estimation Response Contract

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

1. **Autoregressive Field Ordering**: In [`EstimationResult`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py), `phases` precedes `total_duration_weeks` and `total_cost_eur`. This forces the LLM to generate individual phase estimates before computing aggregates, eliminating arithmetic hallucinations.
2. **Schema Invariant Enforcement**: Instructor captures Pydantic validation errors (such as phase cost sum mismatch) and automatically re-prompts the model up to `max_retries`.
3. **Boundary Validation & OpenAPI**: [`EstimationRequest`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/schemas/estimations.py) validates the `model` field against `get_supported_models()` from [`src/aie_estimator/model_catalog.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/model_catalog.py). OpenAPI documentation dropdowns are enriched dynamically via `Server._setup_openapi()` in [`src/aie_estimator/server.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/server.py).
4. **Input Guardrail Violation Mapping**: Semantic violations detected by `check_input` raise `InputGuardrailViolation`, mapped to HTTP 400 with structured detail (`reason`: `"moderation"`, `"prompt_injection"`, or `"pii"`). See [`docs/guardrails.md`](file:///c:/Development/Repos/ai-engineering/aie-estimator/docs/guardrails.md).
