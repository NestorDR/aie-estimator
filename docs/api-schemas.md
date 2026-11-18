# API Schemas and Data Contracts

## Overview

[`src/aie_estimator/schemas/estimations.py`](../src/aie_estimator/schemas/estimations.py) defines the domain data models and validation perimeter for the estimation service using Pydantic v2.

The schema contract governs the core phases of the estimation lifecycle:
1. **Request Intake ([`EstimationRequest`](../src/aie_estimator/schemas/estimations.py))**: Validates project descriptions (20–2000 characters), domain classifications (`ProjectType`, `DetailLevel`, `OutputFormat`), provider mode (`engine="litellm"` or `"native"`), optional `tier`, and engine-compatible target models via [`resolve_engine_model`](../src/aie_estimator/model_catalog.py).
2. **Tiered Structured Response Contracts ([`BaseTierEstimation`](../src/aie_estimator/schemas/estimations.py), [`DeveloperEstimation`](../src/aie_estimator/schemas/estimations.py), [`PmEstimation`](../src/aie_estimator/schemas/estimations.py), [`ExecutiveEstimation`](../src/aie_estimator/schemas/estimations.py))**: Enforces stakeholder-specific payloads and invariants (`phases_sum_matches_total` on `PmEstimation`, `min_not_exceeds_max` on ranges, `low_confidence_requires_out_of_scope_prefix` on `BaseTierEstimation`).
3. **Execution Instrumentation ([`TokenUsage`](../src/aie_estimator/schemas/estimations.py))**: Tracks generation input, output, and total tokens along with calculated execution cost (`cost_usd`).
4. **Response Delivery ([`EstimationResponse`](../src/aie_estimator/schemas/estimations.py) & [`ACBResponse`](../src/aie_estimator/schemas/acb.py))**: Standardized envelope containing the tier-typed `result`, `tier`, `tier_rule`, active `prompt_version`, cache status, resolved `model`, `provider` attribution, token telemetry, and (for ACB) `boss_trace`.

Consult [`src/aie_estimator/schemas/estimations.py`](../src/aie_estimator/schemas/estimations.py) for complete field constraints, type annotations, and validator logic. Conversational session contracts (`CreateSessionResponse`, `SessionInfoResponse`) live in [`src/aie_estimator/schemas/sessions.py`](../src/aie_estimator/schemas/sessions.py) and are documented in [`docs/conversational-sessions.md`](conversational-sessions.md).

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
    "total_tokens": 1060
  }
}
```

---

## Technical Considerations

1. **Autoregressive Field Ordering**: In [`PmEstimation`](../src/aie_estimator/schemas/estimations.py), `phases` precedes `total_duration_weeks` and `total_cost_eur`. This forces the LLM to generate individual phase estimates before computing aggregates, eliminating arithmetic hallucinations.
2. **Schema Invariant Enforcement**: Instructor captures Pydantic validation errors (such as phase cost sum mismatch) and automatically re-prompts the model up to `max_retries`.
3. **Engine/Model Boundary Validation & OpenAPI**: [`EstimationRequest`](../src/aie_estimator/schemas/estimations.py) and [`sessions_router`](../src/aie_estimator/routers/sessions_router.py) validate `(engine, model)` via `resolve_engine_model()` backed by `get_models_for_engine()` from [`src/aie_estimator/model_catalog.py`](../src/aie_estimator/model_catalog.py). OpenAPI documentation dropdowns are enriched dynamically via `Server._setup_openapi()` in [`src/aie_estimator/server.py`](../src/aie_estimator/server.py).
4. **Input Guardrail Violation Mapping**: Semantic violations detected by `check_input` raise `InputGuardrailViolation`, mapped to HTTP 400 with structured detail (`reason`: `"moderation"`, `"prompt_injection"`, or `"pii"`). See [`docs/guardrails.md`](guardrails.md).
