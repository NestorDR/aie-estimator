# User Tier Adaptation

## Overview

`aie-estimator` adapts both prompt framing and structured output contracts to the active stakeholder profile (`developer`, `pm`, or `executive`) via the `s05.3` Tier pattern:

- **Tier Resolution & Propagation**: [tier_resolver.py](../backend/src/aie_estimator/sessions/tier_resolver.py) defines `Tier` and `resolve_tier`, while [dependencies.py](../backend/src/aie_estimator/dependencies.py) (`get_caller_tier`) parses the optional `X-User-Tier` HTTP header.
- **Declarative Registry**: [tier_config.py](../backend/src/aie_estimator/prompts/tier_config.py) maps each `Tier` in `TIER_CONFIG` to its `v3` template and Pydantic output schema.
- **Tier Schemas & Guardrails**: [estimations.py](../backend/src/aie_estimator/schemas/estimations.py) defines `BaseTierEstimation`, `DeveloperEstimation`, `PmEstimation`, and `ExecutiveEstimation`; [output.py](../backend/src/aie_estimator/guardrails/output.py) (`enforce_scope_response`) normalizes low-confidence (`< 30`) outputs per tier schema.
- **Modular `v3` Templates**: [src/aie_estimator/prompts/estimation/v3/](../backend/src/aie_estimator/prompts/estimation/v3/) provides `developer.j2`, `pm.j2`, `executive.j2`, and `user.j2` sharing `{% include %}` partials (`_project_metadata.j2`, `_scope_rates.j2`, `_rules.j2`, `_examples.j2`), rendered under `StrictUndefined` by [loader.py](../backend/src/aie_estimator/prompts/loader.py).

## Usage Examples

Stateless estimation with an explicit tier override:

```json
POST /api/v1/estimate
{
  "description": "Build a multi-tenant SaaS billing portal with Stripe.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table",
  "tier": "executive"
}
```

Trusted header propagation on a conversational session turn:

```bash
curl -X POST http://localhost:8000/api/v1/sessions/<session_id>/estimate \
  -H "X-User-Tier: developer" \
  -F "transcript=Design the Kubernetes and Kafka event ingestion pipeline." \
  -F "project_type=data_pipeline" \
  -F "detail_level=detailed" \
  -F "output_format=line_items"
```

## Technical Considerations

- **Precedence Chain**: `resolve_tier` evaluates the `X-User-Tier` header (`"caller_context"`), explicit request `tier` (`"explicit_override"`), ordered transcript/metadata rules (`nda_detected`, `regulatory_context`, `technical_audience`, `small_team_pm`), and falls back to `Tier.PM` (`"default"`).
- **Distinct Contracts Over Optional Branching**: Each tier enforces its own Pydantic schema (`DeveloperEstimation`, `PmEstimation`, `ExecutiveEstimation`) rather than a single schema with optional fields, preventing models from leaking cross-persona fields.
- **Cache Partitioning**: [exact.py](../backend/src/aie_estimator/services/cache/exact.py) includes `tier` in the SHA-256 cache key, and [semantic.py](../backend/src/aie_estimator/services/cache/semantic.py) partitions RediSearch buckets as `{version}_{tier}_{project_type}_{detail_level}_{output_format}` and validates hits against `tier_config.schema`.
- **Session Tracking**: [models.py](../backend/src/aie_estimator/sessions/models.py) (`Session`) and [sessions.py](../backend/src/aie_estimator/schemas/sessions.py) (`SessionInfoResponse`) record `last_tier`, and [metadata_extractor.py](../backend/src/aie_estimator/sessions/metadata_extractor.py) accepts any `BaseTierEstimation` subclass.
- **UI Consumption**: [streamlit_app.py](../backend/streamlit_app.py) exposes `Stakeholder tier` (`Auto` or explicit `tier` form override, never `X-User-Tier`) and renders tier-specific responses via `_TIER_RENDERERS`. See [Conversational Estimation UI](streamlit-conversational-ui.md).

