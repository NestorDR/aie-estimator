# Actor-Critic-Boss Role Composition

## Overview

The Actor-Critic-Boss (ACB) pipeline decouples initial estimation generation (**Actor**), structured defect auditing (**Critic**), and bounded deterministic governance (**Boss**) on conversational sessions via `POST /api/v1/sessions/{session_id}/estimate-acb`.

- **Schemas**: [acb.py](../backend/src/aie_estimator/schemas/acb.py) (`CriticFeedback`, `CriticIssue`, `ACBIteration`, `BossTrace`, `ACBResponse`)
- **Services**: [critic_service.py](../backend/src/aie_estimator/services/critic_service.py), [boss_service.py](../backend/src/aie_estimator/services/boss_service.py), [estimations_service.py](../backend/src/aie_estimator/services/estimations_service.py)
- **Prompts**: [critic/v1/system.j2](../backend/src/aie_estimator/prompts/critic/v1/system.j2), [critic/v1/user.j2](../backend/src/aie_estimator/prompts/critic/v1/user.j2), [estimation/v3/user.j2](../backend/src/aie_estimator/prompts/estimation/v3/user.j2), [loader.py](../backend/src/aie_estimator/prompts/loader.py)
- **Router & DI**: [sessions_router.py](../backend/src/aie_estimator/routers/sessions_router.py), [dependencies.py](../backend/src/aie_estimator/dependencies.py), [settings.py](../backend/src/aie_estimator/settings.py)

## Usage Examples

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/${SESSION_ID}/estimate-acb" \
  -H "X-User-Tier: pm" \
  -F "transcript=Build a multi-tenant B2B analytics platform with FastAPI and PostgreSQL." \
  -F "project_type=web_saas" \
  -F "detail_level=medium" \
  -F "output_format=phases_table"
```

The response extends `EstimationResponse` with `boss_trace` (`iterations`, `final_decision`, `iterations_run`) and generation `usage` and `cost_usd` aggregated across all Actor and Critic calls (preserving `cost_usd=None` when any Actor call is unpriced), while post-turn compression and metadata extraction tokens and costs roll into `session.last_turn_observed` ([`TurnObservation`](../backend/src/aie_estimator/sessions/models.py)).

## Technical Considerations

- **Dedicated Opt-In Endpoint**: `POST /api/v1/sessions/{session_id}/estimate-acb` keeps single-pass `POST /api/v1/sessions/{session_id}/estimate` and `POST /api/v1/estimate` isolated from 3x+ ACB latency and token cost.
- **Tier-Agnostic Auditing**: `render_critic_prompt` serializes `BaseTierEstimation.model_dump_json(indent=2)` so a single `critic/v1` template audits `DeveloperEstimation`, `PmEstimation`, and `ExecutiveEstimation` uniformly.
- **Anti-Loop Invariants**: `CriticFeedback.iteration_requires_blocking_issue` rejects `needs_iteration` without at least one `critical` or `major` issue, preventing minor-only iteration loops. `Boss` runs deterministically within `Settings.acb_max_iterations` (default `2`, bounds `1..5`).
- **Fallback Synthesis & Graceful Degradation**: On `reject` or budget exhaustion, `Boss._synthesize_fallback` prepends open Critic caveats to `summary` and floors `confidence_pct` at `LOW_CONFIDENCE_THRESHOLD` (`30`) so valid-with-caveats drafts do not trip the `Out of scope:` prefix validator. If `Critic.review` encounters a provider or validation error, it degrades to a synthetic `accept` with `confidence_in_review=0`.
- **Uniform Fast Profile**: Both Actor and `Settings.acb_critic_model` default to `DEFAULT_FAST_PROFILE` (`"reasoner-fast"`).
- **UI Consumption**: [streamlit_app.py](../backend/streamlit_app.py) provides an opt-in `Actor-Critic-Boss review` toggle (default off) and renders `BossTrace` (`final_decision`, `iterations_run`, `issue_summary`) in a collapsed expander. See [Conversational Estimation UI](streamlit-conversational-ui.md).
