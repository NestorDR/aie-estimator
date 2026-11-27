# Conversational Sessions

## Overview

The conversational session subsystem separates bounded raw turn history ([`ConversationHistory`](../backend/src/aie_estimator/generation/conversation/models.py)) from distilled project memory ([`ProjectMetadata`](../backend/src/aie_estimator/generation/conversation/models.py)) inside an in-memory [`SessionStore`](../backend/src/aie_estimator/generation/conversation/store.py), exposed through [`sessions_router`](../backend/src/aie_estimator/api/sessions_router.py) under `/api/v1/sessions`, and refreshes project memory after each turn via [`metadata_extractor`](../backend/src/aie_estimator/generation/conversation/metadata_extractor.py).

## Usage Examples

### Create, run a multi-turn estimation, and inspect a session over HTTP

```http
POST /api/v1/sessions HTTP/1.1
```

```json
{"session_id": "2c8d38d2-3c0e-4f08-9e27-0f0448a5b9dc"}
```

```http
POST /api/v1/sessions/2c8d38d2-3c0e-4f08-9e27-0f0448a5b9dc/estimate HTTP/1.1
Content-Type: multipart/form-data; boundary=---boundary

-----boundary
Content-Disposition: form-data; name="transcript"

Estimate MyProject MVP with FastAPI and PostgreSQL for a 3-developer team.
-----boundary
Content-Disposition: form-data; name="project_type"

web_saas
-----boundary
Content-Disposition: form-data; name="detail_level"

medium
-----boundary
Content-Disposition: form-data; name="output_format"

phases_table
-----boundary
Content-Disposition: form-data; name="attachments"; filename="spec.md"
Content-Type: text/markdown

# Requirements
Include OAuth2 SSO and role-based access control.
-----boundary--
```

```http
GET /api/v1/sessions/2c8d38d2-3c0e-4f08-9e27-0f0448a5b9dc HTTP/1.1
```

### Append turns and merge metadata in Python

```python
from aie_estimator.generation.conversation.models import MetadataExtractionResult
from aie_estimator.generation.conversation.store import SessionStore

store = SessionStore(max_turns=6, ttl_seconds=86_400)
session = store.create()

session.history.append(user="Estimate MyProject MVP.", assistant="Estimated 6 weeks.")
session.metadata = session.metadata.merge_with(
    MetadataExtractionResult(
        project_name="MyProject",
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        rejected_options=["Rails", "Microservices"],
    )
)
```

## Technical Considerations

1. **History vs. Memory Separation**: [`Message`](../backend/src/aie_estimator/generation/conversation/models.py) constrains `role` to `"user"` or `"assistant"`, keeping the system prompt outside [`ConversationHistory`](../backend/src/aie_estimator/generation/conversation/models.py) so each turn can re-render [`ProjectMetadata`](../backend/src/aie_estimator/generation/conversation/models.py) dynamically without duplicating system text.
2. **Hybrid History Compression with Anchors**: [`ConversationHistory.append`](../backend/src/aie_estimator/generation/conversation/models.py) appends pairs without inline eviction, delegating overflow handling to [`CompressionPolicy.apply`](../backend/src/aie_estimator/generation/conversation/compression/policy.py). When `messages` exceeds `session_max_turns * 2`, [`AnchorDetector`](../backend/src/aie_estimator/generation/conversation/compression/anchors.py) (`heuristic` regex default or opt-in `llm` classifier) promotes durable commitment turns to `anchors` (capped at `session_max_anchor_pairs`, default `4`), while non-anchor overflow and overflow anchors are folded into `summary` by [`CumulativeSummarizer`](../backend/src/aie_estimator/generation/conversation/compression/summarizer.py) (`conversation_summary/v1`) with graceful fallback on error. Compression token usage and cost aggregate into `session.last_turn_observed` ([`TurnObservation`](../backend/src/aie_estimator/generation/conversation/models.py), preserving `cost_usd=None` when primary generation is unpriced) without mutating `TokenUsage` or `EstimationResponse.cost_usd`, and `GET /api/v1/sessions/{session_id}` ([`SessionInfoResponse`](../backend/src/aie_estimator/domain/schemas/sessions.py)) exposes `turn_index`, `has_summary`, `summary_chars`, `summary`, `anchors_count`, `anchors`, `last_tier_rule`, and `last_turn_observed`.
3. **Metadata Merge & Policy 1 Retraction Semantics**: [`ProjectMetadata.merge_with`](../backend/src/aie_estimator/generation/conversation/models.py) consumes [`MetadataExtractionResult`](../backend/src/aie_estimator/generation/conversation/models.py) to overwrite scalars on non-empty values, prune rejected options from `mentioned_technologies`, un-reject entries via `retracted_options` or re-adopted technologies, remove `retracted_constraints` from `explicit_constraints`, and merge remaining list items via case-insensitive union preserving first-seen casing.
4. **Hybrid Post-Turn Metadata Extraction**: [`update_metadata`](../backend/src/aie_estimator/generation/conversation/metadata_extractor.py) runs an async structured LLM extraction call (`metadata_extraction/v1`) followed by deterministic post-validation (`sanitize_extraction_delta` stripping whitespace, clearing `"null"`-like literals, and deduplicating lists). Under `litellm`, extraction targets `Settings.metadata_extractor_model` (`reasoner-fast`); under `native`, it falls back to `opts.effective_model`. Extraction is skipped when `enable_metadata_extraction=False` or `confidence_pct < 30` (`Out of scope:`), and any failure logs `metadata_extraction_failed` while returning prior `ProjectMetadata` unchanged inside [`MetadataExtractionOutcome`](../backend/src/aie_estimator/generation/conversation/metadata_extractor.py).
5. **Multipart Form Contract & Fast-Fail Ordering**: [`estimate_in_session`](../backend/src/aie_estimator/api/sessions_router.py) (`POST /api/v1/sessions/{session_id}/estimate`) validates `transcript` (`20..2000` characters after trimming) and `model` ([`resolve_engine_model`](../backend/src/aie_estimator/domain/schemas/estimations.py)) via `Annotated[..., Form()]`, resolves the session via `store.get_or_404(session_id)`, and validates attachment count via [`validate_attachment_count`](../backend/src/aie_estimator/foundation/attachments/extractor.py) before reading upload bytes or calling [`EstimationService.generate_session_estimation`](../backend/src/aie_estimator/domain/estimations_service.py).
6. **Lazy TTL Eviction & Process Volatility**: [`SessionStore`](../backend/src/aie_estimator/generation/conversation/store.py) tracks `last_accessed_at` per session against `Settings.session_ttl_seconds` (default `86_400`), evicting expired sessions lazily in `get_or_404` and sweeping unqueried expired entries during `create()`. State lives in a single-process dictionary (`uvicorn --workers=1`) and resets on restart until replaced by a persistent store behind the same interface.
7. **Singleton Wiring & Centralized Error Mapping**: [`get_session_store`](../backend/src/aie_estimator/dependencies.py) caches the process-local [`SessionStore`](../backend/src/aie_estimator/generation/conversation/store.py) via `@lru_cache(maxsize=1)`, and [`Server`](../backend/src/aie_estimator/server.py) maps [`SessionNotFoundError`](../backend/src/aie_estimator/generation/conversation/store.py) to HTTP `404` and [`AttachmentExtractionError`](../backend/src/aie_estimator/foundation/attachments/extractor.py) to HTTP `422`.
8. **Conversational UI Client**: [`../backend/streamlit_app.py`](../backend/streamlit_app.py) consumes these endpoints via lazy session creation and transparent 404 recovery, retaining the visible conversation thread client-side because `GET /api/v1/sessions/{session_id}` omits message bodies; see [Conversational Estimation UI](streamlit-conversational-ui.md).
