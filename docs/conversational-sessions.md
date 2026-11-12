# Conversational Sessions

## Overview

The conversational session subsystem separates bounded raw turn history ([`ConversationHistory`](../src/aie_estimator/sessions/models.py)) from distilled project memory ([`ProjectMetadata`](../src/aie_estimator/sessions/models.py)) inside an in-memory [`SessionStore`](../src/aie_estimator/sessions/store.py), exposed through [`sessions_router`](../src/aie_estimator/routers/sessions_router.py) under `/api/v1/sessions`, and refreshes project memory after each turn via [`metadata_extractor`](../src/aie_estimator/sessions/metadata_extractor.py).

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
from aie_estimator.sessions.models import MetadataExtractionResult
from aie_estimator.sessions.store import SessionStore

store = SessionStore(max_turns=6, ttl_seconds=86_400)
session = store.create()

evicted = session.history.append(user="Estimate MyProject MVP.", assistant="Estimated 6 weeks.")
session.metadata = session.metadata.merge_with(
    MetadataExtractionResult(
        project_name="MyProject",
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        rejected_options=["Rails", "Microservices"],
    )
)
```

## Technical Considerations

1. **History vs. Memory Separation**: [`Message`](../src/aie_estimator/sessions/models.py) constrains `role` to `"user"` or `"assistant"`, keeping the system prompt outside [`ConversationHistory`](../src/aie_estimator/sessions/models.py) so each turn can re-render [`ProjectMetadata`](../src/aie_estimator/sessions/models.py) dynamically without duplicating system text.
2. **Self-Enforcing Window & Compression Seam**: [`ConversationHistory.append`](../src/aie_estimator/sessions/models.py) trims overflow pairs directly and returns the evicted oldest pairs (`list[Message]`) in chronological order so future compression policies can consume evicted turns without duplicating slice math.
3. **Metadata Merge & Policy 1 Retraction Semantics**: [`ProjectMetadata.merge_with`](../src/aie_estimator/sessions/models.py) consumes [`MetadataExtractionResult`](../src/aie_estimator/sessions/models.py) to overwrite scalars on non-empty values, prune rejected options from `mentioned_technologies`, un-reject entries via `retracted_options` or re-adopted technologies, remove `retracted_constraints` from `explicit_constraints`, and merge remaining list items via case-insensitive union preserving first-seen casing.
4. **Hybrid Post-Turn Metadata Extraction**: [`update_metadata`](../src/aie_estimator/sessions/metadata_extractor.py) runs an async structured LLM extraction call (`metadata_extraction/v1`) followed by deterministic post-validation (`sanitize_extraction_delta` stripping whitespace, clearing `"null"`-like literals, and deduplicating lists). Under `litellm`, extraction targets `Settings.metadata_extractor_model` (`reasoner-fast`); under `native`, it falls back to `opts.effective_model`. Extraction is skipped when `enable_metadata_extraction=False` or `confidence_pct < 30` (`Out of scope:`), and any failure logs `metadata_extraction_failed` while returning prior `ProjectMetadata` unchanged inside [`MetadataExtractionOutcome`](../src/aie_estimator/sessions/metadata_extractor.py).
5. **Multipart Form Contract & Fast-Fail Ordering**: [`estimate_in_session`](../src/aie_estimator/routers/sessions_router.py) (`POST /api/v1/sessions/{session_id}/estimate`) validates `transcript` (`20..2000` characters after trimming) and `model` ([`validate_catalog_model`](../src/aie_estimator/schemas/estimations.py)) via `Annotated[..., Form()]`, resolves the session via `store.get_or_404(session_id)`, and validates attachment count via [`validate_attachment_count`](../src/aie_estimator/attachments/extractor.py) before reading upload bytes or calling [`EstimationService.generate_session_estimation`](../src/aie_estimator/services/estimations_service.py).
6. **Lazy TTL Eviction & Process Volatility**: [`SessionStore`](../src/aie_estimator/sessions/store.py) tracks `last_accessed_at` per session against `Settings.session_ttl_seconds` (default `86_400`), evicting expired sessions lazily in `get_or_404` and sweeping unqueried expired entries during `create()`. State lives in a single-process dictionary (`uvicorn --workers=1`) and resets on restart until replaced by a persistent store behind the same interface.
7. **Singleton Wiring & Centralized Error Mapping**: [`get_session_store`](../src/aie_estimator/dependencies.py) caches the process-local [`SessionStore`](../src/aie_estimator/sessions/store.py) via `@lru_cache(maxsize=1)`, and [`Server`](../src/aie_estimator/server.py) maps [`SessionNotFoundError`](../src/aie_estimator/sessions/store.py) to HTTP `404`, [`UnsupportedAttachmentError`](../src/aie_estimator/attachments/extractor.py) to HTTP `415`, and [`AttachmentExtractionError`](../src/aie_estimator/attachments/extractor.py) to HTTP `422`.
8. **Conversational UI Client**: [`streamlit_app.py`](../streamlit_app.py) consumes these endpoints via lazy session creation and transparent 404 recovery, retaining the visible conversation thread client-side because `GET /api/v1/sessions/{session_id}` omits message bodies; see [Conversational Estimation UI](streamlit-conversational-ui.md).
