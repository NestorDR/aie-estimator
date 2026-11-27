# RAG Embedding & Tokenization

## Overview

The RAG embedding layer provides asynchronous, batched OpenAI vector generation and model-aware BPE token counting under [`../backend/src/aie_estimator/generation/rag/embedding/`](../backend/src/aie_estimator/generation/rag/embedding/):

1. **Tokenizer** ([`tokenizer.py`](../backend/src/aie_estimator/generation/rag/embedding/tokenizer.py)): Exposes `count_tokens()`, resolving and caching `tiktoken` encodings from `Settings.embedding_model` with `cl100k_base` fallback on unknown models.
2. **Async Embedder** ([`embedder.py`](../backend/src/aie_estimator/generation/rag/embedding/embedder.py)): Exposes `OpenAIEmbedder.embed_one()` and `OpenAIEmbedder.embed_many()` over `AsyncOpenAI.embeddings.create` with configurable batching and `asyncio.sleep` exponential backoff on `RateLimitError`.
3. **Configuration, DI & Pricing**: Externalizes `embedding_model`, `embedding_dimensions`, and `embedding_batch_size` in [`Settings`](../backend/src/aie_estimator/settings.py), wires `get_openai_client()`, `get_embedder()`, and `get_chunker()` in [`dependencies.py`](../backend/src/aie_estimator/dependencies.py), and centralizes `text-embedding-3-small` pricing (`$0.02 / 1M` input tokens) in [`pricing.py`](../backend/src/aie_estimator/foundation/llm/pricing.py).
4. **Ingest Endpoint** ([`embeddings_router.py`](../backend/src/aie_estimator/api/embeddings_router.py)): `POST /api/v1/embeddings/ingest` chunks `HistoricalBudget`s with the injected `Chunker`, embeds every chunk with `OpenAIEmbedder.embed_many()`, and returns `EmbeddingIngestResponse` (vectors plus token/cost stats). It is stateless: no vector store, cache, or moderation.
5. **Cosine Similarity & CLI** ([`similarity.py`](../backend/src/aie_estimator/generation/rag/analysis/similarity.py), [`compare.py`](../backend/scripts/compare.py), [`SANITY_CHECK.md`](../backend/src/aie_estimator/generation/rag/SANITY_CHECK.md)): Computes cosine similarity using Python stdlib only, exposes `scripts/compare.py` (`--1st` / `--2nd`, wrapped by [`compare_texts.cmd`](../backend/auto/compare_texts.cmd)), and records the live three-pair sanity check baseline.

---

## Usage Examples

```python
from aie_estimator.dependencies import get_embedder, get_openai_client
from aie_estimator.foundation.llm.pricing import estimate_cost
from aie_estimator.generation.rag.embedding.tokenizer import count_tokens
from aie_estimator.settings import get_settings

settings = get_settings()
embedder = get_embedder(get_openai_client())
tokens = count_tokens("OAuth 2.0 authentication backend")
cost_usd = estimate_cost(settings.embedding_model, tokens_in=tokens, tokens_out=0)
```

```bash
curl -X POST http://localhost:8000/api/v1/embeddings/ingest \
  -H "Content-Type: application/json" \
  -d '{"budgets": [<HistoricalBudget>, ...]}'
```

A sample `budgets` payload is available at [`budgets_sample.json`](../backend/data/samples/budgets_sample.json).

```bash
# Local (from backend/)
uv --directory backend run python scripts/compare.py \
  --1st "Backend services" \
  --2nd "API development"

# Windows wrapper (from backend/)
auto\compare_texts.cmd --1st "Backend services" --2nd "API development"

# Inside the production Docker container
docker compose -f docker/docker-compose.prod.yml exec estimator \
  python scripts/compare.py --1st "Backend services" --2nd "API development"
```

---

## Technical Considerations

1. **Non-Blocking Retries**: `get_openai_client()` caches a single `AsyncOpenAI(..., max_retries=0)` client so `OpenAIEmbedder._create_batch()` owns retry timing across `(1, 2, 4)` second `asyncio.sleep` waits without SDK retry amplification. Non-rate-limit exceptions propagate immediately.
2. **Native Dimension Preservation**: `OpenAIEmbedder` delegates to [`resolve_request_dimensions()`](../backend/src/aie_estimator/utils/embeddings_utils.py) to omit `dimensions` from OpenAI API payloads whenever `Settings.embedding_dimensions` matches the configured model's native size in `EMBEDDING_NATIVE_DIMENSIONS`, forwarding `dimensions` for non-native sizes and unknown models.
3. **Missing Key Handling**: `get_openai_client()` and `get_embedder()` return `None` when `OPENAI_API_KEY` is unset or blank; the ingest endpoint answers `503` with a generic message.
4. **CAG Coexistence**: [`generation/cag/semantic.py`](../backend/src/aie_estimator/generation/cag/semantic.py) retains its own `AsyncOpenAI` wiring until Session 7 live unifies CAG and RAG embedding clients.
5. **Ingest Error Mapping**: `RateLimitError` (after the embedder's retries) maps to `503`, other `openai.APIError` to `502`, and any other exception to `500`. Details are fixed generic strings; the exception type and a 300-character message are only logged. `RateLimitError` subclasses `APIError`, so its `except` clause must come first.
6. **Ingest Statistics**: `total_tokens` is the sum of local tiktoken `Chunk.token_count` values (`embed_many` returns vectors only), so it can differ slightly from API-billed tokens. `estimated_cost_usd` falls back to `0.0` for models missing from the pricing catalog. Chunking runs inline because tiktoken over a few hundred short chunks takes milliseconds.
7. **Pure-Stdlib Similarity & Container Packaging**: [`cosine_similarity()`](../backend/src/aie_estimator/generation/rag/analysis/similarity.py) uses only `math.sqrt` and Python built-ins (no `numpy` or `scikit-learn`), returning `0.0` on zero-norm/empty vectors and raising `ValueError` on mismatched dimensions. [`backend/Dockerfile`](../backend/Dockerfile) copies `scripts/` into `/app/scripts/` so `scripts/compare.py` runs identically inside the container without `sys.path` hacks.
