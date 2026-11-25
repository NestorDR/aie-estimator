# RAG Embedding & Tokenization

## Overview

The RAG embedding layer provides asynchronous, batched OpenAI vector generation and model-aware BPE token counting under [`../backend/src/aie_estimator/generation/rag/embedding/`](../backend/src/aie_estimator/generation/rag/embedding/):

1. **Tokenizer** ([`tokenizer.py`](../backend/src/aie_estimator/generation/rag/embedding/tokenizer.py)): Exposes `count_tokens()`, resolving and caching `tiktoken` encodings from `Settings.embedding_model` with `cl100k_base` fallback on unknown models.
2. **Async Embedder** ([`embedder.py`](../backend/src/aie_estimator/generation/rag/embedding/embedder.py)): Exposes `OpenAIEmbedder.embed_one()` and `OpenAIEmbedder.embed_many()` over `AsyncOpenAI.embeddings.create` with configurable batching and `asyncio.sleep` exponential backoff on `RateLimitError`.
3. **Configuration, DI & Pricing**: Externalizes `embedding_model`, `embedding_dimensions`, and `embedding_batch_size` in [`Settings`](../backend/src/aie_estimator/settings.py), wires `get_openai_client()` and `get_embedder()` in [`dependencies.py`](../backend/src/aie_estimator/dependencies.py), and centralizes `text-embedding-3-small` pricing (`$0.02 / 1M` input tokens) in [`pricing.py`](../backend/src/aie_estimator/foundation/llm/pricing.py).

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

---

## Technical Considerations

1. **Non-Blocking Retries**: `get_openai_client()` caches a single `AsyncOpenAI(..., max_retries=0)` client so `OpenAIEmbedder._create_batch()` owns retry timing across `(1, 2, 4)` second `asyncio.sleep` waits without SDK retry amplification. Non-rate-limit exceptions propagate immediately.
2. **Native Dimension Preservation**: `OpenAIEmbedder` omits `dimensions` from OpenAI API payloads only when both `Settings.embedding_model == DEFAULT_EMBEDDING_MODEL` and `Settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS` (`1536`), forwarding it for any custom dimension or alternative model.
3. **Missing Key Handling**: `get_openai_client()` and `get_embedder()` return `None` when `OPENAI_API_KEY` is unset or blank so downstream routes can surface controlled service-unavailable responses.
4. **CAG Coexistence**: [`generation/cag/semantic.py`](../backend/src/aie_estimator/generation/cag/semantic.py) retains its own `AsyncOpenAI` wiring until Session 7 live unifies CAG and RAG embedding clients.
