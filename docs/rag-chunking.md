# RAG Structural Chunking & Embedding Schemas

## Overview

The RAG structural chunking layer converts normalized component-based historical budgets into embeddable chunks with contextual parent headers and filterable non-embedded metadata:

1. **Embedding Pipeline Schemas** ([`embeddings.py`](../backend/src/aie_estimator/domain/schemas/embeddings.py)): Strict Pydantic v2 contracts (`ClientMetadata`, `BudgetComponent`, `HistoricalBudget`, `Chunk`, `EmbeddedChunk`, `EmbeddingIngestRequest`, `EmbeddingIngestStats`, `EmbeddingIngestResponse`) enforcing `extra="forbid"`, closed `Sector`/`Complexity` literals, non-empty collections, and non-negative numeric bounds.
2. **Chunker Protocol** ([`protocol.py`](../backend/src/aie_estimator/generation/rag/chunking/protocol.py)): Structural `Chunker(Protocol)` contract (`strategy_name: str` and `chunk(budgets) -> list[Chunk]`) enabling downstream DI and multi-strategy pluggability without `ABC` inheritance.
3. **Structural JSON Chunker** ([`structural.py`](../backend/src/aie_estimator/generation/rag/chunking/structural.py)): `JSONStructuralChunker` plus pure helpers `render_component_text()` and `component_metadata()`, producing 1:1 component-to-chunk records with deterministic `{budget_id}::{component_id}` identifiers, exact BPE `token_count` via [`count_tokens()`](../backend/src/aie_estimator/generation/rag/embedding/tokenizer.py), and homogeneous `budget_chunking_completed` structured logging.
4. **Curated Sample Dataset** ([`budgets_sample.json`](../backend/data/samples/budgets_sample.json)): 17 normalized `HistoricalBudget` records (60 components) kept separate from the Session 6 offline ingestion catalog (`backend/data/catalog/catalog.yaml`).

---

## Usage Examples

```python
import json
from pathlib import Path

from aie_estimator.domain.schemas.embeddings import HistoricalBudget
from aie_estimator.generation.rag.chunking.structural import JSONStructuralChunker

raw_budgets = json.loads(Path("backend/data/samples/budgets_sample.json").read_text(encoding="utf-8"))
budgets = [HistoricalBudget.model_validate(item) for item in raw_budgets]

chunks = JSONStructuralChunker().chunk(budgets)
# Produces 60 deterministic chunks (chunk_id="{budget_id}::{component_id}")
```

---

## Technical Considerations

1. **Single-Schema RAG Boundary**: `generation/rag/` operates exclusively on `HistoricalBudget` (`components[]`). Session 6's `CleanedBudget` (`phases[]`) remains isolated inside [`ingestion/parsers/budget_json.py`](../backend/src/aie_estimator/ingestion/parsers/budget_json.py) so downstream chunking strategies and vector metadata filters avoid dual-schema branching and lossy field synthesis.
2. **Zero-Overlap 1:1 Granularity**: Each `BudgetComponent` yields one `Chunk` with no overlap and no fixed-size splitting; abnormally large component descriptions surface transparently through `Chunk.token_count`.
3. **Client-Name Omission**: Both `render_component_text()` and `component_metadata()` intentionally exclude `client_metadata.name` so vector embeddings, filterable metadata payloads, and `budget_chunking_completed` logs do not leak client company names.
4. **Structured Telemetry**: `JSONStructuralChunker.chunk()` emits `budget_chunking_completed` (`strategy`, `budgets`, `chunks`, `tokens`, `latency_ms`), aligning field names with `embedding_batch_completed` in [`OpenAIEmbedder`](../backend/src/aie_estimator/generation/rag/embedding/embedder.py). See [`docs/observability.md`](observability.md).
