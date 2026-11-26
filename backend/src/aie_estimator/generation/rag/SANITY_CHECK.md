# Embedding Sanity Check

- **Model**: `text-embedding-3-small` (1536 native dimensions)
- **Tool**: [`backend/scripts/compare.py`](../../../../scripts/compare.py) (pure-stdlib cosine similarity via [`similarity.py`](analysis/similarity.py))
- **Run date**: 2026-11-26

| Pair | 1st Text | 2nd Text | Cosine Similarity | Expectation |
| :--- | :--- | :--- | :---: | :--- |
| **A — near** | `"OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"` | `"Authorization service using JSON Web Tokens for a banking application"` | **0.5958** | High (>= 0.6) |
| **B — unrelated** | `"OAuth 2.0 authentication backend with JWT tokens for fintech mobile app"` | `"Database migration from MySQL to PostgreSQL with zero downtime"` | **0.1920** | Low (< 0.4) |
| **C — generic** | `"Backend services"` | `"API development"` | **0.5407** | No fixed expectation |

## First Impressions

- Pair B behaves as expected: unrelated authentication and database-migration descriptions separate cleanly at `0.1920`, well below `0.4`.
- Pair A lands at `0.5958`, just under the orientative `0.6` threshold, because the two sentences share almost no surface vocabulary (`OAuth/JWT/fintech` vs. `Authorization/JSON Web Tokens/banking`) and rely purely on semantic bridging.
- Pair C scores `0.5407`—nearly as high as the semantically equivalent Pair A—showing that short, generic phrases carry weak discriminating signal and cluster together in embedding space.

This reinforces prepending project, sector, and technology context in `JSONStructuralChunker` so generic component names do not collapse into the same region during retrieval.
