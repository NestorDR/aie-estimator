# CAG Baseline Stress Test Report

- **Model / Profile**: `reasoner-fast (openai/gpt-6-luna prioritized)` (`--http http://127.0.0.1:8000`, `--repeats 1`)
- **Dataset**: [`evals/stress/results.csv`](./results.csv) — **65 turns** executed (**0 errors**)
- **Total Spend**: **\$0.1149** (`65/65` turns passed the \$0.05/turn cost budget)
- **Latency Budget (8,000 ms)**: **4/65** turns passed (`61/65` exceeded 8 seconds)

## Summary Table

| Scenario / Test | KB | Turns | P50 ms | P95 ms | Total Cost ($) | Cache Hit % | Mean Drift Recall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attachment_0kb | 0 | 1 | 6261 | 6261 | 0.0003 | 0.0% | n/a |
| attachment_100kb | 100 | 1 | 8646 | 8646 | 0.0006 | 0.0% | 0.0% |
| attachment_20kb | 20 | 1 | 7759 | 7759 | 0.0006 | 0.0% | 0.0% |
| attachment_50kb | 50 | 1 | 6353 | 6353 | 0.0005 | 0.0% | 0.0% |
| attachment_5kb | 5 | 1 | 8797 | 8797 | 0.0007 | 0.0% | 0.0% |
| contradiction | 0 | 20 | 11875 | 17194 | 0.0309 | 0.0% | 0.0% |
| growing | 0 | 20 | 15797 | 17991 | 0.0418 | 0.0% | 100.0% |
| pivot | 0 | 20 | 15637 | 19457 | 0.0396 | 0.0% | 0.0% |

## Curve 1: Latency vs Tokens In

| Bucket (tokens_in) | Turns | Mean tokens_in | P50 ms | P95 ms |
| :--- | ---: | ---: | ---: | ---: |
| 1000-2999 | 5 | 2638 | 8797 | 10300 |
| 3000-9999 | 32 | 6925 | 12749 | 16426 |
| >=10000 | 28 | 11760 | 17442 | 19863 |

## Curve 2: Cumulative Cost vs Turn Index (Multi-Turn)

| Turn | growing ($) | pivot ($) | contradiction ($) |
| ---: | ---: | ---: | ---: |
| 1 | 0.0011 | 0.0006 | 0.0005 |
| 2 | 0.0023 | 0.0016 | 0.0010 |
| 3 | 0.0037 | 0.0030 | 0.0017 |
| 4 | 0.0051 | 0.0045 | 0.0025 |
| 5 | 0.0067 | 0.0061 | 0.0033 |
| 6 | 0.0084 | 0.0078 | 0.0043 |
| 7 | 0.0104 | 0.0095 | 0.0055 |
| 8 | 0.0127 | 0.0116 | 0.0072 |
| 9 | 0.0150 | 0.0138 | 0.0088 |
| 10 | 0.0174 | 0.0159 | 0.0105 |
| 11 | 0.0198 | 0.0182 | 0.0123 |
| 12 | 0.0221 | 0.0206 | 0.0142 |
| 13 | 0.0245 | 0.0230 | 0.0160 |
| 14 | 0.0269 | 0.0255 | 0.0180 |
| 15 | 0.0290 | 0.0281 | 0.0200 |
| 16 | 0.0315 | 0.0297 | 0.0220 |
| 17 | 0.0340 | 0.0324 | 0.0241 |
| 18 | 0.0365 | 0.0351 | 0.0263 |
| 19 | 0.0391 | 0.0375 | 0.0286 |
| 20 | 0.0418 | 0.0396 | 0.0309 |

## Curve 3: MemoryDriftMetric Recall vs N

| Checkpoint Turn (N) | growing | pivot | contradiction |
| ---: | ---: | ---: | ---: |
| 1 | n/a | n/a | n/a |
| 3 | 100.0% | 0.0% | 0.0% |
| 6 | 100.0% | 0.0% | 0.0% |
| 10 | 100.0% | 0.0% | 0.0% |
| 20 | 100.0% | 0.0% | 0.0% |

## Where My CAG Starts Breaking and Why

**1. Response time breaks before budget does:** Even though every single turn stayed cheap (`65/65` passed the \$0.05/turn limit, costing **\$0.1149** in total across 65 turns), **61 of 65 turns missed the 8,000 ms latency target**. As soon as conversation history grows past 3,000 input tokens, typical response time jumps from **8.8 s** (`1000-2999` tokens) to **12.7 s** (`3000-9999` tokens) and **17.4 s** (`>=10000` tokens, with P95 reaching **19.9 s**). For a user in a live chat, waiting 15–20 seconds per message feels slow—and because conversational turns bypass both caches (`0.0%` cache hits) and trigger extra background summary/metadata LLM calls from turn 7 onward, latency keeps climbing even with the fastest LLM profile (`reasoner-fast`).

**2. Large files get cut off and early facts are fragile:** On file uploads, the system caps extracted attachment text at **12,000 characters**, silently dropping **4,378 characters** at `20 KB`, **28,930 characters** at `50 KB`, and **69,853 characters** at `100 KB`, which caused **0.0%** canary recall across all PDF sizes (`5–100 KB`). In multi-turn conversations, each scenario introduces a Turn-1 project name tracked against `metadata.project_name`: `growing` kept `"Nimbus"` at **100.0%** recall across all 20 turns, whereas `pivot` (`"Helios"`) and `contradiction` (`"Atlas"`) dropped to **0.0%** from Turn 2 onward because the post-turn metadata extractor failed to capture those project names. In addition, memory anchors only promote contractual or compliance commitments once the 6-turn window overflows (Turn 7+), so ordinary project introductions never became anchors (`anchors_count = 0` across all 65 turns). In plain terms: our current CAG setup works reliably for small, steady conversations, but it loses details when documents exceed ~12k characters or when early facts are missed by metadata extraction.
