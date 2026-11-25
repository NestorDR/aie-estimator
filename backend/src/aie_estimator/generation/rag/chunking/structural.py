# src/aie_estimator/generation/rag/chunking/structural.py
"""
Structural 1:1 JSON chunker for component-based historical budgets.

Trusts the normalized ``HistoricalBudget`` structure: each ``BudgetComponent``
becomes one ``Chunk`` with zero overlap and no fixed-size splitting so oversized
descriptions surface directly via ``token_count``. Prepends a contextual parent
header (omitting ``client_metadata.name``) and attaches non-embedded filterable
metadata for downstream retrieval.
"""

# --- Python Standard Library ---
from collections.abc import Sequence
import time
from typing import Any, Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.domain.schemas.embeddings import BudgetComponent, Chunk, HistoricalBudget
from aie_estimator.generation.rag.embedding.tokenizer import count_tokens

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)


def render_component_text(budget: HistoricalBudget, component: BudgetComponent) -> str:
    """
    Render parent budget context and component details into embeddable text.

    Omits ``client_metadata.name`` so embeddings capture technical and sector
    semantics without leaking client company names into vector representations.

    :param budget: Parent historical budget providing project context.
    :param component: Component line item to render.

    :return: Contextualized text string sent to the embeddings API.
    """
    # Contextual header anchors standalone component names in project/sector semantics while
    # omitting client_metadata.name so vectors stay free of client brand names.
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | "
        f"Year: {budget.year} | Main tech: {budget.main_technology}]\n"
        f"\n"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def component_metadata(budget: HistoricalBudget, component: BudgetComponent) -> dict[str, Any]:
    """
    Build filterable, non-embedded metadata for a component chunk.

    Omits ``client_metadata.name`` so filterable payloads remain free of client
    company names.

    :param budget: Parent historical budget.
    :param component: Component line item.

    :return: Dictionary of filterable chunk metadata fields.
    """
    return {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }


class JSONStructuralChunker:
    """Chunks historical budgets 1:1 per component with contextual parent headers."""

    strategy_name: Final[str] = "structural"

    def chunk(self, budgets: Sequence[HistoricalBudget]) -> list[Chunk]:
        """
        Produce one contextualized ``Chunk`` per component across all input budgets.

        :param budgets: Ordered sequence of historical budgets.

        :return: Ordered list of chunks preserving input budget and component order.
        """
        started_at = time.perf_counter()
        chunks: list[Chunk] = []
        # Strict 1:1 component-to-chunk mapping with no overlap or splitting so oversized
        # descriptions surface directly via token_count instead of being silently truncated.
        for budget in budgets:
            for component in budget.components:
                text = render_component_text(budget, component)
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata=component_metadata(budget, component),
                        token_count=count_tokens(text),
                    )
                )
        total_tokens = sum(chunk.token_count for chunk in chunks)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

        # Log aggregate counters only (never raw chunk text or client names), matching embedding_batch_completed.
        log.info(
            "budget_chunking_completed",
            strategy=self.strategy_name,
            budgets=len(budgets),
            chunks=len(chunks),
            tokens=total_tokens,
            latency_ms=latency_ms,
        )
        return chunks
