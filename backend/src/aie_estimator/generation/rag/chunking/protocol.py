# src/aie_estimator/generation/rag/chunking/protocol.py
"""
Structural typing contract for RAG budget chunking strategies.

Uses ``typing.Protocol`` instead of an abstract base class so chunking
strategies and test doubles satisfy the contract structurally without
inheritance boilerplate or premature LLM-chunker cost attributes.
"""

# --- Python Standard Library ---
from collections.abc import Sequence
from typing import Protocol

# --- App modules ---
from aie_estimator.domain.schemas.embeddings import Chunk, HistoricalBudget


class Chunker(Protocol):
    """Structural contract implemented by historical budget chunking strategies."""

    # Declared as a read-only property so concrete chunkers satisfy the protocol with either
    # a class attribute (e.g. `strategy_name: Final[str] = "structural"`) or a `@property`.
    @property
    def strategy_name(self) -> str:
        """Stable identifier for the chunking strategy."""
        ...

    def chunk(self, budgets: Sequence[HistoricalBudget]) -> list[Chunk]:
        """
        Split historical budgets into embeddable chunks.

        :param budgets: Ordered sequence of historical budgets to chunk.

        :return: Ordered list of chunks ready for embedding generation.
        """
        ...
