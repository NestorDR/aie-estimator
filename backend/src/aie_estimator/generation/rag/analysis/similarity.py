# src/aie_estimator/generation/rag/analysis/similarity.py
"""
Pure-stdlib vector similarity helpers for the RAG analysis layer.

Computes cosine similarity using ``math.sqrt`` and Python built-ins so the
backend stays free of heavy numerical dependencies (``numpy``, ``scikit-learn``)
while preserving explicit guards for zero-norm and mismatched-dimension vectors.
"""

# --- Python Standard Library ---
from collections.abc import Sequence
import math


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """
    Compute the cosine similarity between two numeric vectors.

    Returns ``0.0`` when either vector is empty or has zero L2 norm to avoid ``ZeroDivisionError``.

    :param vector_a: First embedding vector.
    :param vector_b: Second embedding vector.

    :return: Cosine similarity in ``[-1.0, 1.0]``, or ``0.0`` for zero-norm vectors.
    :raises ValueError: If the two vectors have different lengths.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(f"Vectors must have the same length (got {len(vector_a)} and {len(vector_b)}).")

    dot = sum(x * y for x, y in zip(vector_a, vector_b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in vector_a))
    norm_b = math.sqrt(sum(y * y for y in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
