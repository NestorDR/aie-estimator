# tests/generation/rag/analysis/test_similarity.py
"""
Unit tests for pure-stdlib cosine similarity calculations.

Verifies geometric invariants (identical, orthogonal, opposite), zero-norm and
empty-vector safety, and mismatched-dimension error handling offline.
"""

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.generation.rag.analysis.similarity import cosine_similarity


@pytest.mark.parametrize(
    ("vector_a", "vector_b", "expected"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, -2.0], [-1.0, 2.0], -1.0),
        ([0.0, 0.0], [1.0, 2.0], 0.0),
        ([1.0, 2.0], [0.0, 0.0], 0.0),
        ([], [], 0.0),
    ],
)
def test_cosine_similarity_returns_expected_score(
    vector_a: list[float],
    vector_b: list[float],
    expected: float,
) -> None:
    """
    GIVEN two equal-length vectors (identical, orthogonal, opposite, zero, or empty)
    WHEN cosine_similarity is called
    THEN it returns the expected score within floating-point tolerance.
    """
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


def test_cosine_similarity_raises_on_mismatched_dimensions() -> None:
    """
    GIVEN two vectors of different lengths
    WHEN cosine_similarity is called
    THEN it raises ValueError.
    """
    with pytest.raises(ValueError, match=r"same length"):
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
