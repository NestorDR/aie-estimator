# tests/utils/test_embedding_utils.py
"""
Unit tests for shared OpenAI embedding dimension resolution and signature helpers.

Verifies per-model native dimension omission, Redis-safe embedding-space signatures,
and runtime immutability of the native dimensions mapping.
"""

# --- Python Standard Library ---
import re
from typing import Any, cast

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.utils.embeddings_utils import (
    EMBEDDING_NATIVE_DIMENSIONS,
    embedding_signature,
    resolve_request_dimensions,
)


@pytest.mark.parametrize(
    ("model", "dimensions", "expected"),
    [
        ("text-embedding-3-small", 1536, None),
        ("text-embedding-3-small", 512, 512),
        ("text-embedding-3-large", 3072, None),
        ("text-embedding-3-large", 1536, 1536),
        ("unknown-model", 1536, 1536),
    ],
)
def test_resolve_request_dimensions(model: str, dimensions: int, expected: int | None) -> None:
    """
    GIVEN an embedding model name and requested vector dimension
    WHEN resolve_request_dimensions is called
    THEN it returns None for the model's native size and the requested size otherwise.
    """
    assert resolve_request_dimensions(model, dimensions) == expected


def test_embedding_signature_format_and_uniqueness() -> None:
    """
    GIVEN embedding model names and vector dimensions
    WHEN embedding_signature is computed
    THEN it produces deterministic Redis-safe identifiers unique per model and dimension.
    """
    small_sig = embedding_signature("text-embedding-3-small", 1536)
    large_sig = embedding_signature("text-embedding-3-large", 1536)
    exotic_sig = embedding_signature("OpenAI/Custom.Model--v2!", 768)

    assert small_sig == "text_embedding_3_small_1536"
    assert small_sig != large_sig
    for sig in (small_sig, large_sig, exotic_sig):
        assert re.fullmatch(r"[a-z0-9_]+", sig) is not None


def test_embedding_native_dimensions_is_immutable() -> None:
    """
    GIVEN the module-level EMBEDDING_NATIVE_DIMENSIONS mapping
    WHEN a caller attempts item assignment
    THEN a TypeError is raised because the mapping is read-only.
    """
    mutable_view = cast(Any, EMBEDDING_NATIVE_DIMENSIONS)
    with pytest.raises(TypeError, match=r"does not support item assignment"):
        mutable_view["text-embedding-3-small"] = 512
