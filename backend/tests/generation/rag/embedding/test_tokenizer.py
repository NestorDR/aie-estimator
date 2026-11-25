# tests/generation/rag/embedding/test_tokenizer.py
"""
Unit tests for model-aware BPE token counting in ``generation/rag/embedding/tokenizer.py``.
"""

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.generation.rag.embedding.tokenizer import _get_encoding, count_tokens


@pytest.mark.parametrize(
    ("text", "model", "expected_tokens"),
    [
        ("", None, 0),
        ("hello world", None, 2),
        ("hello world", "text-embedding-3-small", 2),
        ("hello world", "unknown-custom-embedding-model", 2),
    ],
)
def test_count_tokens_known_and_fallback_models(
    text: str,
    model: str | None,
    expected_tokens: int,
) -> None:
    """
    GIVEN empty or non-empty text and default, known, or unknown model identifiers
    WHEN count_tokens is called
    THEN it returns the exact BPE token count using Settings.embedding_model or cl100k_base fallback.
    """
    assert count_tokens(text, model=model) == expected_tokens


def test_get_encoding_caches_resolved_encoding() -> None:
    """
    GIVEN repeated calls to _get_encoding with the same model name
    WHEN _get_encoding is invoked
    THEN it returns the cached tiktoken Encoding instance.
    """
    _get_encoding.cache_clear()
    first = _get_encoding("text-embedding-3-small")
    second = _get_encoding("text-embedding-3-small")
    assert first is second
    assert first.name == "cl100k_base"
