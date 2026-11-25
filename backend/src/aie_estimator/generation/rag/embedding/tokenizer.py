# src/aie_estimator/generation/rag/embedding/tokenizer.py
"""
Model-aware BPE token counting for the RAG pipeline.

Computes exact OpenAI BPE token counts offline so chunkers can detect oversized
chunks before calling the embeddings API and callers can estimate USD cost
accurately.
"""

# --- Python Standard Library ---
from functools import lru_cache
from typing import Final

# --- Third Party Libraries ---
import tiktoken

# --- App modules ---
from aie_estimator.settings import get_settings

_FALLBACK_ENCODING: Final[str] = "cl100k_base"


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """
    Resolve and cache the ``tiktoken`` BPE encoding for an embedding model.

    Falls back to ``cl100k_base`` when ``tiktoken`` has no explicit mapping for
    the given model identifier.

    :param model: Target OpenAI embedding model identifier.

    :return: Cached ``tiktoken.Encoding`` instance.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Unregistered or custom model IDs fall back to the text-embedding-3-* BPE encoding.
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def count_tokens(text: str, model: str | None = None) -> int:
    """
    Count BPE tokens in ``text`` using the configured embedding model encoding.

    :param text: Input text to tokenize.
    :param model: Optional embedding model identifier; defaults to ``Settings.embedding_model``.

    :return: Exact token count as an integer.
    """
    resolved_model = model or get_settings().embedding_model
    return len(_get_encoding(resolved_model).encode(text))
