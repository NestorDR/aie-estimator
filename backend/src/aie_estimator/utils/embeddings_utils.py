# src/aie_estimator/utils/embeddings_utils.py
"""
Helpers shared by every component that calls the OpenAI embeddings endpoint.
"""

# --- Python Standard Library ---
import re
from types import MappingProxyType
from typing import Final

# Default embedding model and its native vector size, shared between Settings and OpenAIEmbedder
# so `dimensions=` is omitted from OpenAI API calls unless explicitly overridden.
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS: Final[int] = 1536

# Native vector size per OpenAI embedding model.
# `dimensions` must be omitted when the requested size equals the native one.
# MappingProxyType is a read-only dictionary, which fits with Final.
EMBEDDING_NATIVE_DIMENSIONS: Final[MappingProxyType[str, int]] = MappingProxyType(
    {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        # text-embedding-ada-002 does not support the ``dimensions`` parameter.
        # If it was set to a value other than 1536, the API would return an error.
        # You could validate it in settings.py.
        "text-embedding-ada-002": 1536,
    }
)

_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]+")

def embedding_signature(model: str, dimensions: int) -> str:
    """
    Build a stable identifier for an embedding space (model + vector size).

    Two vectors are comparable only if they share this signature, so it is used to
    namespace vector indexes and keys.

    :param model: Embedding model name.
    :param dimensions: Effective vector size stored in the index.

    :return: Identifier safe for Redis names, e.g. ``text_embedding_3_small_1536``.
    """
    return f"{_NON_ALNUM.sub('_', model).strip('_').lower()}_{dimensions}"


def resolve_request_dimensions(model: str, dimensions: int) -> int | None:
    """
    Decide whether the ``dimensions`` parameter must be sent to the OpenAI embeddings API.

    :param model: Embedding model name.
    :param dimensions: Vector size requested through settings.

    :return: ``None`` when ``dimensions`` equals the model's native size (parameter omitted),
        otherwise the requested ``dimensions``. Unknown models always get it sent.
    """
    return None if EMBEDDING_NATIVE_DIMENSIONS.get(model) == dimensions else dimensions
