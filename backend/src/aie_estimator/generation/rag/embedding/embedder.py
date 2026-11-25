# src/aie_estimator/generation/rag/embedding/embedder.py
"""
Asynchronous batched OpenAI embedding service for the RAG pipeline.

Batches multiple inputs into a single ``AsyncOpenAI.embeddings.create`` request
to avoid per-chunk HTTP latency, uses ``asyncio.sleep`` backoff retries so the
FastAPI event loop never blocks, and omits ``dimensions`` by default to preserve
the model's native vector size.
"""

# --- Python Standard Library ---
import asyncio
from collections.abc import Sequence
import time
from typing import Any, Final

# --- Third Party Libraries ---
from openai import AsyncOpenAI, RateLimitError
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings
from aie_estimator.utils.constants import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_RETRY_BACKOFF_SECONDS: Final[tuple[int, ...]] = (1, 2, 4)


class OpenAIEmbedder:
    """
    Asynchronous wrapper over ``AsyncOpenAI.embeddings.create`` with batching and backoff retries.
    """

    def __init__(self, client: AsyncOpenAI) -> None:
        """
        Initialize the async embedder from the application settings singleton.

        Omits ``dimensions`` from OpenAI API requests when ``embedding_dimensions``
        matches ``DEFAULT_EMBEDDING_DIMENSIONS`` (native ``1536``).

        :param client: Configured ``AsyncOpenAI`` client instance.
        """
        settings = get_settings()
        self._client: AsyncOpenAI = client
        self._model: str = settings.embedding_model
        self._batch_size: int = settings.embedding_batch_size
        # Omit `dimensions` only for the default model's native size (1536); larger models
        # like text-embedding-3-large (native 3072) must still receive dimensions=1536 when truncating.
        is_native_default = (
            settings.embedding_model == DEFAULT_EMBEDDING_MODEL
            and settings.embedding_dimensions == DEFAULT_EMBEDDING_DIMENSIONS
        )
        self._dimensions: int | None = None if is_native_default else settings.embedding_dimensions

    async def embed_one(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a single text string.

        :param text: Input text to embed.

        :return: Embedding vector as a list of floats.
        """
        vectors, _ = await self._create_batch([text])
        return vectors[0]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embedding vectors for a sequence of texts using batched API calls.

        :param texts: Ordered sequence of input texts to embed.

        :return: Embedding vectors aligned with the input sequence order.
        """
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            started_at = time.perf_counter()
            vectors, tokens = await self._create_batch(batch)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 1)

            log.info(
                "embedding_batch_completed",
                chunks=len(batch),
                tokens=tokens,
                latency_ms=latency_ms,
                model=self._model,
            )
            embeddings.extend(vectors)

        return embeddings

    async def _create_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """
        Call ``AsyncOpenAI.embeddings.create`` with exponential backoff on ``RateLimitError``.

        :param texts: Non-empty batch of input strings.

        :return: Tuple of ordered embedding vectors and API-reported total token count.
        :raises ValueError: If any input string is empty or whitespace-only.
        :raises RateLimitError: When all retry attempts are exhausted.
        """
        # Fail fast locally because OpenAI's /v1/embeddings endpoint rejects empty strings with HTTP 400.
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input texts must not be empty.")

        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions

        # Explicit (1s, 2s, 4s) RateLimitError backoff is implemented here (with client max_retries=0
        # in dependencies.py) rather than delegating to AsyncOpenAI's built-in retries so we honor
        # the s07.0 schedule, emit `embedding_batch_rate_limited` structlog warnings, and keep retry
        # timing deterministically testable.
        # Runs 1 initial attempt + `retry_attempts` retries, re-raising on the final failure without an extra sleep.
        retry_attempts = len(_RETRY_BACKOFF_SECONDS)
        for attempt in range(retry_attempts + 1):
            try:
                response = await self._client.embeddings.create(**kwargs)
                break
            except RateLimitError:
                if attempt == retry_attempts:
                    raise

                wait_seconds = _RETRY_BACKOFF_SECONDS[attempt]
                log.warning(
                    "embedding_batch_rate_limited",
                    attempt=attempt + 1,
                    retry_in_s=wait_seconds,
                    model=self._model,
                )
                await asyncio.sleep(wait_seconds)

        # Sort by `item.index` to guarantee 1:1 alignment with the input batch order.
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return vectors, response.usage.total_tokens
