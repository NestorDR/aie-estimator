# src/aie_estimator/api/embeddings_router.py

"""
POST /ingest

Stateless chunk-and-embed endpoint: it chunks the submitted budgets, embeds every chunk in
memory, and returns the vectors with token/cost statistics. Nothing is persisted, cached, or
moderated; those concerns belong to the Session 8 vector store. Failures map to fixed generic
messages so upstream error text never reaches the client.
"""

# --- Python Standard Library ---
import time
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends, HTTPException, status
from openai import APIError, RateLimitError
from opentelemetry import trace
from opentelemetry.trace import Span
import structlog

# --- App modules ---
from aie_estimator.dependencies import get_chunker, get_embedder, get_settings
from aie_estimator.domain.schemas.embeddings import (
    EmbeddedChunk,
    EmbeddingIngestRequest,
    EmbeddingIngestResponse,
    EmbeddingIngestStats,
)
from aie_estimator.foundation.llm.pricing import estimate_cost
from aie_estimator.generation.rag.chunking.protocol import Chunker
from aie_estimator.generation.rag.embedding.embedder import OpenAIEmbedder
from aie_estimator.settings import Settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)
_TRACER: Final[trace.Tracer] = trace.get_tracer(__name__)

_DETAIL_UNAVAILABLE: Final[str] = "Embedding service is not available."
_DETAIL_RATE_LIMITED: Final[str] = "Embedding service is rate limited. Retry later."
_DETAIL_UPSTREAM_ERROR: Final[str] = "Embedding provider request failed."
_DETAIL_UNEXPECTED: Final[str] = "Failed to generate embeddings."

router = APIRouter(
    prefix="/api/v1/embeddings",
    tags=["Embeddings"],
)


def _failure(
        span: Span,
        reason: str,
        status_code: int,
        detail: str,
        exc: Exception | None = None,
) -> HTTPException:
    """
    Log and trace an ingest failure, then build the sanitized HTTP error to raise.

    :param span: Active ``embeddings.ingest`` span.
    :param reason: Stable machine-readable failure reason for logs.
    :param status_code: HTTP status returned to the client.
    :param detail: Generic client-facing message (never upstream error text).
    :param exc: Originating exception, if any.

    :return: ``HTTPException`` for the caller to raise.
    """
    if exc is None:
        log.error("embedding_ingest_failed", reason=reason)
    else:
        span.record_exception(exc)
        # Truncated message: upstream errors can echo request payloads, so keep logs bounded.
        log.error("embedding_ingest_failed", reason=reason, error_type=type(exc).__name__, error=str(exc)[:300])
    return HTTPException(status_code=status_code, detail=detail)


@router.post(
    "/ingest",
    response_model=EmbeddingIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk and embed historical budgets",
    description="Chunk historical budgets per component, embed every chunk in memory, and return vectors with token and cost stats.",
)
async def ingest_embeddings(
        request: EmbeddingIngestRequest,
        chunker: Chunker = Depends(get_chunker),
        embedder: OpenAIEmbedder | None = Depends(get_embedder),
        settings: Settings = Depends(get_settings),
) -> EmbeddingIngestResponse:
    """
    Chunk the submitted budgets, embed every chunk, and return vectors with aggregate statistics.

    :param request: Validated request carrying the historical budgets.
    :param chunker: Budget chunking strategy.
    :param embedder: Async embedder, or ``None`` when ``OPENAI_API_KEY`` is unset.
    :param settings: Application settings providing the embedding model name.

    :return: Embedded chunks and ``EmbeddingIngestStats``.
    :raises HTTPException: 503 if the embedder is missing or rate limited, 502 on other
        provider API errors, 500 on unexpected failures.
    """
    started_at = time.perf_counter()
    with _TRACER.start_as_current_span("embeddings.ingest") as span:
        span.set_attribute("embeddings.total_budgets", len(request.budgets))
        span.set_attribute("embeddings.model", settings.embedding_model)
        log.info("embedding_ingest_started", total_budgets=len(request.budgets))

        # 503 (not 500): a missing API key is a deployment condition, not a server bug.
        if embedder is None:
            raise _failure(
                span=span,
                reason="embedder_unavailable",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_DETAIL_UNAVAILABLE
            )

        try:
            # Chunking runs inline: tiktoken over a few hundred short chunks takes milliseconds.
            chunks = chunker.chunk(request.budgets)
            vectors = await embedder.embed_many([chunk.text for chunk in chunks])
        # RateLimitError subclasses APIError, so it must be caught first to map to 503 instead of 502.
        except RateLimitError as e:
            raise _failure(
                span=span,
                reason="rate_limited",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_DETAIL_RATE_LIMITED,
                exc=e
            ) from e
        except APIError as e:
            raise _failure(
                span=span,
                reason="upstream_api_error",
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=_DETAIL_UPSTREAM_ERROR,
                exc=e
            ) from e
        except Exception as e:
            raise _failure(
                span=span,
                reason="unexpected_error",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=_DETAIL_UNEXPECTED,
                exc=e
            ) from e

        # Local tiktoken count: embed_many returns vectors only, so this can differ slightly from billed tokens.
        total_tokens = sum(chunk.token_count for chunk in chunks)
        # estimate_cost returns None for models absent from the pricing catalog.
        estimated_cost_usd = estimate_cost(settings.embedding_model, total_tokens, 0) or 0.0
        stats = EmbeddingIngestStats(
            total_budgets=len(request.budgets),
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        span.set_attribute("embeddings.total_chunks", stats.total_chunks)
        span.set_attribute("embeddings.total_tokens", stats.total_tokens)
        span.set_attribute("embeddings.estimated_cost_usd", stats.estimated_cost_usd)
        log.info(
            "embedding_ingest_completed",
            **stats.model_dump(),
            latency_ms=round((time.perf_counter() - started_at) * 1000, 1),
        )
        # strict=True: embed_many must return exactly one vector per chunk, in order.
        return EmbeddingIngestResponse(
            chunks=[
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            stats=stats,
        )
