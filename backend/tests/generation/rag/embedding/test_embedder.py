# tests/generation/rag/embedding/test_embedder.py
"""
Offline unit tests for ``OpenAIEmbedder``, embedding settings, and DI factories.
"""

# --- Python Standard Library ---
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

# --- Third Party Libraries ---
import httpx
from openai import RateLimitError
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.dependencies import get_embedder, get_openai_client
from aie_estimator.generation.rag.embedding.embedder import OpenAIEmbedder
from aie_estimator.settings import Environment, Settings


def _make_rate_limit_error() -> RateLimitError:
    """Build a synthetic ``RateLimitError`` without network calls."""
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limit exceeded", response=response, body=None)


def _make_embedding_response(
    vectors: list[list[float]],
    total_tokens: int | None = 42,
    *,
    reverse_order: bool = False,
) -> SimpleNamespace:
    """Build a minimal ``CreateEmbeddingResponse``-compatible stub."""
    usage = SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None
    items = [SimpleNamespace(index=idx, embedding=vector) for idx, vector in enumerate(vectors)]
    if reverse_order:
        items.reverse()
    return SimpleNamespace(data=items, usage=usage)


@pytest.mark.anyio
async def test_embed_one_returns_vector_and_omits_default_dimensions() -> None:
    """
    GIVEN an OpenAIEmbedder with dimensions=None
    WHEN embed_one is awaited
    THEN it returns the single vector without passing dimensions to embeddings.create.
    """
    client = AsyncMock()
    client.embeddings.create = AsyncMock(return_value=_make_embedding_response([[0.1, 0.2, 0.3]]))
    embedder = OpenAIEmbedder(client=client)

    vector = await embedder.embed_one("OAuth 2.0 backend")

    assert vector == [0.1, 0.2, 0.3]
    client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["OAuth 2.0 backend"],
    )


@pytest.mark.anyio
async def test_embed_many_empty_sequence_skips_api() -> None:
    """
    GIVEN an empty sequence of input texts
    WHEN embed_many is awaited
    THEN it returns an empty list without calling the OpenAI API.
    """
    client = AsyncMock()
    client.embeddings.create = AsyncMock()
    embedder = OpenAIEmbedder(client=client)

    assert await embedder.embed_many([]) == []
    client.embeddings.create.assert_not_awaited()


@pytest.mark.anyio
async def test_embed_many_batches_over_100_items_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN 205 input texts and batch_size=100
    WHEN embed_many is awaited
    THEN it splits requests into 100, 100, and 5 items, preserves order, and logs each batch.
    """
    texts = [f"item-{idx}" for idx in range(205)]
    batch_sizes: list[int] = []

    async def _fake_create(**kwargs: Any) -> SimpleNamespace:
        items = kwargs["input"]
        batch_sizes.append(len(items))
        return _make_embedding_response(
            [[float(idx)] for idx in range(len(items))],
            total_tokens=len(items) * 2,
        )

    client = AsyncMock()
    client.embeddings.create = AsyncMock(side_effect=_fake_create)
    mock_log = MagicMock()
    monkeypatch.setattr("aie_estimator.generation.rag.embedding.embedder.log", mock_log)

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    embedder = OpenAIEmbedder(client=client)
    vectors = await embedder.embed_many(texts)

    assert batch_sizes == [100, 100, 5]
    assert len(vectors) == 205
    assert client.embeddings.create.call_args_list[0].kwargs["dimensions"] == 512
    assert mock_log.info.call_count == 3
    assert mock_log.info.call_args_list[0].args[0] == "embedding_batch_completed"


@pytest.mark.anyio
async def test_embed_one_recovers_after_transient_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN two transient RateLimitError responses followed by success
    WHEN embed_one is awaited
    THEN it sleeps 1s and 2s asynchronously and returns the embedding vector.
    """
    sleep_mock = AsyncMock()
    mock_log = MagicMock()
    monkeypatch.setattr("aie_estimator.generation.rag.embedding.embedder.asyncio.sleep", sleep_mock)
    monkeypatch.setattr("aie_estimator.generation.rag.embedding.embedder.log", mock_log)

    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        side_effect=[
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_embedding_response([[0.5, 0.6]]),
        ]
    )
    embedder = OpenAIEmbedder(client=client)

    vector = await embedder.embed_one("retry test")

    assert vector == [0.5, 0.6]
    assert sleep_mock.await_args_list == [call(1), call(2)]
    assert [entry.args[0] for entry in mock_log.warning.call_args_list] == [
        "embedding_batch_rate_limit_hit",
        "embedding_batch_rate_limit_hit",
    ]


@pytest.mark.anyio
async def test_embed_one_raises_after_rate_limit_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN persistent RateLimitError responses across all retry attempts
    WHEN embed_one is awaited
    THEN it sleeps 1s, 2s, and 4s and re-raises RateLimitError.
    """
    sleep_mock = AsyncMock()
    monkeypatch.setattr("aie_estimator.generation.rag.embedding.embedder.asyncio.sleep", sleep_mock)

    client = AsyncMock()
    client.embeddings.create = AsyncMock(side_effect=[_make_rate_limit_error() for _ in range(4)])
    embedder = OpenAIEmbedder(client=client)

    with pytest.raises(RateLimitError):
        await embedder.embed_one("exhausted")

    assert sleep_mock.await_args_list == [call(1), call(2), call(4)]


@pytest.mark.anyio
async def test_embed_one_propagates_non_rate_limit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a non-RateLimitError exception from AsyncOpenAI
    WHEN embed_one is awaited
    THEN it propagates immediately without sleeping.
    """
    sleep_mock = AsyncMock()
    monkeypatch.setattr("aie_estimator.generation.rag.embedding.embedder.asyncio.sleep", sleep_mock)

    client = AsyncMock()
    client.embeddings.create = AsyncMock(side_effect=RuntimeError("upstream failure"))
    embedder = OpenAIEmbedder(client=client)

    with pytest.raises(RuntimeError, match=r"upstream failure"):
        await embedder.embed_one("boom")

    sleep_mock.assert_not_awaited()


def test_embedding_settings_defaults_and_validation() -> None:
    """
    GIVEN Settings in test environment
    WHEN embedding fields are inspected or given invalid non-positive values
    THEN defaults match Session 7 contracts and non-positive values fail validation.
    """
    settings = Settings(app_env=Environment.TEST)
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.embedding_batch_size == 100

    with pytest.raises(ValidationError):
        Settings(app_env=Environment.TEST, embedding_dimensions=0)

    with pytest.raises(ValidationError):
        Settings(app_env=Environment.TEST, embedding_batch_size=0)


@pytest.mark.anyio
async def test_embed_one_rejects_empty_text() -> None:
    """
    GIVEN a blank or empty input string
    WHEN embed_one is awaited
    THEN it raises ValueError without calling the OpenAI API.
    """
    client = AsyncMock()
    client.embeddings.create = AsyncMock()
    embedder = OpenAIEmbedder(client=client)

    with pytest.raises(ValueError, match=r"must not be empty"):
        await embedder.embed_one("   ")

    client.embeddings.create.assert_not_awaited()


@pytest.mark.anyio
async def test_embed_many_sorts_out_of_order_response_data() -> None:
    """
    GIVEN an API response whose data items arrive out of index order
    WHEN embed_many is awaited
    THEN returned vectors are ordered by item.index.
    """
    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        return_value=_make_embedding_response([[1.0], [2.0]], reverse_order=True)
    )
    embedder = OpenAIEmbedder(client=client)

    vectors = await embedder.embed_many(["first", "second"])

    assert vectors == [[1.0], [2.0]]


def test_dependency_factories_handle_missing_key_caching_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN missing or configured OpenAI API keys, models, and embedding dimensions
    WHEN get_openai_client and get_embedder are called
    THEN missing keys return None, client is cached with max_retries=0, 1536 on default model omits dimensions,
    and custom dimensions or non-default models forward dimensions explicitly.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert get_openai_client() is None
    assert get_embedder(None) is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = get_openai_client()
    assert client is not None
    assert client is get_openai_client()
    assert client.max_retries == 0

    default_embedder = get_embedder(client)
    assert default_embedder is not None
    assert default_embedder._dimensions is None

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    custom_embedder = get_embedder(client)
    assert custom_embedder is not None
    assert custom_embedder._dimensions == 512

    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    large_embedder = get_embedder(client)
    assert large_embedder is not None
    assert large_embedder._dimensions == 1536

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    large_native_embedder = get_embedder(client)
    assert large_native_embedder is not None
    assert large_native_embedder._dimensions is None
