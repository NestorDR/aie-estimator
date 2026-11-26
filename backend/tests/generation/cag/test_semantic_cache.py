# tests/generation/cag/test_semantic_cache.py

# --- Python Standard Library ---
from collections.abc import Callable
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError, ResponseError

# --- App modules ---
from aie_estimator.domain.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    PmEstimation,
    OutputFormat,
    ProjectType,
)
from aie_estimator.generation.cag.semantic import (
    DEFAULT_INDEX_NAME,
    SEMANTIC_KEY_PREFIX,
    EstimationSemanticCache,
    get_semantic_cache,
    make_semantic_key,
    vector_to_bytes,
)
from aie_estimator.settings import Environment, Settings, get_settings
from aie_estimator.utils.embeddings_utils import DEFAULT_EMBEDDING_DIMENSIONS, embedding_signature


@pytest.fixture
def sample_request() -> EstimationRequest:
    """Provide a validated EstimationRequest instance for cache tests."""
    return EstimationRequest(
        description="Mobile application with user authentication, live chat, and push notifications.",
        project_type=ProjectType.MOBILE_APP,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )


def test_vector_to_bytes_format_and_length() -> None:
    """
    GIVEN a list of 1,536 float values
    WHEN vector_to_bytes is invoked
    THEN the returned bytes object has length 6,144 and unpacks to the exact original floats.
    """
    vector = [float(i) for i in range(DEFAULT_EMBEDDING_DIMENSIONS)]
    packed = vector_to_bytes(vector)

    assert isinstance(packed, bytes)
    assert len(packed) == DEFAULT_EMBEDDING_DIMENSIONS * 4

    unpacked = list(struct.unpack(f"{DEFAULT_EMBEDDING_DIMENSIONS}f", packed))
    assert unpacked == vector


def test_make_semantic_key_deterministic() -> None:
    """
    GIVEN identical bucket and description parameters
    WHEN make_semantic_key is executed multiple times
    THEN identical SHA-256 prefixed keys are returned.
    """
    bucket = "v1:mobile_app:medium:phases_table"
    description = "Mobile app with login and chat."

    key_1 = make_semantic_key(bucket, description)
    key_2 = make_semantic_key(bucket, description)

    assert key_1 == key_2
    assert key_1.startswith(f"{SEMANTIC_KEY_PREFIX}:")
    assert len(key_1) == len(SEMANTIC_KEY_PREFIX) + 1 + 64


def test_bucket_for_construction(sample_request: EstimationRequest) -> None:
    """
    GIVEN an EstimationRequest and prompt version
    WHEN bucket_for is computed
    THEN the formatted underscore-separated bucket tag string is produced.
    """
    bucket = EstimationSemanticCache.bucket_for(sample_request, prompt_version="v2")
    assert bucket == "v2_pm_mobile_app_medium_phases_table"


@pytest.mark.anyio
async def test_initialize_index_creates_schema() -> None:
    """
    GIVEN an active Redis client and new index name
    WHEN initialize_index is invoked
    THEN create_index is executed on the RediSearch client with schema fields and Hash definition.
    """
    mock_redis = AsyncMock()
    mock_ft = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    cache = EstimationSemanticCache(redis_client=mock_redis)
    await cache.initialize_index()

    expected_signature = embedding_signature(cache.embedding_model, cache.embedding_dimensions)
    assert cache.index_name == f"{DEFAULT_INDEX_NAME}_{expected_signature}"
    mock_redis.ft.assert_called_once_with(cache.index_name)
    mock_ft.create_index.assert_called_once()
    call_kwargs = mock_ft.create_index.call_args.kwargs
    assert "fields" in call_kwargs
    assert "definition" in call_kwargs

    vector_field = next(field for field in call_kwargs["fields"] if field.name == "embedding")
    dim_idx = vector_field.args.index("DIM")
    assert vector_field.args[dim_idx + 1] == cache.embedding_dimensions

    definition_args = call_kwargs["definition"].args
    prefix_idx = definition_args.index("PREFIX")
    assert definition_args[prefix_idx + 2] == f"{cache.key_namespace}:"


@pytest.mark.anyio
async def test_initialize_index_idempotent_when_already_exists() -> None:
    """
    GIVEN a RediSearch instance where the index already exists
    WHEN initialize_index is invoked and raises ResponseError('Index already exists')
    THEN the error is caught and suppressed without raising to the caller.
    """
    mock_redis = AsyncMock()
    mock_ft = AsyncMock()
    mock_ft.create_index.side_effect = ResponseError("Index already exists")
    mock_redis.ft = MagicMock(return_value=mock_ft)

    cache = EstimationSemanticCache(redis_client=mock_redis)
    await cache.initialize_index()

    mock_ft.create_index.assert_called_once()


@pytest.mark.anyio
async def test_initialize_index_handles_redis_error() -> None:
    """
    GIVEN a Redis client encountering network or connection issues
    WHEN initialize_index is executed and raises RedisError
    THEN the exception is caught and suppressed gracefully.
    """
    mock_redis = AsyncMock()
    mock_ft = AsyncMock()
    mock_ft.create_index.side_effect = RedisConnectionError("Cannot connect")
    mock_redis.ft = MagicMock(return_value=mock_ft)

    cache = EstimationSemanticCache(redis_client=mock_redis)
    await cache.initialize_index()

    mock_ft.create_index.assert_called_once()


@pytest.mark.anyio
async def test_get_embedding_success_and_failure() -> None:
    """
    GIVEN an AsyncOpenAI client
    WHEN get_embedding is called on success and on error
    THEN the vector list is returned on success and None is returned on exception.
    """
    mock_openai = AsyncMock()
    mock_embedding_data = MagicMock(embedding=[0.01] * DEFAULT_EMBEDDING_DIMENSIONS)
    mock_openai.embeddings.create.return_value = MagicMock(data=[mock_embedding_data])

    cache = EstimationSemanticCache(redis_client=AsyncMock(), openai_client=mock_openai)
    vector = await cache.get_embedding("Test text")
    assert vector == [0.01] * DEFAULT_EMBEDDING_DIMENSIONS
    mock_openai.embeddings.create.assert_awaited_once_with(
        model=cache.embedding_model,
        input="Test text",
    )

    mock_openai.embeddings.create.side_effect = RuntimeError("OpenAI API Down")
    failing_vector = await cache.get_embedding("Test text")
    assert failing_vector is None


@pytest.mark.anyio
async def test_semantic_cache_lookup_hit(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a stored estimation with cosine distance 0.05 (similarity 0.95 >= threshold 0.92)
    WHEN lookup is invoked
    THEN the cached PmEstimation is deserialized and returned.
    """
    sample_payload = make_estimation_result()
    serialized_result = PmEstimation.model_validate(sample_payload["result"]).model_dump_json()

    mock_doc = MagicMock()
    mock_doc.vector_distance = "0.05"
    mock_doc.result_json = serialized_result

    mock_ft = AsyncMock()
    mock_ft.search.return_value = MagicMock(docs=[mock_doc])

    mock_redis = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.02] * DEFAULT_EMBEDDING_DIMENSIONS)])

    cache = EstimationSemanticCache(
        redis_client=mock_redis,
        openai_client=mock_openai,
        threshold=0.92,
    )

    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is not None
    assert isinstance(result, PmEstimation)
    assert result.total_cost_eur == sample_payload["result"]["total_cost_eur"]
    mock_ft.search.assert_called_once()


@pytest.mark.anyio
async def test_semantic_cache_lookup_miss_empty_index(
    sample_request: EstimationRequest,
) -> None:
    """
    GIVEN an empty RediSearch index
    WHEN lookup is invoked and search returns no documents
    THEN None is returned.
    """
    mock_ft = AsyncMock()
    mock_ft.search.return_value = MagicMock(docs=[])

    mock_redis = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.02] * DEFAULT_EMBEDDING_DIMENSIONS)])

    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai)
    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is None


@pytest.mark.anyio
async def test_semantic_cache_lookup_miss_below_threshold(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a nearest neighbor with cosine distance 0.15 (similarity 0.85 < threshold 0.92)
    WHEN lookup is invoked
    THEN the candidate is rejected as below_threshold and None is returned.
    """
    sample_payload = make_estimation_result()
    serialized_result = PmEstimation.model_validate(sample_payload["result"]).model_dump_json()

    mock_doc = MagicMock()
    mock_doc.vector_distance = "0.15"  # sim = 0.85 < 0.92
    mock_doc.result_json = serialized_result

    mock_ft = AsyncMock()
    mock_ft.search.return_value = MagicMock(docs=[mock_doc])

    mock_redis = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.02] * DEFAULT_EMBEDDING_DIMENSIONS)])

    cache = EstimationSemanticCache(
        redis_client=mock_redis,
        openai_client=mock_openai,
        threshold=0.92,
    )

    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is None


@pytest.mark.anyio
async def test_semantic_cache_lookup_log_only_mode(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a high similarity match (0.95 >= 0.92) but log_only is enabled
    WHEN lookup is invoked
    THEN None is returned so the request proceeds to LLM generation.
    """
    sample_payload = make_estimation_result()
    serialized_result = PmEstimation.model_validate(sample_payload["result"]).model_dump_json()

    mock_doc = MagicMock()
    mock_doc.vector_distance = "0.05"
    mock_doc.result_json = serialized_result

    mock_ft = AsyncMock()
    mock_ft.search.return_value = MagicMock(docs=[mock_doc])

    mock_redis = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.02] * DEFAULT_EMBEDDING_DIMENSIONS)])

    cache = EstimationSemanticCache(
        redis_client=mock_redis,
        openai_client=mock_openai,
        threshold=0.92,
        log_only=True,
    )

    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is None


@pytest.mark.anyio
async def test_semantic_cache_lookup_embedding_failure_returns_none(
    sample_request: EstimationRequest,
) -> None:
    """
    GIVEN an embedding generation failure
    WHEN lookup is invoked
    THEN lookup fails open by returning None without executing vector search.
    """
    mock_openai = AsyncMock()
    mock_openai.embeddings.create.side_effect = RuntimeError("Embedding unavailable")

    mock_redis = AsyncMock()
    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai)

    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is None
    mock_redis.ft.assert_not_called()


@pytest.mark.anyio
async def test_semantic_cache_lookup_redis_error_returns_none(
    sample_request: EstimationRequest,
) -> None:
    """
    GIVEN a RediSearch search error
    WHEN lookup is executed
    THEN the error is caught and None is returned.
    """
    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.02] * DEFAULT_EMBEDDING_DIMENSIONS)])

    mock_ft = AsyncMock()
    mock_ft.search.side_effect = RedisConnectionError("Search connection lost")

    mock_redis = AsyncMock()
    mock_redis.ft = MagicMock(return_value=mock_ft)

    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai)
    result = await cache.lookup(sample_request, prompt_version="v1")
    assert result is None


@pytest.mark.anyio
async def test_semantic_cache_store_success(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a validated PmEstimation and prompt version
    WHEN store is executed
    THEN redis.hset is called with bucket, result_json, and float32 binary embedding bytes.
    """
    sample_payload = make_estimation_result()
    structured_result = PmEstimation.model_validate(sample_payload["result"])

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.03] * DEFAULT_EMBEDDING_DIMENSIONS)])

    mock_redis = AsyncMock()
    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai, ttl=3600)

    await cache.store(sample_request, structured_result, prompt_version="v1")

    mock_redis.hset.assert_called_once()
    call_args = mock_redis.hset.call_args
    assert call_args.args[0].startswith(f"{cache.key_namespace}:")
    mapping = call_args.kwargs["mapping"]
    assert mapping["bucket"] == "v1_pm_mobile_app_medium_phases_table"
    assert "result_json" in mapping
    assert isinstance(mapping["embedding"], bytes)
    assert len(mapping["embedding"]) == DEFAULT_EMBEDDING_DIMENSIONS * 4

    mock_redis.expire.assert_called_once_with(call_args.args[0], 3600)


@pytest.mark.anyio
async def test_semantic_cache_store_embedding_failure_skips_write(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN an embedding generation error during cache write
    WHEN store is executed
    THEN Redis hset is not called and the failure is handled gracefully.
    """
    sample_payload = make_estimation_result()
    structured_result = PmEstimation.model_validate(sample_payload["result"])

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.side_effect = RuntimeError("Embedding model offline")

    mock_redis = AsyncMock()
    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai)

    await cache.store(sample_request, structured_result, prompt_version="v1")
    mock_redis.hset.assert_not_called()


@pytest.mark.anyio
async def test_semantic_cache_store_redis_error_suppressed(
    sample_request: EstimationRequest,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a Redis hset write failure
    WHEN store is executed
    THEN the exception is caught without raising.
    """
    sample_payload = make_estimation_result()
    structured_result = PmEstimation.model_validate(sample_payload["result"])

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.03] * DEFAULT_EMBEDDING_DIMENSIONS)])

    mock_redis = AsyncMock()
    mock_redis.hset.side_effect = RedisConnectionError("Write timed out")

    cache = EstimationSemanticCache(redis_client=mock_redis, openai_client=mock_openai)
    await cache.store(sample_request, structured_result, prompt_version="v1")
    mock_redis.hset.assert_called_once()


@pytest.mark.anyio
async def test_semantic_cache_close() -> None:
    """
    GIVEN an active EstimationSemanticCache instance
    WHEN close is executed
    THEN redis.aclose is called.
    """
    mock_redis = AsyncMock()
    cache = EstimationSemanticCache(redis_client=mock_redis)
    await cache.close()
    mock_redis.aclose.assert_called_once()


def test_get_semantic_cache_builds_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured Settings
    WHEN get_semantic_cache is invoked
    THEN an EstimationSemanticCache instance is initialized with settings attributes.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        redis_host="semantic.local",
        redis_port=6381,
        semantic_cache_threshold=0.89,
        semantic_cache_log_only=True,
    )
    monkeypatch.setattr("aie_estimator.generation.cag.semantic.get_settings", lambda: test_settings)

    with patch("redis.asyncio.Redis", return_value=AsyncMock()) as mock_redis_cls:
        get_semantic_cache.cache_clear()
        cache = get_semantic_cache()
        get_semantic_cache.cache_clear()

    assert cache.threshold == 0.89
    assert cache.log_only is True
    mock_redis_cls.assert_called_once_with(
        host="semantic.local",
        port=6381,
        decode_responses=False,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
    )


def test_semantic_cache_versioning_and_explicit_index_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN default and overridden embedding model/dimension settings
    WHEN EstimationSemanticCache instances and semantic keys are created
    THEN index names and key namespaces track the embedding signature, explicit index_name overrides the default,
    and different embedding spaces never produce colliding keys.
    """
    small_cache = EstimationSemanticCache(redis_client=AsyncMock())
    override_cache = EstimationSemanticCache(redis_client=AsyncMock(), index_name="custom_idx")

    assert small_cache.index_name == "estimations_idx_text_embedding_3_small_1536"
    assert small_cache.key_namespace == "estimation:semantic:text_embedding_3_small_1536"
    assert override_cache.index_name == "custom_idx"

    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    get_settings.cache_clear()
    large_cache = EstimationSemanticCache(redis_client=AsyncMock())

    assert large_cache.index_name == "estimations_idx_text_embedding_3_large_3072"
    assert large_cache.key_namespace == "estimation:semantic:text_embedding_3_large_3072"
    assert small_cache.index_name != large_cache.index_name

    bucket = "v1_pm_mobile_app_medium_phases_table"
    description = "Mobile app with login and chat."
    small_key = make_semantic_key(bucket, description, namespace=small_cache.key_namespace)
    large_key = make_semantic_key(bucket, description, namespace=large_cache.key_namespace)

    assert small_key != large_key
    assert small_key.startswith(f"{small_cache.key_namespace}:")
    assert len(small_key) == len(small_cache.key_namespace) + 1 + 64


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("model", "dimensions", "expected_dimensions"),
    [
        ("text-embedding-3-small", 512, 512),
        ("text-embedding-3-large", 3072, None),
        ("text-embedding-3-large", 1536, 1536),
    ],
)
async def test_get_embedding_dimensions_forwarding(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    dimensions: int,
    expected_dimensions: int | None,
) -> None:
    """
    GIVEN configured embedding model and dimensions
    WHEN get_embedding invokes OpenAI embeddings.create
    THEN dimensions is forwarded for non-native sizes and omitted for native sizes.
    """
    monkeypatch.setenv("EMBEDDING_MODEL", model)
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(dimensions))
    get_settings.cache_clear()

    mock_openai = AsyncMock()
    mock_openai.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.01] * dimensions)])
    cache = EstimationSemanticCache(redis_client=AsyncMock(), openai_client=mock_openai)

    vector = await cache.get_embedding("Test text")
    assert vector == [0.01] * dimensions

    expected_kwargs: dict[str, Any] = {"model": model, "input": "Test text"}
    if expected_dimensions is not None:
        expected_kwargs["dimensions"] = expected_dimensions
    mock_openai.embeddings.create.assert_awaited_once_with(**expected_kwargs)
