# tests/services/test_exact_cache.py

# --- Python Standard Library ---
from collections.abc import Callable
import json
from typing import Any
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

# --- App modules ---
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from aie_estimator.services.cache.exact import (
    CACHE_KEY_PREFIX,
    EstimationExactCache,
    get_exact_cache,
)
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.settings import Environment, Settings
from aie_estimator.utils.model_constants import DEFAULT_CLAUDE_LLM, DEFAULT_GPT_LLM


def test_make_key_deterministic_output() -> None:
    """
    GIVEN identical estimation generation parameters
    WHEN make_key is computed multiple times
    THEN the returned hexadecimal digest keys are identical.
    """
    req = EstimationRequest(
        description="Client requires web dashboard with user management.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    opts = GenerationOptions(model=DEFAULT_GPT_LLM, max_tokens=4000)

    key_a = EstimationExactCache.make_key(
        request=req,
        prompt_version="v1",
        opts=opts,
    )
    key_b = EstimationExactCache.make_key(
        request=req,
        prompt_version="v1",
        opts=opts,
    )

    assert key_a == key_b
    assert key_a.startswith(f"{CACHE_KEY_PREFIX}:")
    assert len(key_a) == len(CACHE_KEY_PREFIX) + 1 + 64


@pytest.mark.parametrize(
    ("override_request", "override_prompt_version", "override_opts"),
    [
        (None, "v2", None),
        (None, None, GenerationOptions(model=DEFAULT_CLAUDE_LLM)),
        (None, None, GenerationOptions(max_tokens=2000)),
        (None, None, GenerationOptions(reasoning_effort="low")),
        (None, None, GenerationOptions(engine="native")),
        (
            EstimationRequest(
                description="Completely different project requirements.",
                project_type=ProjectType.MOBILE_APP,
                detail_level=DetailLevel.DETAILED,
                output_format=OutputFormat.LINE_ITEMS,
            ),
            None,
            None,
        ),
    ],
    ids=[
        "different_prompt_version",
        "different_model",
        "different_max_tokens",
        "different_reasoning_effort",
        "different_engine",
        "different_request",
    ],
)
def test_make_key_differentiates_on_parameter_changes(
    override_request: EstimationRequest | None,
    override_prompt_version: str | None,
    override_opts: GenerationOptions | None,
) -> None:
    """
    GIVEN baseline parameters and a single parameter alteration
    WHEN make_key is computed for both parameter sets
    THEN the resulting keys differ.

    :param override_request: Optional replacement EstimationRequest.
    :param override_prompt_version: Optional replacement prompt version string.
    :param override_opts: Optional replacement GenerationOptions.
    """
    base_req = EstimationRequest(
        description="Baseline user description for estimation.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    base_opts = GenerationOptions(model=DEFAULT_GPT_LLM, max_tokens=4000)
    base_prompt_version = "v1"

    base_key = EstimationExactCache.make_key(
        request=base_req,
        prompt_version=base_prompt_version,
        opts=base_opts,
    )

    modified_key = EstimationExactCache.make_key(
        request=override_request or base_req,
        prompt_version=override_prompt_version or base_prompt_version,
        opts=override_opts or base_opts,
    )

    assert base_key != modified_key


@pytest.mark.anyio
async def test_cache_get_hit_deserializes_payload(
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a key present in Redis storing valid JSON
    WHEN cache.get is invoked
    THEN the stored dictionary is deserialized and returned.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    """
    sample_payload = make_estimation_result(cost_usd=0.0001)
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(sample_payload)

    cache = EstimationExactCache(mock_redis, ttl=3600)
    result = await cache.get("estimation:test-key")

    assert result == sample_payload
    mock_redis.get.assert_called_once_with("estimation:test-key")


@pytest.mark.anyio
async def test_cache_get_miss_returns_none() -> None:
    """
    GIVEN a key not found in Redis
    WHEN cache.get is invoked
    THEN None is returned.
    """
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    cache = EstimationExactCache(mock_redis)
    result = await cache.get("estimation:non-existent-key")

    assert result is None
    mock_redis.get.assert_called_once_with("estimation:non-existent-key")


@pytest.mark.anyio
async def test_cache_set_stores_serialized_payload_with_ttl(
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN an estimation payload and optional custom TTL
    WHEN cache.set is invoked
    THEN redis.set is called with the serialized payload and TTL expiration parameter.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    """
    sample_payload = make_estimation_result(cost_usd=0.0001)
    mock_redis = AsyncMock()
    cache = EstimationExactCache(mock_redis)

    await cache.set("estimation:test-key", sample_payload, ttl=1800)

    mock_redis.set.assert_called_once_with(
        "estimation:test-key",
        json.dumps(sample_payload),
        ex=1800,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method_name", "args", "expected_result"),
    [
        ("get", ("estimation:failing-key",), None),
        ("set", ("estimation:failing-key", {"data": 1}), None),
        ("close", (), None),
    ],
    ids=["get_connection_error", "set_connection_error", "close_connection_error"],
)
async def test_cache_graceful_degradation_on_redis_error(
    method_name: str,
    args: tuple[Any, ...],
    expected_result: None,
) -> None:
    """
    GIVEN a Redis client that raises redis.exceptions.RedisError
    WHEN cache operations are executed
    THEN exceptions are caught and suppressed without raising to the caller.

    :param method_name: Method name on EstimationExactCache to invoke.
    :param args: Arguments to pass to the method.
    :param expected_result: Value the suppressed call returns.
    """
    mock_redis = AsyncMock()
    mock_redis.get.side_effect = RedisConnectionError("Connection refused")
    mock_redis.set.side_effect = RedisTimeoutError("Timeout occurred")
    mock_redis.aclose.side_effect = RedisConnectionError("Close failed")

    cache = EstimationExactCache(mock_redis)
    method = getattr(cache, method_name)

    assert await method(*args) is expected_result


@pytest.mark.anyio
async def test_cache_close_calls_underlying_aclose() -> None:
    """
    GIVEN an active EstimationExactCache instance
    WHEN close is invoked
    THEN the underlying Redis client aclose method is executed.
    """
    mock_redis = AsyncMock()
    cache = EstimationExactCache(mock_redis)
    await cache.close()
    mock_redis.aclose.assert_called_once()


def test_get_exact_cache_builds_client_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured Redis settings
    WHEN get_exact_cache is invoked
    THEN the Redis client and TTL come from settings.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        redis_host="cache.local",
        redis_port=6380,
        exact_cache_ttl=1800,
    )
    monkeypatch.setattr("aie_estimator.services.cache.exact.get_settings", lambda: test_settings)

    with patch("redis.asyncio.Redis", return_value=AsyncMock()) as mock_redis_cls:
        get_exact_cache.cache_clear()
        cache = get_exact_cache()
        get_exact_cache.cache_clear()

    assert cache.ttl == 1800
    mock_redis_cls.assert_called_once_with(
        host="cache.local",
        port=6380,
        decode_responses=True,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
    )
