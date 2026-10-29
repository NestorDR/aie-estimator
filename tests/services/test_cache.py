# tests/services/test_cache.py

# --- Python Standard Library ---
import json
from typing import Any, Final
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pytest
import redis

# --- App modules ---
from aie_estimator.services.cache import (
    CACHE_KEY_PREFIX,
    DEFAULT_CACHE_TTL,
    EstimationCache,
    _reset_estimation_cache,
    get_estimation_cache,
)
from aie_estimator.settings import Environment, Settings, get_settings

_SAMPLE_ESTIMATION_PAYLOAD: Final[dict[str, Any]] = {
    "estimation": "## Project Summary\nEstimation details...",
    "model": "gpt-4o-mini",
    "provider": "openai",
    "finish_reason": "stop",
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "preprocessing_input_tokens": 0,
        "preprocessing_output_tokens": 0,
    },
    "cost_usd": 0.0001,
}


def test_make_key_deterministic_output() -> None:
    """
    GIVEN identical estimation generation parameters
    WHEN make_key is computed multiple times
    THEN the returned hexadecimal digest keys are identical.
    """
    key_a = EstimationCache.make_key(
        system_prompt="System instructions with few-shot examples.",
        user_message="Client requires web dashboard.",
        model="gpt-4o-mini",
        max_tokens=4000,
        thinking_budget=None,
        preprocessing="none",
    )
    key_b = EstimationCache.make_key(
        system_prompt="System instructions with few-shot examples.",
        user_message="Client requires web dashboard.",
        model="gpt-4o-mini",
        max_tokens=4000,
        thinking_budget=None,
        preprocessing="none",
    )

    assert key_a == key_b
    assert key_a.startswith(f"{CACHE_KEY_PREFIX}:")
    assert len(key_a) == len(CACHE_KEY_PREFIX) + 1 + 64


@pytest.mark.parametrize(
    ("param_override", "expected_different"),
    [
        ({"system_prompt": "Updated system instructions."}, True),
        ({"user_message": "Different client requirements."}, True),
        ({"model": "claude-haiku-4-5-20251001"}, True),
        ({"max_tokens": 2000}, True),
        ({"thinking_budget": 1024}, True),
        ({"preprocessing": "inline_cleaning"}, True),
    ],
    ids=[
        "different_system_prompt",
        "different_user_message",
        "different_model",
        "different_max_tokens",
        "different_thinking_budget",
        "different_preprocessing",
    ],
)
def test_make_key_differentiates_on_parameter_changes(
    param_override: dict[str, Any],
    expected_different: bool,
) -> None:
    """
    GIVEN baseline parameters and a single parameter alteration
    WHEN make_key is computed for both parameter sets
    THEN the resulting keys differ.

    :param param_override: Dictionary of keyword argument overrides.
    :param expected_different: Whether the generated key must differ from baseline.
    """
    base_params: dict[str, Any] = {
        "system_prompt": "Baseline system prompt.",
        "user_message": "Baseline user message.",
        "model": "gpt-4o-mini",
        "max_tokens": 4000,
        "thinking_budget": None,
        "preprocessing": "none",
    }
    modified_params = {**base_params, **param_override}

    base_key = EstimationCache.make_key(**base_params)
    modified_key = EstimationCache.make_key(**modified_params)

    assert (base_key != modified_key) is expected_different


def test_cache_get_hit_deserializes_payload() -> None:
    """
    GIVEN a key present in Redis storing valid JSON
    WHEN cache.get is invoked
    THEN the stored dictionary is deserialized and returned.
    """
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.get.return_value = json.dumps(_SAMPLE_ESTIMATION_PAYLOAD)

    cache = EstimationCache(mock_redis, ttl=3600)
    result = cache.get("estimation:test-key")

    assert result == _SAMPLE_ESTIMATION_PAYLOAD
    mock_redis.get.assert_called_once_with("estimation:test-key")


def test_cache_get_miss_returns_none() -> None:
    """
    GIVEN a key not found in Redis
    WHEN cache.get is invoked
    THEN None is returned.
    """
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.get.return_value = None

    cache = EstimationCache(mock_redis)
    result = cache.get("estimation:non-existent-key")

    assert result is None
    mock_redis.get.assert_called_once_with("estimation:non-existent-key")


def test_cache_default_ttl_resolved_from_settings() -> None:
    """
    GIVEN an EstimationCache initialized without explicit TTL
    WHEN inspected
    THEN its ttl property reflects the value from get_settings().cache_ttl.
    """
    mock_redis = MagicMock(spec=redis.Redis)
    cache = EstimationCache(mock_redis)
    assert cache.ttl == get_settings().cache_ttl
    assert DEFAULT_CACHE_TTL == get_settings().cache_ttl


def test_cache_set_stores_serialized_payload_with_ttl() -> None:
    """
    GIVEN an estimation payload and optional custom TTL
    WHEN cache.set is invoked
    THEN redis.set is called with the serialized payload and TTL expiration parameter.
    """
    mock_redis = MagicMock(spec=redis.Redis)
    cache = EstimationCache(mock_redis, ttl=DEFAULT_CACHE_TTL)

    cache.set("estimation:test-key", _SAMPLE_ESTIMATION_PAYLOAD, ttl=1800)

    mock_redis.set.assert_called_once_with(
        "estimation:test-key",
        json.dumps(_SAMPLE_ESTIMATION_PAYLOAD),
        ex=1800,
    )


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("get", ("estimation:failing-key",)),
        ("set", ("estimation:failing-key", {"data": 1})),
    ],
    ids=["get_connection_error", "set_connection_error"],
)
def test_cache_graceful_degradation_on_redis_error(
    method_name: str,
    args: tuple[Any, ...],
) -> None:
    """
    GIVEN a Redis client that raises redis.RedisError
    WHEN cache operations are executed
    THEN exceptions are caught and suppressed without raising to the caller.

    :param method_name: Method name on EstimationCache to invoke.
    :param args: Arguments to pass to the method.
    """
    mock_redis = MagicMock(spec=redis.Redis)
    mock_redis.get.side_effect = redis.ConnectionError("Connection refused")
    mock_redis.set.side_effect = redis.TimeoutError("Timeout occurred")

    cache = EstimationCache(mock_redis)
    method = getattr(cache, method_name)

    result = method(*args)
    if method_name == "get":
        assert result is None


def test_estimation_cache_factories() -> None:
    """
    GIVEN host, port, or connection URL
    WHEN from_host_port or from_url factory methods are executed
    THEN configured EstimationCache instances are constructed.
    """
    mock_client = MagicMock()

    with patch("redis.Redis", return_value=mock_client) as mock_redis_cls:
        cache_host_port = EstimationCache.from_host_port(host="cache.local", port=6380, ttl=1800)
        assert isinstance(cache_host_port, EstimationCache)
        assert cache_host_port.ttl == 1800
        mock_redis_cls.assert_called_once_with(
            host="cache.local",
            port=6380,
            decode_responses=True,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )

    with patch("redis.from_url", return_value=mock_client) as mock_from_url:
        cache_url = EstimationCache.from_url("redis://localhost:6379", ttl=3600)
        assert isinstance(cache_url, EstimationCache)
        assert cache_url.ttl == 3600
        mock_from_url.assert_called_once_with(
            "redis://localhost:6379",
            decode_responses=True,
            socket_timeout=1.0,
            socket_connect_timeout=1.0,
        )


def test_get_estimation_cache_singleton_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN configured Redis settings
    WHEN get_estimation_cache is invoked multiple times
    THEN the same instance is returned until explicitly reset.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    test_settings = Settings(
        app_env=Environment.TEST,
        redis_host="localhost",
        redis_port=6379,
        cache_ttl=3600,
    )
    monkeypatch.setattr("aie_estimator.services.cache.get_settings", lambda: test_settings)

    mock_client = MagicMock()
    with patch("redis.Redis", return_value=mock_client) as mock_redis_cls:
        _reset_estimation_cache()

        cache_1 = get_estimation_cache()
        cache_2 = get_estimation_cache()

        assert cache_1 is cache_2
        assert mock_redis_cls.call_count == 1

        _reset_estimation_cache()
        cache_3 = get_estimation_cache()
        assert cache_3 is not cache_1
        assert mock_redis_cls.call_count == 2
