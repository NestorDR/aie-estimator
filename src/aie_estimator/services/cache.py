# src/aie_estimator/services/cache.py

# --- Python Standard Library ---
import hashlib
import json
from typing import Any, Final

# --- Third Party Libraries ---
import redis
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

DEFAULT_CACHE_TTL: Final[int] = get_settings().cache_ttl
DEFAULT_CACHE_TIMEOUT: Final[float] = 1.0
CACHE_KEY_PREFIX: Final[str] = "estimation"


class EstimationCache:
    """Exact-match Redis cache for LLM software project estimation responses."""

    def __init__(self, redis_client: redis.Redis, ttl: int | None = None) -> None:
        """
        Initialize the cache wrapper with an active Redis client and default expiration.

        :param redis_client: Configured redis-py client instance.
        :param ttl: Default time-to-live in seconds for stored estimation entries; defaults to Settings.cache_ttl.
        """
        self.redis: redis.Redis = redis_client
        self.ttl: int = ttl if ttl is not None else get_settings().cache_ttl

    @classmethod
    def from_host_port(
        cls,
        host: str = "localhost",
        port: int = 6379,
        ttl: int | None = None,
        socket_timeout: float = DEFAULT_CACHE_TIMEOUT,
    ) -> "EstimationCache":
        """
        Construct an EstimationCache instance from host and port with connection timeouts.

        :param host: Redis server hostname or IP address.
        :param port: Redis server port number.
        :param ttl: Cache retention TTL in seconds; defaults to Settings.cache_ttl.
        :param socket_timeout: Timeout in seconds for socket operations to prevent blocking.

        :return: Initialized EstimationCache instance.
        """
        client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        return cls(client, ttl=ttl)

    @classmethod
    def from_url(
        cls,
        url: str,
        ttl: int | None = None,
        socket_timeout: float = DEFAULT_CACHE_TIMEOUT,
    ) -> "EstimationCache":
        """
        Construct an EstimationCache instance from a connection URL with connection timeouts.

        :param url: Redis connection URI (e.g., redis://localhost:6379).
        :param ttl: Cache retention TTL in seconds; defaults to Settings.cache_ttl.
        :param socket_timeout: Timeout in seconds for socket operations to prevent blocking.

        :return: Initialized EstimationCache instance.
        """
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
        )
        return cls(client, ttl=ttl)

    @staticmethod
    def make_key(
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int,
        thinking_budget: int | None = None,
        preprocessing: str = "none",
    ) -> str:
        """
        Generate a deterministic SHA-256 cache key from all parameters influencing LLM output.

        :param system_prompt: Complete rendered system prompt including role and CAG examples.
        :param user_message: Input transcription or preprocessed requirements text.
        :param model: Model identifier used for inference.
        :param max_tokens: Maximum token limit for the response.
        :param thinking_budget: Extended reasoning budget (Anthropic only) or None.
        :param preprocessing: Preprocessing mode identifier.

        :return: Prefixed hexadecimal cache key string.
        """
        payload = json.dumps(
            {
                "max_tokens": max_tokens,
                "model": model,
                "preprocessing": preprocessing,
                "system_prompt": system_prompt,
                "thinking_budget": thinking_budget,
                "user_message": user_message,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{CACHE_KEY_PREFIX}:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Retrieve and deserialize a cached estimation response by key.

        :param key: Cache key to lookup.

        :return: Deserialized response dictionary if present, or None on cache miss or connection error.
        """
        try:
            cached = self.redis.get(key)
        except redis.RedisError as exc:
            log.warning("cache_get_failed", error=str(exc))
            return None

        if cached and isinstance(cached, (str, bytes, bytearray)):
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    log.info("cache_hit", key_prefix=key[:24])
                    return data
            except json.JSONDecodeError as exc:
                log.warning("cache_deserialize_failed", error=str(exc))
                return None

        log.info("cache_miss", key_prefix=key[:24])
        return None

    def set(self, key: str, response: dict[str, Any], ttl: int | None = None) -> None:
        """
        Persist an estimation response dictionary in cache with an expiration deadline.

        :param key: Cache key identifier.
        :param response: Response dictionary to store.
        :param ttl: Optional TTL override in seconds; defaults to configured instance TTL.
        """
        effective_ttl = ttl if ttl is not None else self.ttl
        try:
            self.redis.set(key, json.dumps(response), ex=effective_ttl)
            log.info("cache_stored", key_prefix=key[:24], ttl=effective_ttl)
        except redis.RedisError as exc:
            log.warning("cache_set_failed", error=str(exc))


_cache_instance: EstimationCache | None = None


def get_estimation_cache() -> EstimationCache:
    """
    Retrieve or initialize the application-wide singleton EstimationCache instance.

    :return: Authenticated EstimationCache singleton.
    """
    global _cache_instance
    if _cache_instance is None:
        settings = get_settings()
        _cache_instance = EstimationCache.from_host_port(
            host=settings.redis_host,
            port=settings.redis_port,
            ttl=settings.cache_ttl,
        )
    return _cache_instance


def _reset_estimation_cache() -> None:
    """
    Reset cached singleton instance to force re-initialization.

    Used primarily in testing environments when settings or environment variables are overridden.
    """
    global _cache_instance
    _cache_instance = None
