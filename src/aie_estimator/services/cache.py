# src/aie_estimator/services/cache.py

# --- Python Standard Library ---
from functools import lru_cache
import hashlib
import json
from typing import Any, Final

# --- Third Party Libraries ---
import redis.asyncio as redis
from redis.exceptions import RedisError
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

DEFAULT_CACHE_TTL: Final[int] = 86400
DEFAULT_CACHE_TIMEOUT: Final[float] = 1.0
CACHE_KEY_PREFIX: Final[str] = "estimation"


class EstimationCache:
    """Exact-match Redis cache for LLM software project estimation responses."""

    def __init__(self, redis_client: redis.Redis, ttl: int | None = None) -> None:
        """
        Initialize the cache wrapper with an active Redis client and default expiration.

        :param redis_client: Configured redis.asyncio client instance.
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
        :param socket_timeout: Connection and operation timeout in seconds.

        :return: Configured EstimationCache instance.
        """
        client = redis.Redis(
            host=host,
            port=port,
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
        reasoning_effort: str | None = None,
        preprocessing: str = "none",
        engine: str = "litellm",
    ) -> str:
        """
        Generate a deterministic SHA-256 cache key from all parameters influencing LLM output.

        :param system_prompt: Complete rendered system prompt including role and CAG examples.
        :param user_message: Input transcription or preprocessed requirements text.
        :param model: Model identifier used for inference.
        :param max_tokens: Maximum token limit for the response.
        :param reasoning_effort: Qualitative reasoning effort tier ('low', 'medium', 'high') or None.
        :param preprocessing: Preprocessing mode identifier.
        :param engine: Execution engine identifier ('litellm' or 'native').

        :return: Prefixed hexadecimal cache key string.
        """
        payload = json.dumps(
            {
                "engine": engine,
                "max_tokens": max_tokens,
                "model": model,
                "preprocessing": preprocessing,
                "reasoning_effort": reasoning_effort,
                "system_prompt": system_prompt,
                "user_message": user_message,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{CACHE_KEY_PREFIX}:{digest}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """
        Retrieve and deserialize a cached estimation response by key asynchronously.

        :param key: Cache key to lookup.

        :return: Deserialized response dictionary if present, or None on cache miss or connection error.
        """
        try:
            cached = await self.redis.get(key)
        except RedisError as e:
            log.warning("cache_get_failed", error=repr(e))
            return None

        if cached and isinstance(cached, (str, bytes, bytearray)):
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    log.info("cache_hit", key_prefix=key[:24])
                    return data
            except json.JSONDecodeError as e:
                log.warning("cache_deserialize_failed", error=repr(e))
                return None

        log.info("cache_miss", key_prefix=key[:24])
        return None

    async def set(self, key: str, response: dict[str, Any], ttl: int | None = None) -> None:
        """
        Persist an estimation response dictionary in cache with an expiration deadline asynchronously.

        :param key: Cache key identifier.
        :param response: Response dictionary to store.
        :param ttl: Optional TTL override in seconds; defaults to configured instance TTL.
        """
        effective_ttl = ttl if ttl is not None else self.ttl
        try:
            await self.redis.set(key, json.dumps(response), ex=effective_ttl)
            log.info("cache_stored", key_prefix=key[:24], ttl=effective_ttl)
        except RedisError as e:
            log.warning("cache_set_failed", error=repr(e))

    async def close(self) -> None:
        """
        Close the underlying Redis client and connection pool cleanly.
        """
        try:
            await self.redis.aclose()
            log.info("cache_client_closed")
        except RedisError as e:
            log.warning("cache_close_failed", error=repr(e))


@lru_cache(maxsize=1)
def get_estimation_cache() -> EstimationCache:
    """
    Retrieve or initialize the application-wide singleton EstimationCache instance.

    :return: Authenticated EstimationCache singleton.
    """
    settings = get_settings()
    return EstimationCache.from_host_port(
        host=settings.redis_host,
        port=settings.redis_port,
        ttl=settings.cache_ttl,
    )
