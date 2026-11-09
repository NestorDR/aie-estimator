# src/aie_estimator/services/cache/exact.py

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
from aie_estimator.schemas.estimations import EstimationRequest
from aie_estimator.services.providers.generation_options import GenerationOptions
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

DEFAULT_CACHE_TIMEOUT: Final[float] = 1.0
CACHE_KEY_PREFIX: Final[str] = "estimation"


class ExactEstimationCache:
    """Exact-match Redis cache for LLM software project estimation responses."""

    def __init__(self, redis_client: redis.Redis, ttl: int | None = None) -> None:
        """
        Initialize the cache wrapper with an active Redis client and default expiration.

        :param redis_client: Configured redis.asyncio client instance.
        :param ttl: Default time-to-live in seconds for stored estimation entries; defaults to Settings.exact_cache_ttl.
        """
        settings = get_settings()
        self.redis: redis.Redis = redis_client
        self.ttl: int = ttl if ttl is not None else getattr(settings, "exact_cache_ttl", settings.cache_ttl)

    @classmethod
    def from_host_port(
        cls,
        host: str = "localhost",
        port: int = 6379,
        ttl: int | None = None,
        socket_timeout: float = DEFAULT_CACHE_TIMEOUT,
    ) -> "ExactEstimationCache":
        """
        Construct an ExactEstimationCache instance from host and port with connection timeouts.

        :param host: Redis server hostname or IP address.
        :param port: Redis server port number.
        :param ttl: Cache retention TTL in seconds; defaults to Settings.exact_cache_ttl.
        :param socket_timeout: Connection and operation timeout in seconds.

        :return: Configured ExactEstimationCache instance.
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
        request: EstimationRequest,
        prompt_version: str,
        opts: GenerationOptions,
    ) -> str:
        """
        Generate a deterministic SHA-256 cache key from typed request and generation options.

        :param request: Validated EstimationRequest domain object.
        :param prompt_version: Active prompt template version.
        :param opts: GenerationOptions controlling model, token limits, and engine.

        :return: Prefixed hexadecimal cache key string.
        """
        payload = json.dumps(
            {
                "description": request.description,
                "detail_level": request.detail_level.value,
                "engine": opts.engine,
                "max_tokens": opts.max_tokens,
                "model": opts.effective_model,
                "output_format": request.output_format.value,
                "project_type": request.project_type.value,
                "prompt_version": prompt_version,
                "reasoning_effort": opts.reasoning_effort,
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
            log.warning("exact_cache_lookup_failed", error=repr(e))
            return None

        if cached and isinstance(cached, (str, bytes, bytearray)):
            try:
                data = json.loads(cached)
                if isinstance(data, dict):
                    log.info("exact_cache_entry_retrieved", key_prefix=key[:24])
                    return data
            except json.JSONDecodeError as e:
                log.warning("exact_cache_deserialization_failed", error=repr(e))
                return None

        log.info("exact_cache_entry_missed", key_prefix=key[:24])
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
            log.info("exact_cache_entry_stored", key_prefix=key[:24], ttl=effective_ttl)
        except RedisError as e:
            log.warning("exact_cache_store_failed", error=repr(e))

    async def close(self) -> None:
        """
        Close the underlying Redis client and connection pool cleanly.
        """
        try:
            await self.redis.aclose()
            log.info("exact_cache_client_closed")
        except RedisError as e:
            log.warning("exact_cache_close_failed", error=repr(e))


@lru_cache(maxsize=1)
def get_exact_cache() -> ExactEstimationCache:
    """
    Retrieve or initialize the application-wide singleton ExactEstimationCache instance.

    :return: Authenticated ExactEstimationCache singleton.
    """
    settings = get_settings()
    effective_ttl = getattr(settings, "exact_cache_ttl", settings.cache_ttl)
    return ExactEstimationCache.from_host_port(
        host=settings.redis_host,
        port=settings.redis_port,
        ttl=effective_ttl,
    )

