# src/aie_estimator/services/cache/semantic.py

# --- Python Standard Library ---
import asyncio
from functools import lru_cache
import hashlib
import re
import struct
from typing import Final

# --- Third Party Libraries ---
from openai import AsyncOpenAI
import redis.asyncio as redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import RedisError, ResponseError
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResult
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

# Default RediSearch index identifier and key prefix constants.
DEFAULT_INDEX_NAME: Final[str] = "estimations_idx"
SEMANTIC_KEY_PREFIX: Final[str] = "estimation:semantic"
EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: Final[int] = 1536
DEFAULT_SOCKET_TIMEOUT: Final[float] = 1.0
_TAG_SPECIAL_CHARS: Final[re.Pattern[str]] = re.compile(r"([,.<>{}\[\]\"':;!@#$%^&*()\-+=~|/\\\s])")


def escape_tag(value: str) -> str:
    """
    Escape RediSearch tag punctuation so the value is matched as one literal tag.
    Inside a RediSearch tag filter ({...}), punctuation such as `:` `.` `-` and spaces must be backslash-escaped.

    :param value: Raw tag value (e.g. the semantic cache bucket).
    :return: Value with special characters backslash-escaped.
    """
    return _TAG_SPECIAL_CHARS.sub(r"\\\1", value)


def vector_to_bytes(vector: list[float]) -> bytes:
    """
    Convert a list of high-level Python float values into a raw binary byte buffer.

    :param vector: High-level list of floating-point numbers from the embedding model.

    :return: 32-bit float binary bytes suitable for RediSearch vector query parameters.
    """

    # RediSearch vector indices expect 32-bit floating-point values in raw C IEEE 754 byte order (FLOAT32).
    # Using Python's standard library `struct.pack` with the format string "{len(vector)}f" serializes the
    #  1,536 float values directly into a 6,144-byte binary payload without requiring third-party libraries
    #  like NumPy (used by the `redisvl` library).

    return struct.pack(f"{len(vector)}f", *vector)


def make_semantic_key(bucket: str, description: str) -> str:
    """
    Generate a deterministic SHA-256 Redis Hash key for an entry in the semantic cache.

    :param bucket: Deterministic category tag grouping request parameters.
    :param description: Raw natural language project description text.
    :return: Prefixed hexadecimal Redis key string.
    """

    # Deriving the key from the bucket and description guarantees idempotency: storing the exact
    # same project description within a bucket updates the existing document rather than accumulating
    # duplicate vector representations in the RediSearch index.

    raw_payload = f"{bucket}:{description}".encode("utf-8")
    digest = hashlib.sha256(raw_payload).hexdigest()
    return f"{SEMANTIC_KEY_PREFIX}:{digest}"


class EstimationSemanticCache:
    """
    Tier 2 semantic vector cache built on native `redis.asyncio` RediSearch.

    Recognizes semantically equivalent project descriptions using cosine vector similarity,
    partitioned by a deterministic bucket tag to isolate differing project parameters.
    """

    def __init__(
            self,
            redis_client: redis.Redis,
            openai_client: AsyncOpenAI | None = None,
            threshold: float | None = None,
            ttl: int | None = None,
            log_only: bool | None = None,
            index_name: str = DEFAULT_INDEX_NAME,
    ) -> None:
        """
        Initialize the semantic vector cache with Redis connection, OpenAI client, and thresholds.

        :param redis_client: Configured `redis.asyncio` client instance.
        :param openai_client: Optional `AsyncOpenAI` client for computing text embeddings.
        :param threshold: Minimum cosine similarity threshold (0.0 to 1.0) to consider a cache hit.
        :param ttl: Default time-to-live in seconds for stored cache keys.
        :param log_only: When True, performs lookup and logs similarity scores without returning cached hits.
        :param index_name: Name of the RediSearch vector index in Redis.
        """
        settings = get_settings()
        self.redis: redis.Redis = redis_client
        self.openai_client: AsyncOpenAI = openai_client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value() if settings.openai_api_key else "none"
        )
        self.threshold: float = threshold if threshold is not None else settings.semantic_cache_threshold
        self.ttl: int = ttl if ttl is not None else settings.semantic_cache_ttl
        self.log_only: bool = log_only if log_only is not None else settings.semantic_cache_log_only
        self.index_name: str = index_name

    async def initialize_index(self) -> None:
        """
        Initialize the RediSearch schema and index in Redis asynchronously if not already present.

        Schema Architecture:
        - `TagField("bucket")`: Partitions the search space by exact category string
          (`prompt_version:project_type:detail_level:output_format`). It prevents linguistic
          stemming or tokenization, ensuring exact literal equality filtering.
        - `TextField("result_json")`: Stores the serialized JSON representation of the validated
          `EstimationResult` for immediate retrieval upon a cache hit.
        - `VectorField("embedding", "FLAT", ...)`: Stores the 1,536-dimensional FLOAT32 vector representation.
          Uses FLAT indexing with COSINE distance metric for exact nearest-neighbor search.

        Execution is idempotent: if the index already exists, `ResponseError` is caught and suppressed.
        """
        schema = (
            TagField("bucket"),
            TextField("result_json"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDING_DIMENSIONS,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )

        # Index definition specifies the storage structure (HASH) and key prefix to monitor.
        definition = IndexDefinition(
            prefix=[f"{SEMANTIC_KEY_PREFIX}:"],
            index_type=IndexType.HASH,
        )

        try:
            # ft: stands for "Full Text"
            # ft() returns the RediSearch client bound to the specified index name.
            # Bounded with asyncio.timeout to fail open quickly if Redis is unreachable.
            async with asyncio.timeout(DEFAULT_SOCKET_TIMEOUT):
                await self.redis.ft(self.index_name).create_index(fields=schema, definition=definition)
            log.info("semantic_index_created", index_name=self.index_name)

        except ResponseError as e:
            # RediSearch raises ResponseError when an index with this name already exists.
            if "already exists" in str(e).lower():
                log.debug("semantic_index_exists", index_name=self.index_name)
            else:
                log.warning("semantic_index_create_failed", index_name=self.index_name, error=repr(e))

        except (RedisError, TimeoutError) as e:
            log.warning("semantic_index_create_failed", index_name=self.index_name, error=repr(e))

    @staticmethod
    def bucket_for(request: EstimationRequest, prompt_version: str) -> str:
        """
        Construct the deterministic bucket key from request options and prompt version.

        Preferred underscore-separated format to avoid RediSearch tag parsing issues with special characters.
        Inside a RediSearch tag filter ({...}), punctuation such as `:` `.` `-` and spaces must be backslash-escaped.

        :param request: Validated user estimation request.
        :param prompt_version: Active prompt template version (e.g. 'v1').

        :return: Underscore-separated bucket identifier string.
        """
        return f"{prompt_version}_{request.project_type.value}_{request.detail_level.value}_{request.output_format.value}"

    async def get_embedding(self, text: str) -> list[float] | None:
        """
        Generate a 1,536-dimensional vector embedding for the given text using OpenAI.

        :param text: Natural language text to embed.
        :return: List of 1,536 floating-point values, or None if embedding generation fails.
        """
        try:
            response = await self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            log.warning("semantic_embedding_failed", error=repr(e))
            return None

    async def lookup(
            self,
            request: EstimationRequest,
            prompt_version: str,
    ) -> EstimationResult | None:
        """
        Search the semantic vector cache for a previously cached response matching the request intent.

        Steps:
        1. Form the deterministic bucket tag (`@bucket:{...}`).
        2. Compute the 1,536-dim vector embedding of `request.description`.
        3. Issue a RediSearch KNN vector query:
           `(@bucket:{<bucket>})=>[KNN 1 @embedding $vec AS vector_distance]`
           - `@bucket`: Filters to documents possessing the identical bucket tag.
           - `=>[KNN 1 ...]`: Finds the 1-nearest neighbor to vector `$vec`.
           - `AS vector_distance`: Aliases the cosine distance metric (0 = identical, 2 = opposite).
           - `.dialect(2)`: Activates RediSearch vector syntax Dialect 2.
        4. Calculate cosine similarity = `1.0 - vector_distance`.
        5. Verify `similarity >= self.threshold`. If below, return None.
        6. If `self.log_only` is active, record telemetry and return None to allow LLM execution.

        :param request: User estimation request containing natural language description.
        :param prompt_version: Active prompt version for bucket partitioning.
        :return: Deserialized `EstimationResult` on hit, or None on miss or error.
        """
        bucket = self.bucket_for(request, prompt_version)
        embedding = await self.get_embedding(request.description)
        if embedding is None:
            return None

        # Build RediSearch KNN vector query targeting the bucket partition.
        query_str = f"(@bucket:{{{escape_tag(bucket)}}})=>[KNN 1 @embedding $vec AS vector_distance]"
        query = Query(query_str).sort_by("vector_distance").return_fields("result_json", "vector_distance").dialect(2)

        try:
            search_results = await self.redis.ft(self.index_name).search(
                query,
                query_params={"vec": vector_to_bytes(embedding)},
            )

        except (RedisError, ResponseError) as e:
            log.warning("semantic_cache_lookup_failed", bucket=bucket, error=repr(e))
            return None

        if not search_results or not search_results.docs:
            log.info("semantic_cache_entry_missed", bucket=bucket, reason="empty_index")
            return None

        top_hit = search_results.docs[0]
        # RediSearch COSINE distance ranges from 0.0 (identical) to 2.0 (opposite).
        distance = float(getattr(top_hit, "vector_distance", 1.0))
        similarity = 1.0 - distance

        log.info(
            "semantic_cache_lookup_evaluated",
            bucket=bucket,
            similarity=round(similarity, 4),
            threshold=self.threshold,
        )

        if similarity < self.threshold:
            log.info(
                "semantic_cache_entry_missed",
                bucket=bucket,
                reason="below_threshold",
                similarity=round(similarity, 4),
            )
            return None

        if self.log_only:
            log.info(
                "semantic_cache_hit_log_only",
                bucket=bucket,
                similarity=round(similarity, 4),
                threshold=self.threshold,
            )
            return None

        log.info(
            "semantic_cache_entry_retrieved",
            bucket=bucket,
            similarity=round(similarity, 4),
        )
        return EstimationResult.model_validate_json(top_hit.result_json)

    async def store(
            self,
            request: EstimationRequest,
            result: EstimationResult,
            prompt_version: str,
    ) -> None:
        """
        Persist a validated estimation result, its bucket tag, and its vector embedding in Redis.

        :param request: Input estimation request.
        :param result: Validated estimation domain model.
        :param prompt_version: Prompt version used during generation.
        """
        bucket = self.bucket_for(request, prompt_version)
        embedding = await self.get_embedding(request.description)
        if embedding is None:
            return

        key = make_semantic_key(bucket, request.description)
        payload = {
            "bucket": bucket,
            "result_json": result.model_dump_json(),
            "embedding": vector_to_bytes(embedding),
        }

        try:
            # Store document as a Redis Hash matching the schema definitions.
            await self.redis.hset(key, mapping=payload)
            if self.ttl > 0:
                await self.redis.expire(key, self.ttl)

            log.info("semantic_cache_entry_stored", bucket=bucket, key_prefix=key[:32], ttl=self.ttl)

        except RedisError as e:
            log.warning("semantic_cache_store_failed", bucket=bucket, error=repr(e))

    async def close(self) -> None:
        """
        Close the underlying Redis client connection pool cleanly.
        """
        try:
            await self.redis.aclose()
            log.info("semantic_cache_client_closed")
        except RedisError as e:
            log.warning("semantic_cache_close_failed", error=repr(e))


@lru_cache(maxsize=1)
def get_semantic_cache() -> EstimationSemanticCache:
    """
    Retrieve or initialize the application-wide singleton EstimationSemanticCache instance.

    :return: Authenticated EstimationSemanticCache singleton.
    """
    settings = get_settings()
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        decode_responses=False,  # Binary responses required for raw float32 vector bytes
        socket_timeout=DEFAULT_SOCKET_TIMEOUT,
        socket_connect_timeout=DEFAULT_SOCKET_TIMEOUT,
    )
    return EstimationSemanticCache(
        redis_client=client,
        threshold=settings.semantic_cache_threshold,
        ttl=settings.semantic_cache_ttl,
        log_only=settings.semantic_cache_log_only,
    )
