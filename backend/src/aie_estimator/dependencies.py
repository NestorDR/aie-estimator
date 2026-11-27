# src/aie_estimator/dependencies.py

# --- Python Standard Library ---
from functools import lru_cache
from typing import Final

# --- Third Party Libraries ---
from fastapi import Depends, Header, HTTPException, Request, status
from openai import AsyncOpenAI
from psycopg_pool import AsyncConnectionPool
import structlog

# --- App modules ---
from aie_estimator.foundation.guardrails.input import get_moderation_client
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.catalog.models import DataCatalog
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.parsers.registry import ParserRegistry, default_registry
from aie_estimator.ingestion.pii.analyzer import PiiAnalyzer, PresidioPiiAnalyzer
from aie_estimator.ingestion.pii.mapping_store import PostgresMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer, resolve_pii_salt
from aie_estimator.foundation.persistence.repositories.jobs import JobsRepository
from aie_estimator.foundation.persistence.repositories.mappings import MappingsRepository
from aie_estimator.foundation.persistence.repositories.quarantine import QuarantinedRecordsRepository
from aie_estimator.generation.agentic.boss_service import Boss
from aie_estimator.generation.cag import (
    EstimationExactCache,
    EstimationSemanticCache,
    get_exact_cache as _get_exact_cache,
    get_semantic_cache as _get_semantic_cache,
)
from aie_estimator.generation.agentic.critic_service import Critic
from aie_estimator.domain.estimations_service import EstimationService
from aie_estimator.generation.conversation.compression.policy import CompressionPolicy
from aie_estimator.generation.conversation.store import SessionStore
from aie_estimator.generation.conversation.tier_resolver import Tier
from aie_estimator.generation.rag.chunking.protocol import Chunker
from aie_estimator.generation.rag.chunking.structural import JSONStructuralChunker
from aie_estimator.generation.rag.embedding.embedder import OpenAIEmbedder
from aie_estimator.settings import Settings, get_settings as _get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


def get_settings() -> Settings:
    """
    Retrieve the cached application-wide Settings singleton for FastAPI dependency wiring.

    :return: Application Settings singleton instance.
    """
    return _get_settings()


@lru_cache(maxsize=1)
def get_openai_client() -> AsyncOpenAI | None:
    """
    Return the cached ``AsyncOpenAI`` client for RAG embeddings when ``openai_api_key`` is configured.

    Disables SDK-level retries (``max_retries=0``) so ``OpenAIEmbedder`` owns retry timing and logging.

    :return: Cached ``AsyncOpenAI`` client, or ``None`` if the API key is unset or empty.
    """
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value().strip() if settings.openai_api_key else ""
    if not api_key:
        return None
    # AsyncOpenAI already retries 429/5xx with jitter + Retry-After (default max_retries=2).
    # Set max_retries=0 so SDK retries do not nest with OpenAIEmbedder's explicit (1s, 2s, 4s)
    # backoff loop (required for s07.0 parity and structured `embedding_batch_rate_limit_hit` logs).
    return AsyncOpenAI(api_key=api_key, max_retries=0)


def get_embedder(
    client: AsyncOpenAI | None = Depends(get_openai_client),
) -> OpenAIEmbedder | None:
    """
    Construct the async ``OpenAIEmbedder`` when an OpenAI client is available.

    :param client: Optional ``AsyncOpenAI`` client from ``get_openai_client``.

    :return: Configured ``OpenAIEmbedder``, or ``None`` when ``OPENAI_API_KEY`` is missing.
    """
    if client is None:
        return None
    return OpenAIEmbedder(client=client)


def get_chunker() -> Chunker:
    """
    Construct the budget chunking strategy used by the embedding ingest endpoint.

    Typed as the ``Chunker`` protocol so Session 7 strategies can replace it without router changes.

    :return: Stateless ``JSONStructuralChunker`` instance.
    """
    return JSONStructuralChunker()


def get_exact_cache(settings: Settings = Depends(get_settings)) -> EstimationExactCache | None:
    """
    Resolve the Tier 1 exact-match hash cache instance.

    :param settings: Injected application configuration settings.

    :return: Configured EstimationExactCache instance.
    """
    if not settings.enable_exact_cache:
        return None
    return _get_exact_cache()


def get_semantic_cache(settings: Settings = Depends(get_settings)) -> EstimationSemanticCache | None:
    """
    Resolve the Tier 2 vector similarity cache instance if enabled.

    :param settings: Injected application configuration settings.

    :return: Configured EstimationSemanticCache instance, or None if disabled or unconfigured.
    """
    if not settings.enable_semantic_cache:
        return None
    try:
        return _get_semantic_cache()
    except Exception as e:
        log.warning("semantic_cache_initialization_failed", error=str(e))
        return None


def get_compression_policy(settings: Settings = Depends(get_settings)) -> CompressionPolicy:
    """
    Construct the ``CompressionPolicy`` orchestrator configured from application settings.

    :param settings: Injected application configuration settings.

    :return: Configured ``CompressionPolicy`` instance.
    """
    return CompressionPolicy.from_settings(settings)


def get_estimation_service(
        exact_cache: EstimationExactCache | None = Depends(get_exact_cache),
        semantic_cache: EstimationSemanticCache | None = Depends(get_semantic_cache),
        moderation_client: AsyncOpenAI | None = Depends(get_moderation_client),
        compression_policy: CompressionPolicy = Depends(get_compression_policy),
) -> EstimationService:
    """
    Compose the primary EstimationService instance with its caching, moderation, and compression collaborators.

    :param exact_cache: Resolved Tier 1 exact-match hash cache.
    :param semantic_cache: Optional Tier 2 vector similarity cache.
    :param moderation_client: Optional AsyncOpenAI client for content moderation.
    :param compression_policy: Hybrid history compression policy for conversational turns.

    :return: Fully wired EstimationService instance.
    """
    return EstimationService(
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        moderation_client=moderation_client,
        compression_policy=compression_policy,
    )


def get_critic(settings: Settings = Depends(get_settings)) -> Critic:
    """
    Construct the stateless ``Critic`` service configured from application settings.

    :param settings: Injected application configuration settings.

    :return: Configured ``Critic`` instance.
    """
    return Critic(model=settings.acb_critic_model)


def get_boss(settings: Settings = Depends(get_settings)) -> Boss:
    """
    Construct the deterministic ``Boss`` orchestrator configured from application settings.

    :param settings: Injected application configuration settings.

    :return: Configured ``Boss`` instance.
    """
    return Boss(max_iterations=settings.acb_max_iterations)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    """
    Return the process-local ``SessionStore`` singleton configured from settings.

    :return: Cached ``SessionStore`` instance.
    """
    settings = get_settings()
    return SessionStore(
        max_turns=settings.session_max_turns,
        max_anchor_pairs=settings.session_max_anchor_pairs,
        ttl_seconds=settings.session_ttl_seconds,
    )


def get_caller_tier(
    x_user_tier: str | None = Header(default=None),
) -> Tier | None:
    """
    Extract optional trusted stakeholder tier from the ``X-User-Tier`` HTTP header.

    :param x_user_tier: Optional ``X-User-Tier`` header value (``developer``, ``pm``, ``executive``).

    :return: Parsed ``Tier`` enum or ``None`` if omitted.
    :raises HTTPException: HTTP 400 if ``X-User-Tier`` is provided with an unknown tier value.
    """
    if x_user_tier is None:
        return None

    normalized = x_user_tier.strip().lower()
    try:
        return Tier(normalized)
    except ValueError as e:
        log.warning("tier_header_invalid", raw_tier=x_user_tier[:64])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown tier '{x_user_tier}'. Expected one of: developer, pm, executive.",
        ) from e


def get_db_pool(request: Request) -> AsyncConnectionPool:
    """
    Resolve the lifespan-managed PostgreSQL connection pool from application state.

    :param request: Incoming FastAPI HTTP request.

    :return: Active AsyncConnectionPool instance.
    :raises ConnectionError: If PostgreSQL persistence is disabled or unconfigured.
    """
    db_pool: AsyncConnectionPool | None = getattr(request.app.state, "db_pool", None)
    if db_pool is None:
        raise ConnectionError("PostgreSQL connection pool is not initialized.")
    return db_pool


def get_jobs_repository(
    pool: AsyncConnectionPool = Depends(get_db_pool),
) -> JobsRepository:
    """
    Construct a JobsRepository bound to the active PostgreSQL connection pool.

    :param pool: Active AsyncConnectionPool resolved from application state.

    :return: Configured JobsRepository instance.
    """
    return JobsRepository(pool)


def get_mappings_repository(
    pool: AsyncConnectionPool = Depends(get_db_pool),
) -> MappingsRepository:
    """
    Construct a MappingsRepository bound to the active PostgreSQL connection pool.

    :param pool: Active AsyncConnectionPool resolved from application state.

    :return: Configured MappingsRepository instance.
    """
    return MappingsRepository(pool)


def get_quarantined_records_repository(
    pool: AsyncConnectionPool = Depends(get_db_pool),
) -> QuarantinedRecordsRepository:
    """
    Construct a QuarantinedRecordsRepository bound to the active PostgreSQL connection pool.

    :param pool: Active AsyncConnectionPool resolved from application state.

    :return: Configured QuarantinedRecordsRepository instance.
    """
    return QuarantinedRecordsRepository(pool)


def get_catalog(settings: Settings = Depends(get_settings)) -> DataCatalog:
    """
    Load and validate the data-source catalog from disk for each resolution.

    :param settings: Injected application configuration settings.

    :return: Validated DataCatalog snapshot.
    """
    return load_catalog(settings.catalog_path)


def get_filesystem_loader(settings: Settings = Depends(get_settings)) -> FileSystemLoader:
    """
    Construct a FileSystemLoader rooted at ``Settings.ingestion_data_root``.

    :param settings: Injected application configuration settings.

    :return: Configured FileSystemLoader instance.
    """
    return FileSystemLoader(settings.ingestion_data_root)


def get_parser_registry() -> ParserRegistry:
    """
    Return the default ParserRegistry configured with built-in format parsers.

    :return: Configured ParserRegistry instance.
    """
    return default_registry()


def get_pii_analyzer(settings: Settings = Depends(get_settings)) -> PiiAnalyzer:
    """
    Build the configured Presidio-backed ``PiiAnalyzer`` for offline ingestion.

    :param settings: Injected application configuration settings.

    :return: Configured PresidioPiiAnalyzer instance.
    """
    return PresidioPiiAnalyzer(
        spacy_model=settings.pii_spacy_model,
        language=settings.pii_language,
        score_threshold=settings.pii_confidence_threshold,
    )


def get_pii_pseudonymizer(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ConsistentPseudonymizer | None:
    """
    Build the async ``ConsistentPseudonymizer`` backed by PostgreSQL when PII pseudonymization is enabled.

    Checks ``pii_pseudonymization_enabled`` before constructing the analyzer or resolving the
    database pool so disabled deployments never eagerly load ``spaCy`` models.

    :param request: Incoming FastAPI HTTP request used to resolve the lifespan DB pool.
    :param settings: Injected application configuration settings.

    :return: Configured ConsistentPseudonymizer, or None when disabled in settings.
    """
    if not settings.pii_pseudonymization_enabled:
        return None
    analyzer = get_pii_analyzer(settings=settings)
    mappings_repo = get_mappings_repository(pool=get_db_pool(request))
    return ConsistentPseudonymizer(
        analyzer=analyzer,
        mapping_store=PostgresMappingStore(mappings_repo),
        salt=resolve_pii_salt(settings),
        faker_locale=settings.pii_faker_locale,
    )
