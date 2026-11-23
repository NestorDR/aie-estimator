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
from aie_estimator.guardrails.input import get_moderation_client
from aie_estimator.ingestion.catalog.loader import load_catalog
from aie_estimator.ingestion.catalog.models import DataCatalog
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.parsers.registry import ParserRegistry, default_registry
from aie_estimator.persistence.repositories.jobs import JobsRepository
from aie_estimator.persistence.repositories.mappings import MappingsRepository
from aie_estimator.services.boss_service import Boss
from aie_estimator.services.cache import (
    EstimationExactCache,
    EstimationSemanticCache,
    get_exact_cache as _get_exact_cache,
    get_semantic_cache as _get_semantic_cache,
)
from aie_estimator.services.critic_service import Critic
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.sessions.compression.policy import CompressionPolicy
from aie_estimator.sessions.store import SessionStore
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.settings import Settings, get_settings as _get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


def get_settings() -> Settings:
    """
    Retrieve the cached application-wide Settings singleton for FastAPI dependency wiring.

    :return: Application Settings singleton instance.
    """
    return _get_settings()


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
    except Exception as e:  # noqa: BLE001
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
