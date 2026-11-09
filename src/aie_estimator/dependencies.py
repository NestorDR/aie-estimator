# src/aie_estimator/dependencies.py

# --- Python Standard Library ---
from functools import lru_cache
from typing import Final

# --- Third Party Libraries ---
from fastapi import Depends
from openai import AsyncOpenAI
import structlog

# --- App modules ---
from aie_estimator.guardrails.input import get_moderation_client
from aie_estimator.services.cache import (
    EstimationExactCache,
    EstimationSemanticCache,
    get_exact_cache,
    get_semantic_cache as _get_semantic_cache,
)
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.settings import Settings, get_settings as _get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retrieve or initialize the cached application-wide Settings singleton.

    :return: Authenticated Settings singleton instance.
    """
    return _get_settings()


def get_semantic_cache(
    settings: Settings = Depends(get_settings),
) -> EstimationSemanticCache | None:
    """
    Resolve the Tier 2 vector similarity cache instance if enabled.

    :param settings: Injected application configuration settings.
    :return: Configured EstimationSemanticCache instance, or None if disabled or unconfigured.
    """
    if not settings.enable_semantic_cache:
        return None
    try:
        return _get_semantic_cache()
    except Exception as exc:  # noqa: BLE001
        log.warning("semantic_cache_initialization_failed", error=str(exc))
        return None


def get_estimation_service(
    exact_cache: EstimationExactCache = Depends(get_exact_cache),
    semantic_cache: EstimationSemanticCache | None = Depends(get_semantic_cache),
    moderation_client: AsyncOpenAI | None = Depends(get_moderation_client),
) -> EstimationService:
    """
    Compose the primary EstimationService instance with its caching and moderation collaborators.

    :param exact_cache: Resolved Tier 1 exact-match hash cache.
    :param semantic_cache: Optional Tier 2 vector similarity cache.
    :param moderation_client: Optional AsyncOpenAI client for content moderation.
    :return: Fully wired EstimationService instance.
    """
    return EstimationService(
        exact_cache=exact_cache,
        semantic_cache=semantic_cache,
        moderation_client=moderation_client,
    )
