# tests/test_dependencies.py

# --- Python Standard Library ---
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
from fastapi import HTTPException
from psycopg_pool import AsyncConnectionPool
import pytest

# --- App modules ---
from aie_estimator.dependencies import (
    get_caller_tier,
    get_catalog,
    get_db_pool,
    get_estimation_service,
    get_exact_cache,
    get_jobs_repository,
    get_mappings_repository,
    get_pii_analyzer,
    get_pii_pseudonymizer,
    get_semantic_cache,
    get_session_store,
    get_settings,
)
from aie_estimator.ingestion.pii.analyzer import PresidioPiiAnalyzer
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer
from aie_estimator.persistence.repositories.jobs import JobsRepository
from aie_estimator.persistence.repositories.mappings import MappingsRepository
from aie_estimator.services.cache import (
    EstimationExactCache,
    EstimationSemanticCache,
)
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.sessions.store import SessionStore
from aie_estimator.sessions.tier_resolver import Tier
from aie_estimator.settings import Environment, Settings


def test_get_settings_returns_singleton() -> None:
    """
    GIVEN multiple invocations of dependencies.get_settings
    WHEN resolved sequentially
    THEN the exact same cached Settings instance from settings.get_settings is returned.
    """
    assert get_settings() is get_settings()


def test_get_exact_cache_enabled() -> None:
    """
    GIVEN settings with exact cache enabled
    WHEN get_exact_cache is invoked
    THEN it returns the initialized exact cache instance.
    """
    mock_exact = MagicMock(spec=EstimationExactCache)
    settings = Settings(openai_api_key="test-key", enable_exact_cache=True)
    with patch("aie_estimator.dependencies._get_exact_cache", return_value=mock_exact) as mock_getter:
        result = get_exact_cache(settings)
        mock_getter.assert_called_once()
        assert result is mock_exact


def test_get_exact_cache_disabled() -> None:
    """
    GIVEN settings with exact cache disabled
    WHEN get_exact_cache is invoked
    THEN it returns None without calling the underlying factory.
    """
    settings = Settings(openai_api_key="test-key", enable_exact_cache=False)
    with patch("aie_estimator.dependencies._get_exact_cache") as mock_getter:
        result = get_exact_cache(settings)
        mock_getter.assert_not_called()
        assert result is None


def test_get_semantic_cache_enabled() -> None:
    """
    GIVEN settings with semantic cache enabled
    WHEN get_semantic_cache is invoked
    THEN it returns the initialized semantic cache instance.
    """
    mock_semantic = MagicMock(spec=EstimationSemanticCache)
    settings = Settings(openai_api_key="test-key", enable_semantic_cache=True)
    with patch("aie_estimator.dependencies._get_semantic_cache", return_value=mock_semantic) as mock_getter:
        result = get_semantic_cache(settings)
        mock_getter.assert_called_once()
        assert result is mock_semantic


def test_get_semantic_cache_disabled() -> None:
    """
    GIVEN settings with semantic cache disabled
    WHEN get_semantic_cache is invoked
    THEN it returns None without calling the underlying factory.
    """
    settings = Settings(openai_api_key="test-key", enable_semantic_cache=False)
    with patch("aie_estimator.dependencies._get_semantic_cache") as mock_getter:
        result = get_semantic_cache(settings)
        mock_getter.assert_not_called()
        assert result is None


def test_get_semantic_cache_error_handling() -> None:
    """
    GIVEN settings with semantic cache enabled but the factory raises an exception
    WHEN get_semantic_cache is invoked
    THEN it catches the exception and returns None gracefully.
    """
    settings = Settings(openai_api_key="test-key", enable_semantic_cache=True)
    with patch("aie_estimator.dependencies._get_semantic_cache", side_effect=RuntimeError("Redis connection failed")):
        result = get_semantic_cache(settings)
        assert result is None


def test_get_estimation_service_wires_collaborators() -> None:
    """
    GIVEN resolved cache and moderation collaborators
    WHEN get_estimation_service is invoked
    THEN it constructs an EstimationService instance with all collaborators assigned.
    """
    mock_exact = MagicMock(spec=EstimationExactCache)
    mock_semantic = MagicMock(spec=EstimationSemanticCache)
    mock_moderation = MagicMock()

    service = get_estimation_service(
        exact_cache=mock_exact,
        semantic_cache=mock_semantic,
        moderation_client=mock_moderation,
    )

    assert isinstance(service, EstimationService)
    assert service.exact_cache is mock_exact
    assert service.semantic_cache is mock_semantic
    assert service.moderation_client is mock_moderation


def test_get_session_store_returns_singleton_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN environment overrides for SESSION_MAX_TURNS and SESSION_TTL_SECONDS
    WHEN get_session_store is invoked multiple times
    THEN the exact same SessionStore singleton instance is returned with matching configuration.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("SESSION_MAX_TURNS", "8")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "1200")

    store_1 = get_session_store()
    store_2 = get_session_store()

    assert isinstance(store_1, SessionStore)
    assert store_1 is store_2
    assert store_1.max_turns == 8
    assert store_1.ttl_seconds == 1200


@pytest.mark.parametrize(
    ("raw_tier", "expected_tier"),
    [
        (None, None),
        ("developer", Tier.DEVELOPER),
        ("PM", Tier.PM),
        (" executive ", Tier.EXECUTIVE),
    ],
)
def test_get_caller_tier_valid(
    raw_tier: str | None,
    expected_tier: Tier | None,
) -> None:
    """
    GIVEN valid or omitted X-User-Tier headers
    WHEN get_caller_tier is called
    THEN it returns the normalized Tier or None.
    """
    tier = get_caller_tier(x_user_tier=raw_tier)

    assert tier is expected_tier


def test_get_caller_tier_rejects_invalid_tier() -> None:
    """
    GIVEN an unknown X-User-Tier header value
    WHEN get_caller_tier is called
    THEN it raises HTTPException with status 400.
    """
    with pytest.raises(HTTPException) as exc_info:
        get_caller_tier(x_user_tier="admin")

    assert exc_info.value.status_code == 400
    assert "Unknown tier" in str(exc_info.value.detail)


def test_get_db_pool_returns_pool_or_raises_when_uninitialized() -> None:
    """
    GIVEN a FastAPI Request with app.state.db_pool initialized or None
    WHEN get_db_pool is called
    THEN the active pool is returned when present and ConnectionError is raised when uninitialized.
    """
    uninitialized_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_pool=None)))
    with pytest.raises(ConnectionError, match=r"not initialized"):
        get_db_pool(uninitialized_request)  # type: ignore[arg-type]

    mock_pool = MagicMock(spec=AsyncConnectionPool)
    initialized_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_pool=mock_pool)))
    assert get_db_pool(initialized_request) is mock_pool  # type: ignore[arg-type]


def test_get_persistence_repositories_wire_pool() -> None:
    """
    GIVEN an active AsyncConnectionPool
    WHEN get_jobs_repository and get_mappings_repository are invoked
    THEN each returns its repository bound to the provided pool.
    """
    mock_pool = MagicMock(spec=AsyncConnectionPool)
    jobs_repo = get_jobs_repository(pool=mock_pool)
    mappings_repo = get_mappings_repository(pool=mock_pool)

    assert isinstance(jobs_repo, JobsRepository)
    assert isinstance(mappings_repo, MappingsRepository)


def test_get_catalog_reloads_on_each_call(tmp_path: Path) -> None:
    """
    GIVEN a catalog YAML file updated on disk between get_catalog calls
    WHEN get_catalog is called twice with the same Settings
    THEN the second call reflects the updated file without process restart.
    """
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text('version: "1.0.0"\nsources: []\n', encoding="utf-8")
    settings = Settings(app_env=Environment.TEST, catalog_path=catalog_file)

    first = get_catalog(settings=settings)
    assert first.version == "1.0.0"

    catalog_file.write_text('version: "1.0.1"\nsources: []\n', encoding="utf-8")
    second = get_catalog(settings=settings)
    assert second.version == "1.0.1"


def test_get_pii_dependencies_enabled_and_disabled() -> None:
    """
    GIVEN Settings with PII pseudonymization enabled or disabled
    WHEN get_pii_analyzer and get_pii_pseudonymizer are resolved
    THEN enabled returns ConsistentPseudonymizer and disabled returns None without touching db_pool.
    """
    enabled_settings = Settings(app_env=Environment.TEST, pii_pseudonymization_enabled=True)
    analyzer = get_pii_analyzer(settings=enabled_settings)
    assert isinstance(analyzer, PresidioPiiAnalyzer)

    mock_pool = MagicMock(spec=AsyncConnectionPool)
    initialized_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_pool=mock_pool)))
    pseudonymizer = get_pii_pseudonymizer(
        request=initialized_request,  # type: ignore[arg-type]
        settings=enabled_settings,
    )
    assert isinstance(pseudonymizer, ConsistentPseudonymizer)

    uninitialized_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_pool=None)))
    disabled_settings = Settings(app_env=Environment.TEST, pii_pseudonymization_enabled=False)
    assert get_pii_pseudonymizer(request=uninitialized_request, settings=disabled_settings) is None  # type: ignore[arg-type]
