# tests/test_dependencies.py

# --- Python Standard Library ---
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.dependencies import (
    get_estimation_service,
    get_exact_cache,
    get_semantic_cache,
    get_session_store,
    get_settings,
)
from aie_estimator.services.cache import (
    EstimationExactCache,
    EstimationSemanticCache,
)
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.sessions.store import SessionStore
from aie_estimator.settings import Settings


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
