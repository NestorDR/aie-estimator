# tests/test_dependencies.py

# --- Python Standard Library ---
from unittest.mock import MagicMock, patch

# --- App modules ---
from aie_estimator.dependencies import (
    get_estimation_service,
    get_semantic_cache,
    get_settings,
)
from aie_estimator.services.cache import (
    EstimationExactCache,
    EstimationSemanticCache,
)
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.settings import Settings


def test_get_settings_cached() -> None:
    """
    GIVEN multiple invocations of get_settings
    WHEN resolved sequentially
    THEN the exact same cached instance is returned.
    """
    settings_1 = get_settings()
    settings_2 = get_settings()
    assert settings_1 is settings_2


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
