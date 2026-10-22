# tests/test_settings.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.settings import Environment, Settings, get_settings


def test_get_settings_returns_cached_singleton() -> None:
    """
    Verify that repeated calls to get_settings return the exact same instance.
    """
    first_instance = get_settings()
    second_instance = get_settings()

    assert isinstance(first_instance, Settings)
    assert first_instance is second_instance


def test_get_settings_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify that cache_clear enables reloading configuration with updated environment variables.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    try:
        monkeypatch.setenv("APP_ENV", "test")
        get_settings.cache_clear()

        reloaded_settings = get_settings()
        assert reloaded_settings.app_env == Environment.TEST
    finally:
        get_settings.cache_clear()
