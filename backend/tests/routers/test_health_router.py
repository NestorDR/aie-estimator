# tests/routers/test_health_router.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.settings import Settings, get_settings


@pytest.mark.parametrize(
    ("override_version", "expected_version"),
    [
        (None, None),
        ("9.9.9-test", "9.9.9-test"),
    ],
    ids=["default_settings", "dependency_override"],
)
def test_get_health(
    app: FastAPI,
    client: TestClient,
    override_version: str | None,
    expected_version: str | None,
) -> None:
    """
    GIVEN an application instance and an optional settings dependency override
    WHEN a GET request is sent to the /health endpoint
    THEN a 200 OK status is returned with status healthy and the matching version.

    :param app: FastAPI application fixture.
    :param client: Shared TestClient fixture.
    :param override_version: Custom version string for dependency override, or None for default.
    :param expected_version: Expected version string in response body, or None to assert present.
    """
    if override_version is not None:
        custom_settings = Settings()
        custom_settings.app_version = override_version
        app.dependency_overrides[get_settings] = lambda: custom_settings

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    if expected_version is not None:
        assert data["version"] == expected_version
    else:
        assert "version" in data
