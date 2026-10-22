# tests/routers/test_health_router.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient

# --- App modules ---
from aie_estimator.settings import Settings, get_settings


def test_get_health_default_settings(client: TestClient) -> None:
    """
    Verify that the health check endpoint returns status healthy with default settings.

    :param client: Shared TestClient fixture.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_get_health_dependency_override(app: FastAPI, client: TestClient) -> None:
    """
    Verify that overriding get_settings on app.dependency_overrides updates endpoint output.

    :param app: FastAPI application fixture.
    :param client: Shared TestClient fixture.
    """
    overridden_version = "9.9.9-test"
    custom_settings = Settings()
    custom_settings.app_version = overridden_version

    app.dependency_overrides[get_settings] = lambda: custom_settings

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == overridden_version
