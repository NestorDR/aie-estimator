# tests/routers/test_health_router.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
from fastapi.testclient import TestClient

# --- App modules ---
from aie_estimator.server import Server
from aie_estimator.settings import Settings, get_settings


def test_get_health_default_settings() -> None:
    """
    Verify that the health check endpoint returns status healthy with default settings.
    """
    server = Server()
    client = TestClient(server.app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_get_health_dependency_override() -> None:
    """
    Verify that overriding get_settings on app.dependency_overrides updates endpoint output.
    """
    server = Server()
    client = TestClient(server.app)
    
    overridden_version = "9.9.9-test"
    custom_settings = Settings()
    custom_settings.app_version = overridden_version

    server.app.dependency_overrides[get_settings] = lambda: custom_settings

    try:
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == overridden_version
    finally:
        server.app.dependency_overrides.clear()
