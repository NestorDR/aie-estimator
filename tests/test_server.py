# tests/test_server.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.model_catalog import get_supported_models
from aie_estimator.server import Server, create_app
from aie_estimator.services.exceptions import LLMProviderError
from aie_estimator.settings import Settings


def test_server_read_root_endpoint(client: TestClient) -> None:
    """
    GIVEN a running application server
    WHEN a GET request is sent to /
    THEN HTTP 200 is returned with welcome metadata.

    :param client: Shared TestClient fixture.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "Welcome to the Estimator Services" in data["message"]


def test_server_llm_provider_error_handler(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a failing LLM provider invocation
    WHEN a client sends a POST request to /api/v1/estimate
    THEN HTTP 500 is returned with a consistent, sanitized error message.

    :param client: TestClient fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """

    async def mock_failing_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise LLMProviderError("Upstream provider connection refused.")

    monkeypatch.setattr(
        "aie_estimator.services.estimations_service.EstimationService.generate_estimation",
        mock_failing_generation,
    )

    response = client.post(
        "/api/v1/estimate",
        json={
            "description": "This description is long enough to satisfy schema validation constraints for testing.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error occurred while generating the estimation" in data["detail"].lower()


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (RuntimeError("Fatal memory segmentation or hardware error."), 500, "Unexpected server error"),
        (ConnectionError("Connection pool refused."), 503, "Database service unavailable"),
    ],
    ids=["unhandled_runtime_error", "connection_error"],
)
def test_server_unhandled_exception_handler(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    """
    GIVEN an unexpected runtime error or ConnectionError raised during request handling
    WHEN a client sends a request to an endpoint
    THEN a sanitized HTTP 500 or 503 response is returned.

    :param app: FastAPI application fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    :param error: Exception raised by the service layer.
    :param expected_status: Expected HTTP status code.
    :param expected_detail: Substring expected in the response detail.
    """

    async def mock_failing_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(
        "aie_estimator.services.estimations_service.EstimationService.generate_estimation",
        mock_failing_generation,
    )

    with TestClient(app, raise_server_exceptions=False) as no_raise_client:
        response = no_raise_client.post(
            "/api/v1/estimate",
            json={
                "description": "Valid length project description for runtime exception testing.",
                "project_type": "web_saas",
                "detail_level": "medium",
                "output_format": "phases_table",
            },
        )

    assert response.status_code == expected_status
    assert expected_detail in response.json()["detail"]


def test_server_openapi_includes_supported_models_enum() -> None:
    """
    GIVEN a configured Server instance with customized OpenAPI generation
    WHEN create_app().openapi() is called
    THEN the EstimationRequest model schema contains an enum containing all supported models and None.
    """
    openapi_doc = create_app().openapi()
    model_prop = openapi_doc["components"]["schemas"]["EstimationRequest"]["properties"]["model"]
    assert "enum" in model_prop
    enum_values = model_prop["enum"]
    assert None in enum_values
    for model in get_supported_models():
        assert model in enum_values


def test_create_app_factory_returns_configured_app() -> None:
    """
    GIVEN the create_app factory
    WHEN invoked without parameters
    THEN a valid, configured FastAPI application is returned.
    """
    new_app = create_app()
    assert isinstance(new_app, FastAPI)
    assert new_app.title == "API - Estimator"
    assert "/api/v1/estimate" in new_app.openapi()["paths"]


def test_create_app_factory_with_custom_settings() -> None:
    """
    GIVEN custom application settings
    WHEN create_app is invoked with the custom settings
    THEN the created application reflects the provided settings.
    """
    custom_settings = Settings(openai_api_key="custom-key", server_port=9999)
    custom_app = create_app(custom_settings)
    assert isinstance(custom_app, FastAPI)


def test_server_init_with_custom_settings() -> None:
    """
    GIVEN custom application settings
    WHEN Server is instantiated with those settings
    THEN server.settings stores the custom instance directly.
    """
    custom_settings = Settings(openai_api_key="custom-key", server_port=8888)
    custom_server = Server(settings=custom_settings)
    assert custom_server.settings is custom_settings
    assert custom_server.settings.server_port == 8888
