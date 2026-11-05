# tests/test_server.py

# --- Python Standard Library ---
from typing import Any

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.services import LLMProviderError


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
    app: FastAPI,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a failing LLM provider invocation
    WHEN a client sends a POST request to /api/v1/estimate
    THEN HTTP 500 is returned with a consistent, sanitized error message.

    :param app: FastAPI application fixture.
    :param client: TestClient fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """

    async def mock_failing_generation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise LLMProviderError("Upstream provider connection refused.")

    monkeypatch.setattr("aie_estimator.services.estimations_service.generate_estimation", mock_failing_generation)

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


def test_server_database_connection_error_handler(
    app: FastAPI,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a ConnectionError raised during request handling
    WHEN a client sends a request to an endpoint
    THEN HTTP 503 is returned with a structured service unavailable response.

    :param app: FastAPI application fixture.
    :param client: TestClient fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """

    async def mock_connection_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ConnectionError("PostgreSQL pool refused connection.")

    monkeypatch.setattr("aie_estimator.services.estimations_service.generate_estimation", mock_connection_error)

    response = client.post(
        "/api/v1/estimate",
        json={
            "description": "Valid length project description for database exception testing.",
            "project_type": "web_saas",
            "detail_level": "medium",
            "output_format": "phases_table",
        },
    )

    assert response.status_code == 503
    data = response.json()
    assert "Database service unavailable" in data["detail"]


def test_server_unhandled_exception_handler(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an unexpected runtime error raised during request handling
    WHEN a client sends a request to an endpoint
    THEN HTTP 500 is returned with sanitized unexpected server error details.

    :param app: FastAPI application fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """

    async def mock_unexpected_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Fatal memory segmentation or hardware error.")

    monkeypatch.setattr("aie_estimator.services.estimations_service.generate_estimation", mock_unexpected_error)

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

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Unexpected server error"
