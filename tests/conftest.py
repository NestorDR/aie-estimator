# tests/conftest.py

# --- Python Standard Library ---
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock

# --- Third Party Libraries ---
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.server import Server
from aie_estimator.services.llm_service import _reset_clients
from aie_estimator.settings import Environment, Settings, get_settings


@pytest.fixture(autouse=True)
def clean_settings_cache() -> Generator[None, None, None]:
    """
    Ensure the settings lru_cache and cached LLM clients are cleared before and after each test.

    :return: None.
    """
    get_settings.cache_clear()
    _reset_clients()
    yield
    get_settings.cache_clear()
    _reset_clients()


@pytest.fixture
def test_settings() -> Settings:
    """
    Provide an isolated Settings instance configured for the test environment.

    :return: Test-configured Settings instance.
    """
    return Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-mock-key-12345"),
        llm_model="gpt-4o-mini",
    )


@pytest.fixture
def app() -> Generator[FastAPI, None, None]:
    """
    Instantiate the FastAPI application and clear dependency overrides on teardown.

    :return: Configured FastAPI application instance.
    """
    server = Server()
    yield server.app
    # Teardown: ensure dependency overrides do not leak across test cases
    server.app.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Provide a TestClient managing the application lifespan context.

    :param app: The FastAPI application fixture.
    :return: Active TestClient instance.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """
    Generate a mock response conforming to the OpenAI Responses API structure.

    :return: Configured MagicMock representing an OpenAI response object.
    """
    mock_resp = MagicMock()
    mock_resp.status = "completed"
    mock_resp.output_text = "Estimated project duration: 3 weeks. Total effort: 120 hours."
    mock_resp.model = "gpt-4o-mini"
    return mock_resp


@pytest.fixture
def sample_transcription() -> str:
    """
    Provide a representative meeting transcript for estimation testing.

    :return: Multi-line sample transcript string.
    """
    return (
        "Client: We need a customer onboarding portal with auth, document upload, and admin review.\n"
        "Team: We will implement standard OAuth2 and an S3-compatible document pipeline.\n"
        "Client: Launch target is within 4 weeks."
    )


@pytest.fixture
def sample_estimation_payload(sample_transcription: str) -> dict[str, Any]:
    """
    Provide a valid request body for POST /api/v1/estimate.

    :param sample_transcription: Sample transcript fixture.
    
    :return: Payload dictionary matching EstimationRequest schema.
    """
    return {"transcription": sample_transcription}