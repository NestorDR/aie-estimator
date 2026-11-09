# tests/conftest.py

# --- Python Standard Library ---
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import MagicMock

# --- Third Party Libraries ---
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
import pytest

# --- App modules ---
from aie_estimator.guardrails.input import get_moderation_client
from aie_estimator.model_catalog import get_model_catalog
from aie_estimator.server import Server
from aie_estimator.services.cache import get_estimation_cache
from aie_estimator.services.providers.litellm_provider import get_litellm_router
from aie_estimator.services.providers.native_providers import get_anthropic_client, get_openai_client
from aie_estimator.settings import get_settings
from aie_estimator.utils.model_constants import DEFAULT_GPT_LLM


def _clear_all_caches() -> None:
    """Clear all application @lru_cache singletons between tests."""
    get_settings.cache_clear()
    get_openai_client.cache_clear()
    get_anthropic_client.cache_clear()
    get_litellm_router.cache_clear()
    get_model_catalog.cache_clear()
    get_estimation_cache.cache_clear()
    get_moderation_client.cache_clear()


@pytest.fixture(autouse=True)
def clean_settings_cache() -> Generator[None, None, None]:
    """
    Ensure the settings lru_cache, cached LLM clients, and estimation cache are cleared before and after each test.

    :return: None.
    """
    _clear_all_caches()
    yield
    _clear_all_caches()


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
def make_estimation_result() -> Callable[..., dict[str, Any]]:
    """
    Provide a factory fixture generating valid LLM estimation result dictionaries with overrides.

    :return: Factory function returning an estimation result dictionary.
    """

    def _factory(**overrides: Any) -> dict[str, Any]:
        result_payload = overrides.get("result") or {
            "summary": "Full-stack web application delivery breakdown.",
            "confidence_pct": 85,
            "phases": [
                {
                    "name": "Design & Architecture",
                    "duration_weeks": 2,
                    "cost_eur": 5000,
                    "summary": "Technical architecture specification and UX wireframes.",
                },
                {
                    "name": "Core Implementation",
                    "duration_weeks": 6,
                    "cost_eur": 15000,
                    "summary": "Frontend, backend services, database schema, and test automation.",
                },
            ],
            "total_duration_weeks": 8,
            "total_cost_eur": 20000,
        }
        body = overrides.get("estimation") or overrides.get("text") or "Project Breakdown:\n- Backend: 40h\nTotal: 40h"
        return {
            "result": result_payload,
            "estimation": body,
            "text": body,
            "model": DEFAULT_GPT_LLM,
            "provider": "openai",
            "finish_reason": "stop",
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            "preprocessing": "none",
            "extracted_requirements": None,
            "latency_ms": 250,
            "cost_usd": None,
            **overrides,
        }

    return _factory


@pytest.fixture
def mock_async_openai_client() -> MagicMock:
    """
    Provide a reusable MagicMock fixture typed to AsyncOpenAI for Instructor compatibility.

    :return: Mocked AsyncOpenAI client instance.
    """
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def mock_async_anthropic_client() -> MagicMock:
    """
    Provide a reusable MagicMock fixture typed to AsyncAnthropic for Instructor compatibility.

    :return: Mocked AsyncAnthropic client instance.
    """
    client = MagicMock(spec=AsyncAnthropic)
    client.messages = MagicMock()
    return client
