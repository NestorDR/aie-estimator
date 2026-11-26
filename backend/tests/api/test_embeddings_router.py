# tests/api/test_embeddings_router.py

# --- Python Standard Library ---
from collections.abc import Generator
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import httpx
from openai import APIConnectionError, RateLimitError
import pytest

# --- App modules ---
from aie_estimator.dependencies import get_chunker, get_embedder, get_settings
from aie_estimator.foundation.llm.pricing import estimate_cost
from aie_estimator.generation.rag.chunking.structural import JSONStructuralChunker

_URL: Final[str] = "/api/v1/embeddings/ingest"
_REQUEST: Final[httpx.Request] = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
_PAYLOAD: Final[dict[str, Any]] = {
    "budgets": [
        {
            "budget_id": "BUD-2024-001",
            "client_metadata": {"name": "Acme Corp", "sector": "finance", "country": "ES"},
            "project_summary": "Online banking portal",
            "main_technology": "Python",
            "year": 2024,
            "total_estimated_hours": 80,
            "components": [
                {
                    "component_id": "AUTH-001",
                    "name": "Authentication",
                    "description": "OAuth 2.0 login",
                    "tech_stack": ["FastAPI"],
                    "estimated_hours": 40,
                    "complexity": "medium",
                    "dependencies": [],
                },
                {
                    "component_id": "PAY-001",
                    "name": "Payments",
                    "description": "Card payments",
                    "tech_stack": ["Stripe"],
                    "estimated_hours": 40,
                    "complexity": "high",
                    "dependencies": ["AUTH-001"],
                },
            ],
        }
    ]
}


@pytest.fixture(autouse=True)
def clean_dependency_overrides(client: TestClient) -> Generator[None, None, None]:
    """
    Ensure client.app.dependency_overrides is cleaned before and after each test.

    :param client: Shared TestClient fixture.
    """
    client.app.dependency_overrides.clear()
    yield
    client.app.dependency_overrides.clear()


def _override_embedder(client: TestClient, embedder: object | None) -> None:
    """
    Install the structural chunker and the given embedder double as dependency overrides.

    :param client: Shared TestClient fixture.
    :param embedder: Embedder double, or ``None`` to simulate a missing API key.
    """
    client.app.dependency_overrides[get_chunker] = JSONStructuralChunker
    client.app.dependency_overrides[get_embedder] = lambda: embedder


def test_ingest_returns_embedded_chunks_and_stats(client: TestClient) -> None:
    """
    GIVEN a valid budget with two components and a mocked embedder
    WHEN POST /api/v1/embeddings/ingest is called
    THEN HTTP 200 returns one embedded chunk per component in order and cost computed from token totals.

    :param client: Shared TestClient fixture.
    """
    embedder = MagicMock()
    embedder.embed_many = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    _override_embedder(client, embedder)

    response = client.post(_URL, json=_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert [chunk["chunk_id"] for chunk in body["chunks"]] == ["BUD-2024-001::AUTH-001", "BUD-2024-001::PAY-001"]
    assert [chunk["embedding"] for chunk in body["chunks"]] == [[0.1, 0.2], [0.3, 0.4]]
    assert "Acme" not in response.text
    stats = body["stats"]
    assert (stats["total_budgets"], stats["total_chunks"]) == (1, 2)
    assert stats["total_tokens"] == sum(chunk["token_count"] for chunk in body["chunks"])
    assert stats["estimated_cost_usd"] == estimate_cost(get_settings().embedding_model, stats["total_tokens"], 0)


def test_ingest_unpriced_model_reports_zero_cost(client: TestClient) -> None:
    """
    GIVEN an embedding model absent from the pricing catalog
    WHEN POST /api/v1/embeddings/ingest is called
    THEN estimated_cost_usd is 0.0.

    :param client: Shared TestClient fixture.
    """
    embedder = MagicMock()
    embedder.embed_many = AsyncMock(return_value=[[0.1], [0.2]])
    _override_embedder(client, embedder)
    settings = MagicMock(embedding_model="unpriced-embedding-model")
    client.app.dependency_overrides[get_settings] = lambda: settings

    response = client.post(_URL, json=_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["stats"]["estimated_cost_usd"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [{"budgets": []}, {"budgets": _PAYLOAD["budgets"], "unexpected": 1}],
    ids=["empty_budgets", "unknown_field"],
)
def test_ingest_invalid_payload_returns_422(client: TestClient, payload: dict[str, Any]) -> None:
    """
    GIVEN a payload with an empty budget list or an unknown field
    WHEN POST /api/v1/embeddings/ingest is called
    THEN HTTP 422 is returned and the embedder is never called.

    :param client: Shared TestClient fixture.
    :param payload: Invalid request body.
    """
    embedder = MagicMock()
    embedder.embed_many = AsyncMock()
    _override_embedder(client, embedder)

    response = client.post(_URL, json=payload)

    assert response.status_code == 422
    embedder.embed_many.assert_not_called()


@pytest.mark.parametrize(
    ("embedder_error", "expected_status"),
    [
        (RateLimitError("rate limited", response=httpx.Response(429, request=_REQUEST), body=None), 503),
        (APIConnectionError(request=_REQUEST), 502),
        (RuntimeError("secret upstream detail"), 500),
    ],
    ids=["rate_limit_503", "api_error_502", "unexpected_500"],
)
def test_ingest_embedder_failure_maps_to_sanitized_status(
    client: TestClient,
    embedder_error: Exception,
    expected_status: int,
) -> None:
    """
    GIVEN an embedder raising a rate-limit, API, or unexpected error
    WHEN POST /api/v1/embeddings/ingest is called
    THEN the mapped HTTP status is returned with a generic detail that hides the upstream message.

    :param client: Shared TestClient fixture.
    :param embedder_error: Exception raised by the mocked embedder.
    :param expected_status: Expected HTTP status code.
    """
    embedder = MagicMock()
    embedder.embed_many = AsyncMock(side_effect=embedder_error)
    _override_embedder(client, embedder)

    response = client.post(_URL, json=_PAYLOAD)

    assert response.status_code == expected_status
    assert "secret upstream detail" not in response.text


def test_ingest_missing_embedder_returns_503(client: TestClient) -> None:
    """
    GIVEN no embedder because OPENAI_API_KEY is unset
    WHEN POST /api/v1/embeddings/ingest is called
    THEN HTTP 503 is returned with a generic message.

    :param client: Shared TestClient fixture.
    """
    _override_embedder(client, None)

    response = client.post(_URL, json=_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service is not available."


def test_ingest_unversioned_path_is_not_exposed(client: TestClient) -> None:
    """
    GIVEN the application routes
    WHEN POST /embeddings/ingest is called without the /api/v1 prefix
    THEN HTTP 404 is returned.

    :param client: Shared TestClient fixture.
    """
    assert client.post("/embeddings/ingest", json=_PAYLOAD).status_code == 404
