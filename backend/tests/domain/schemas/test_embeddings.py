# tests/domain/schemas/test_embeddings.py
"""
Unit tests for strict Pydantic v2 RAG embedding pipeline schemas.

Verifies valid payload deserialization (including the curated 17-budget sample
corpus), ``extra="forbid"`` rejection, closed literal domain enforcement, and
non-empty / non-negative field bounds offline.
"""

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Any, Final

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.domain.schemas.embeddings import (
    BudgetComponent,
    Chunk,
    ClientMetadata,
    EmbeddedChunk,
    EmbeddingIngestRequest,
    EmbeddingIngestResponse,
    EmbeddingIngestStats,
    HistoricalBudget,
)

_SAMPLE_BUDGETS_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "samples" / "budgets_sample.json"
_VALID_COMPONENT: Final[dict[str, Any]] = {
    "component_id": "AUTH-001",
    "name": "OAuth 2.0 authentication backend",
    "description": "Authorization code flow with JWT sessions.",
    "tech_stack": ["python", "postgresql"],
    "estimated_hours": 120,
    "complexity": "high",
    "dependencies": [],
}


def _valid_budget_payload() -> dict[str, Any]:
    """Build a minimal valid ``HistoricalBudget`` dictionary for schema tests."""
    return {
        "budget_id": "BUD-2024-001",
        "client_metadata": {"name": "FintechCorp", "sector": "finance", "country": "ES"},
        "project_summary": "Mobile banking API with OAuth 2.0",
        "main_technology": "python_fastapi",
        "year": 2024,
        "total_estimated_hours": 120,
        "components": [_VALID_COMPONENT],
    }


def test_curated_sample_budgets_validate_cleanly() -> None:
    """
    GIVEN the curated 17-budget sample dataset under backend/data/samples/budgets_sample.json
    WHEN each record is validated into HistoricalBudget and wrapped in EmbeddingIngestRequest
    THEN all 17 budgets and all 60 components pass strict validation.
    """
    raw_budgets = json.loads(_SAMPLE_BUDGETS_PATH.read_text(encoding="utf-8"))
    request = EmbeddingIngestRequest(budgets=[HistoricalBudget.model_validate(item) for item in raw_budgets])

    assert len(request.budgets) == 17
    assert sum(len(budget.components) for budget in request.budgets) == 60


def test_embedding_ingest_response_round_trip() -> None:
    """
    GIVEN valid EmbeddedChunk and EmbeddingIngestStats models
    WHEN assembled into EmbeddingIngestResponse
    THEN all fields serialize and retain expected values.
    """
    chunk = EmbeddedChunk(
        chunk_id="BUD-2024-001::AUTH-001",
        text="[Project: API]\n\nComponent: Auth",
        metadata={"budget_id": "BUD-2024-001"},
        token_count=12,
        embedding=[0.1, 0.2, 0.3],
    )
    stats = EmbeddingIngestStats(
        total_budgets=1,
        total_chunks=1,
        total_tokens=12,
        estimated_cost_usd=0.00000024,
    )
    response = EmbeddingIngestResponse(chunks=[chunk], stats=stats)

    assert len(response.chunks) == 1
    assert response.chunks[0].embedding == [0.1, 0.2, 0.3]
    assert response.stats.total_tokens == 12


@pytest.mark.parametrize(
    ("model_cls", "payload", "error_pattern"),
    [
        (
            ClientMetadata,
            {"name": "Acme", "sector": "finance", "country": "ES", "unexpected": True},
            r"Extra inputs are not permitted",
        ),
        (
            ClientMetadata,
            {"name": "Acme", "sector": "aerospace", "country": "ES"},
            r"Input should be",
        ),
        (
            BudgetComponent,
            {
                "component_id": "AUTH-001",
                "name": "Auth",
                "description": "JWT auth",
                "estimated_hours": 40,
                "complexity": "extreme",
            },
            r"Input should be",
        ),
        (
            BudgetComponent,
            {
                "component_id": "AUTH-001",
                "name": "Auth",
                "description": "JWT auth",
                "estimated_hours": -5,
                "complexity": "low",
            },
            r"greater than or equal to 0",
        ),
        (
            HistoricalBudget,
            {**_valid_budget_payload(), "extra_field": "disallowed"},
            r"Extra inputs are not permitted",
        ),
        (
            HistoricalBudget,
            {**_valid_budget_payload(), "components": []},
            r"at least 1 item",
        ),
        (
            HistoricalBudget,
            {**_valid_budget_payload(), "year": 1999},
            r"greater than or equal to 2000",
        ),
        (
            HistoricalBudget,
            {**_valid_budget_payload(), "budget_id": "   "},
            r"at least 1 character",
        ),
        (
            HistoricalBudget,
            {**_valid_budget_payload(), "components": [_VALID_COMPONENT, _VALID_COMPONENT]},
            r"Duplicate component_id",
        ),
        (
            EmbeddingIngestRequest,
            {"budgets": []},
            r"at least 1 item",
        ),
        (
            Chunk,
            {"chunk_id": "B::C", "text": "hello", "token_count": -1},
            r"greater than or equal to 0",
        ),
        (
            EmbeddedChunk,
            {"chunk_id": "B::C", "text": "hello", "token_count": 1, "embedding": []},
            r"at least 1 item",
        ),
        (
            EmbeddingIngestStats,
            {"total_budgets": 1, "total_chunks": 1, "total_tokens": 10, "estimated_cost_usd": -0.01},
            r"greater than or equal to 0",
        ),
    ],
)
def test_schema_validation_rejects_invalid_payloads(
    model_cls: type[Any],
    payload: dict[str, Any],
    error_pattern: str,
) -> None:
    """
    GIVEN an invalid payload violating extra-field, literal-domain, min-length, or numeric bounds
    WHEN validated against its Pydantic schema
    THEN ValidationError is raised matching the expected constraint.
    """
    with pytest.raises(ValidationError, match=error_pattern):
        model_cls.model_validate(payload)
