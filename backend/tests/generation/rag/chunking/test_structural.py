# tests/generation/rag/chunking/test_structural.py
"""
Offline unit tests for ``Chunker`` protocol compliance and ``JSONStructuralChunker``.

Verifies 1:1 component-to-chunk cardinality, deterministic chunk IDs,
contextual parent headers, exact BPE ``token_count`` values, filterable
metadata, omission of ``client_metadata.name``, and homogeneous
``budget_chunking_completed`` structured logging.
"""

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
import pytest
import structlog

# --- App modules ---
from aie_estimator.domain.schemas.embeddings import BudgetComponent, ClientMetadata, HistoricalBudget
from aie_estimator.generation.rag.chunking.protocol import Chunker
from aie_estimator.generation.rag.chunking.structural import JSONStructuralChunker
from aie_estimator.generation.rag.embedding.tokenizer import count_tokens

_SAMPLE_BUDGETS_PATH: Final[Path] = Path(__file__).resolve().parents[4] / "data" / "samples" / "budgets_sample.json"


def _make_budget(
    budget_id: str = "BUD-2024-001",
    client_name: str = "FintechCorp",
    component_ids: tuple[str, ...] = ("AUTH-001", "TXN-002"),
) -> HistoricalBudget:
    """Build a deterministic ``HistoricalBudget`` fixture with the given component IDs."""
    components = [
        BudgetComponent(
            component_id=component_id,
            name=f"Component {component_id}",
            description=f"Implementation details for {component_id}.",
            tech_stack=["python", "postgresql"],
            estimated_hours=80,
            complexity="high",
            dependencies=[],
        )
        for component_id in component_ids
    ]
    return HistoricalBudget(
        budget_id=budget_id,
        client_metadata=ClientMetadata(name=client_name, sector="finance", country="ES"),
        project_summary="Mobile banking API with OAuth 2.0",
        main_technology="python_fastapi",
        year=2024,
        total_estimated_hours=80 * len(components),
        components=components,
    )


@pytest.mark.parametrize(
    ("budgets", "expected_ids"),
    [
        ([], []),
        ([_make_budget("BUD-1", component_ids=("A-1",))], ["BUD-1::A-1"]),
        (
            [
                _make_budget("BUD-1", component_ids=("A-1", "A-2")),
                _make_budget("BUD-2", component_ids=("B-1",)),
            ],
            ["BUD-1::A-1", "BUD-1::A-2", "BUD-2::B-1"],
        ),
    ],
)
def test_structural_chunker_maps_components_one_to_one_in_order(
    budgets: list[HistoricalBudget],
    expected_ids: list[str],
) -> None:
    """
    GIVEN zero, one, or multiple HistoricalBudget records
    WHEN chunked via JSONStructuralChunker through the Chunker protocol
    THEN exactly one Chunk is emitted per component in input order with deterministic chunk_ids.
    """
    chunker: Chunker = JSONStructuralChunker()
    assert chunker.strategy_name == "structural"
    chunks = chunker.chunk(budgets)

    assert [chunk.chunk_id for chunk in chunks] == expected_ids
    # Repeatability check guarantees deterministic IDs and text across calls.
    assert chunks == chunker.chunk(budgets)


def test_structural_chunker_renders_header_metadata_and_exact_tokens() -> None:
    """
    GIVEN a HistoricalBudget with one component
    WHEN chunked by JSONStructuralChunker
    THEN text contains parent context header and component body, metadata has the 7 filter keys,
    and token_count matches count_tokens(text).
    """
    budget = _make_budget(budget_id="BUD-2024-001", client_name="SecretBank SA", component_ids=("AUTH-001",))

    [chunk] = JSONStructuralChunker().chunk([budget])

    assert chunk.chunk_id == "BUD-2024-001::AUTH-001"
    assert chunk.text.startswith(
        "[Project: Mobile banking API with OAuth 2.0]\n[Client sector: finance | Year: 2024 | Main tech: python_fastapi]\n\n"
    )
    assert "Component: Component AUTH-001" in chunk.text
    assert "Tech stack: python, postgresql" in chunk.text
    assert "Estimated hours: 80" in chunk.text
    assert chunk.token_count == count_tokens(chunk.text)
    assert chunk.metadata == {
        "budget_id": "BUD-2024-001",
        "component_id": "AUTH-001",
        "client_sector": "finance",
        "main_technology": "python_fastapi",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 80,
    }


def test_structural_chunker_omits_client_name_and_emits_homogeneous_log() -> None:
    """
    GIVEN a HistoricalBudget with a distinct client_metadata.name
    WHEN chunked with structlog log capture active
    THEN client_metadata.name is absent from chunk text, metadata, and logs,
    and budget_chunking_completed carries strategy, budgets, chunks, tokens, and latency_ms.
    """
    client_name = "ConfidentialClientCorp"
    budget = _make_budget(budget_id="BUD-2024-009", client_name=client_name, component_ids=("API-001", "DB-002"))

    with structlog.testing.capture_logs() as captured:
        chunks = JSONStructuralChunker().chunk([budget])

    for chunk in chunks:
        assert client_name not in chunk.text
        assert client_name not in json.dumps(chunk.metadata)

    completed_events = [event for event in captured if event.get("event") == "budget_chunking_completed"]
    assert len(completed_events) == 1
    log_record = completed_events[0]
    assert log_record["strategy"] == "structural"
    assert log_record["budgets"] == 1
    assert log_record["chunks"] == 2
    assert log_record["tokens"] == sum(chunk.token_count for chunk in chunks)
    assert isinstance(log_record["latency_ms"], float)
    assert client_name not in json.dumps(log_record)


def test_structural_chunker_processes_curated_sample_corpus() -> None:
    """
    GIVEN the curated 17-budget sample dataset under backend/data/samples/budgets_sample.json
    WHEN chunked with JSONStructuralChunker
    THEN all 60 components produce 60 unique chunks with positive token counts and no client names leaked.
    """
    raw_budgets = json.loads(_SAMPLE_BUDGETS_PATH.read_text(encoding="utf-8"))
    budgets = [HistoricalBudget.model_validate(item) for item in raw_budgets]

    chunks = JSONStructuralChunker().chunk(budgets)

    assert len(chunks) == 60
    assert len({chunk.chunk_id for chunk in chunks}) == 60
    assert all(chunk.token_count > 0 for chunk in chunks)
    client_names = {budget.client_metadata.name for budget in budgets}
    for chunk in chunks:
        assert not any(name in chunk.text for name in client_names)
        assert "name" not in chunk.metadata
