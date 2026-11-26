# src/aie_estimator/domain/schemas/embeddings.py
"""
Strict Pydantic v2 data contracts for the RAG embedding and chunking pipeline.

Models historical component-based budgets (``HistoricalBudget``), structural
chunks (``Chunk`` / ``EmbeddedChunk``), and HTTP ingest request/response
envelopes. All contracts enforce ``extra="forbid"`` and closed domain literals
so schema drift fails loudly at the API and chunking boundaries rather than
leaking malformed keys into vector payloads or metadata filters.
"""

# --- Python Standard Library ---
from typing import Any, Final, Literal, Self

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Closed literal domains ensure typos fail loudly (HTTP 422) instead of polluting metadata filters.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]

# Enforce strict config for all models
# str_strip_whitespace tells Pydantic, when validating str-type fields, to remove
# blank spaces at the beginning and end of each value before saving it in the model
_STRICT_CONFIG: Final[ConfigDict] = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClientMetadata(BaseModel):
    """Client ownership metadata attached to a historical budget."""

    model_config = _STRICT_CONFIG

    name: str = Field(min_length=1, description="Client company name (excluded from embedded text and chunk metadata).")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(min_length=1, description="ISO-like country code, e.g. 'ES'.")


class BudgetComponent(BaseModel):
    """Single technical work component within a historical budget."""

    model_config = _STRICT_CONFIG

    component_id: str = Field(min_length=1, description="Stable component identifier within the budget, e.g. 'AUTH-001'.")
    name: str = Field(min_length=1, description="Short human-readable component name.")
    description: str = Field(min_length=1, description="Detailed description of the component work.")
    tech_stack: list[str] = Field(default_factory=list, description="Technologies used in this component.")
    estimated_hours: int = Field(ge=0, description="Estimated effort in hours for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Component identifiers this component depends on.",
    )


class HistoricalBudget(BaseModel):
    """Normalized component-based historical budget consumed by the RAG pipeline."""

    model_config = _STRICT_CONFIG

    budget_id: str = Field(min_length=1, description="Stable budget identifier, e.g. 'BUD-2024-001'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(min_length=1, description="One-line summary of the project.")
    main_technology: str = Field(min_length=1, description="Primary technology stack of the project.")
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    # Recorded total is not cross-checked against sum(components.estimated_hours) because historical
    # commercial budgets can carry rounded or negotiated totals.
    total_estimated_hours: int = Field(ge=0, description="Recorded sum of component estimated hours.")
    components: list[BudgetComponent] = Field(min_length=1, description="Historical budget components.")

    @model_validator(mode="after")
    def _validate_unique_component_ids(self) -> Self:
        """
        Ensure component identifiers are unique within a budget so chunk IDs never collide.

        :return: Validated ``HistoricalBudget`` instance.
        :raises ValueError: When duplicate ``component_id`` values are present.
        """
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate component_id in budget {self.budget_id}.")
        return self


class Chunk(BaseModel):
    """Embeddable text fragment with filterable non-embedded metadata and token count."""

    model_config = _STRICT_CONFIG

    chunk_id: str = Field(min_length=1, description="Deterministic identifier formatted as '{budget_id}::{component_id}'.")
    text: str = Field(min_length=1, description="Contextualized text sent to the embeddings API.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Filterable metadata that travels alongside the chunk without being embedded.",
    )
    token_count: int = Field(ge=0, description="Exact BPE token count of text.")


class EmbeddedChunk(Chunk):
    """Chunk enriched with its dense embedding vector."""

    embedding: list[float] = Field(
        min_length=1,
        description="Dense embedding vector produced by the configured embedding model.",
    )


class EmbeddingIngestRequest(BaseModel):
    """Request payload for embedding ingestion."""

    model_config = _STRICT_CONFIG

    budgets: list[HistoricalBudget] = Field(min_length=1, description="Historical budgets to chunk and embed.")


class EmbeddingIngestStats(BaseModel):
    """Aggregate token, count, and USD cost metrics for an embedding ingest call."""

    model_config = _STRICT_CONFIG

    total_budgets: int = Field(ge=0, description="Number of input budgets processed.")
    total_chunks: int = Field(ge=0, description="Number of chunks generated and embedded.")
    total_tokens: int = Field(ge=0, description="Total input tokens billed by the embeddings API.")
    estimated_cost_usd: float = Field(ge=0.0, description="Estimated USD cost for the embedding call.")


class EmbeddingIngestResponse(BaseModel):
    """Response envelope containing vectorized chunks and aggregate ingest statistics."""

    model_config = _STRICT_CONFIG

    chunks: list[EmbeddedChunk]
    stats: EmbeddingIngestStats
