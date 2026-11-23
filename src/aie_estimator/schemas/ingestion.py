# src/aie_estimator/schemas/ingestion.py

"""
Pydantic v2 schemas for the ingestion HTTP layer.

The API has two endpoints (POST + GET) and three response shapes. Keeping them
in one module makes the contract easy to read against the router code.
"""

# --- Python Standard Library ---
from datetime import datetime
from uuid import UUID

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field

# --- App modules ---
from aie_estimator.persistence.repositories.jobs import JobStatus


class IngestionRunRequest(BaseModel):
    """Request payload triggering an offline ingestion run for a catalog source."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)


class IngestionRunResponse(BaseModel):
    """Immediate HTTP 202 response payload returned when an ingestion job is queued."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    source_name: str
    status: JobStatus


class IngestionJobResponse(BaseModel):
    """Status inspection payload returned by ``GET /api/v1/ingestion/jobs/{job_id}``."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    job_id: UUID
    source_name: str
    status: JobStatus
    documents_count: int
    error_message: str | None
    started_at: datetime
    finished_at: datetime | None
