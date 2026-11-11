# src/aie_estimator/schemas/sessions.py

# --- Python Standard Library ---
from datetime import datetime

# --- Third Party Libraries ---
from pydantic import BaseModel, Field

# --- App modules ---
from aie_estimator.sessions.models import ProjectMetadata


class CreateSessionResponse(BaseModel):
    """Response payload returned when creating a new conversational estimation session."""

    session_id: str = Field(..., description="UUID v4 identifier for the new conversational session.")


class SessionInfoResponse(BaseModel):
    """Inspection payload exposing current session state, sliding window bounds, and timestamps."""

    session_id: str
    message_count: int
    max_turns: int
    metadata: ProjectMetadata
    created_at: datetime
    last_accessed_at: datetime

