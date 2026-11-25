# src/aie_estimator/schemas/sessions.py

# --- Python Standard Library ---
from datetime import datetime

# --- Third Party Libraries ---
from pydantic import BaseModel, Field

# --- App modules ---
from aie_estimator.sessions.models import ProjectMetadata, TurnObservation
from aie_estimator.sessions.tier_resolver import Tier


class CreateSessionResponse(BaseModel):
    """Response payload returned when creating a new conversational estimation session."""

    session_id: str = Field(..., description="UUID v4 identifier for the new conversational session.")


class SessionInfoResponse(BaseModel):
    """Inspection payload exposing current session state, sliding window bounds, and timestamps."""

    session_id: str
    turn_index: int = 0
    message_count: int
    max_turns: int
    has_summary: bool = False
    summary_chars: int = 0
    summary: str | None = None
    anchors_count: int = 0
    anchors: list[str] = Field(default_factory=list)
    metadata: ProjectMetadata
    last_tier: Tier | None = None
    last_tier_rule: str | None = None
    last_turn_observed: TurnObservation | None = None
    created_at: datetime
    last_accessed_at: datetime

