# src/aie_estimator/models/estimations.py

# --- Third Party Libraries ---
from pydantic import BaseModel, Field


class EstimationRequest(BaseModel):
    """Payload schema for software project estimation requests."""

    transcription: str = Field(
        ...,
        min_length=1,
        description="Meeting transcript text containing client requirements and project scope.",
        examples=["In the meeting with the marketing team, the client explained that they need a landing page with a contact form..."],
    )


class EstimationResponse(BaseModel):
    """Response schema containing the generated estimation and provider metadata."""

    estimation: str = Field(..., description="Formatted software project effort estimation breakdown.")
    model: str = Field(..., description="Model identifier used to produce the estimation.")
    provider: str = Field(..., description="LLM provider name used to produce the estimation.")
