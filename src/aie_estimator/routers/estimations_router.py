# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
import logging
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, status
from pydantic import BaseModel, Field

# --- App modules ---
from aie_estimator.services import llm_service

logger: Final[logging.Logger] = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Estimations"],
)


class EstimateRequest(BaseModel):
    """Payload schema for software project estimation requests."""

    transcription: str = Field(
        ...,
        min_length=1,
        description="Meeting transcript text containing client requirements and project scope.",
        examples=[
            "In the meeting with the marketing team, the client explained that they need a landing page with a contact form..."],
    )


class EstimateResponse(BaseModel):
    """Response schema containing the generated estimation and provider metadata."""

    estimation: str = Field(..., description="Formatted software project effort estimation breakdown.")
    model: str = Field(..., description="Model identifier used to produce the estimation.")
    provider: str = Field(..., description="LLM provider name used to produce the estimation.")


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate software project estimation",
    description="Analyzes a meeting transcript and generates an effort breakdown using Context-Augmented Generation.",
)
async def create_estimation(payload: EstimateRequest) -> EstimateResponse:
    """
    Generate an estimation from the provided meeting transcription text.

    :param payload: Validated request containing the meeting transcript.

    :return: Generated estimation with model and provider metadata.
    """
    logger.info(f"Received estimation request with transcript length: {len(payload.transcription)} chars")
    result = llm_service.estimate(transcription=payload.transcription)

    return EstimateResponse(
        estimation=result["estimation"],
        model=result["model"],
        provider=result["provider"],
    )
