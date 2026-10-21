# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
import logging
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, status

# --- App modules ---
from aie_estimator.models import EstimationRequest, EstimationResponse
from aie_estimator.services import llm_service

logger: Final[logging.Logger] = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Estimations"],
)


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate software project estimation",
    description="Analyzes a meeting transcript and generates an effort breakdown using Context-Augmented Generation.",
)
async def create_estimation(payload: EstimationRequest) -> EstimationResponse:
    """
    Generate an estimation from the provided meeting transcription text.

    :param payload: Validated request containing the meeting transcript.

    :return: Generated estimation with model and provider metadata.
    """
    logger.info(f"Received estimation request with transcript length: {len(payload.transcription)} chars")
    result = llm_service.estimate(transcription=payload.transcription)

    return EstimationResponse(
        estimation=result["estimation"],
        model=result["model"],
        provider=result["provider"],
    )
