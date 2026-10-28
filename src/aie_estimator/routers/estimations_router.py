# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
import logging
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, status
import structlog

# --- App modules ---
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services import llm_service, evaluation

log: Final[logging.Logger] = structlog.get_logger()

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
async def create_estimation(request: EstimationRequest) -> EstimationResponse:
    """
    Generate an estimation from the provided meeting transcription text.

    :param request: Validated request containing the meeting transcript.

    :return: Generated estimation with model and provider metadata.
    """
    log.info("create_estimation_requested")
    opts = llm_service.GenerationOptions(
        preprocessing=request.preprocessing,
        example_format=request.example_format,
        num_examples=request.num_examples,
        use_examples=request.use_examples,
        model=request.model,
        max_tokens=request.max_tokens,
        thinking_budget=request.thinking_budget,
    )

    result = llm_service.generate_estimation(request.transcription, opts)

    validation = (
        evaluation.evaluate_estimation_structure(result["estimation"], result["finish_reason"])
        if request.evaluate
        else None
    )

    return EstimationResponse(**result, validation=validation)