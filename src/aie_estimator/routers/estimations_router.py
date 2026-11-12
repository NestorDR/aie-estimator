# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends, status
import structlog

# --- App modules ---
from aie_estimator.dependencies import get_estimation_service
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services.estimations_service import EstimationService

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

router = APIRouter(
    prefix="/api/v1",
    tags=["Estimations"],
)


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate software project estimation",
    description="Analyzes a project description and generates an effort breakdown using prompt templates.",
)
async def create_estimation(
    request: EstimationRequest,
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """
    Generate an estimation from the provided project request parameters.

    :param request: Validated estimation request containing project description and preferences.
    :param service: Injected EstimationService orchestrator.

    :return: Generated estimation text with prompt version and telemetry metadata.
    """
    log.info("estimation_request_received", model=request.model, engine=request.engine)

    result = await service.generate_estimation(request)

    return EstimationResponse.from_service_result(result)
