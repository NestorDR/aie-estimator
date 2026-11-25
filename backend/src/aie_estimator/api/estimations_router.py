# src/aie_estimator/api/estimations_router.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends, status
import structlog

# --- App modules ---
from aie_estimator.dependencies import get_caller_tier, get_estimation_service
from aie_estimator.domain.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.domain.estimations_service import EstimationService
from aie_estimator.generation.conversation.tier_resolver import Tier

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
    caller_tier: Tier | None = Depends(get_caller_tier),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """
    Generate an estimation from the provided project request parameters.

    :param request: Validated estimation request containing project description and preferences.
    :param caller_tier: Optional trusted stakeholder tier parsed from HTTP headers.
    :param service: Injected EstimationService orchestrator.

    :return: Generated estimation payload with tier, prompt version, and telemetry metadata.
    """
    log.info(
        "estimation_request_received",
        model=request.model,
        engine=request.engine,
        tier_override=request.tier.value if request.tier else None,
        caller_tier=caller_tier.value if caller_tier else None,
    )

    response = await service.generate_estimation(request, caller_tier=caller_tier)

    return EstimationResponse.model_validate(response)
