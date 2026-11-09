# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, status
import structlog

# --- App modules ---
import aie_estimator.services.estimations_service as estimations_service
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services.providers import GenerationOptions

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
async def create_estimation(request: EstimationRequest) -> EstimationResponse:
    """
    Generate an estimation from the provided project request parameters.

    :param request: Validated estimation request containing project description and preferences.

    :return: Generated estimation text with prompt version and telemetry metadata.
    """
    opts = GenerationOptions(model=request.model, engine=request.engine)
    log.info("estimation_request_received", model=request.model, engine=request.engine)

    result = await estimations_service.generate_estimation(
        request,
        opts=opts,
    )

    return EstimationResponse(
        result=result["result"],
        cached=bool(result.get("cache_hit", False)),
        prompt_version=result.get("prompt_version", "v1"),
        model=result["model"],
        provider=result["provider"],
        cost_usd=result.get("cost_usd"),
        usage=result["usage"],
    )
