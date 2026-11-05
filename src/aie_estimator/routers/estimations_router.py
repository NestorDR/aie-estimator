# src/aie_estimator/routers/estimations_router.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, status
import structlog

# --- App modules ---
from aie_estimator.prompts.loader import render_estimation_prompt, resolve_prompt_version
from aie_estimator.schemas.estimations import EstimationRequest, EstimationResponse
from aie_estimator.services import estimations_service

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
    prompt_version = resolve_prompt_version()
    log.info("create_estimation_requested", prompt_version=prompt_version, model=request.model)

    system_prompt, user_prompt = render_estimation_prompt(request, version=prompt_version)
    opts = estimations_service.GenerationOptions(model=request.model, engine=request.engine)

    result = await estimations_service.generate_estimation(system_prompt, user_prompt, opts)

    return EstimationResponse(
        text=result.get("text") or result.get("estimation", ""),
        prompt_version=prompt_version,
        model=result["model"],
        provider=result["provider"],
        cost_usd=result.get("cost_usd"),
        usage=result["usage"],
    )
