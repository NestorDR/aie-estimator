# src/aie_estimator/routers/health_router.py

# --- Python Standard Library ---
import logging
from typing import Annotated

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends

# --- App modules ---
from aie_estimator.settings import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Status"],
)


@router.get("/health")
async def get_status(
        settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str]:
    """
    Health check endpoint to verify API is running.

    :param settings: Injected application settings.
    
    :return: Dictionary indicating service health and version.
    """
    return {"status": "healthy", "version": settings.app_version}
