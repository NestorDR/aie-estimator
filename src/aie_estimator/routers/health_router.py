# src/aie_estimator/routers/health_router.py

# --- Python Standard Library ---
import logging

# --- Third Party Libraries ---
from fastapi import APIRouter

# --- App modules ---
from aie_estimator.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["Status"],
)


@router.get("/health")
async def get_status():
    """
    Health check endpoint to verify API is running.

    :return: Dictionary indicating service health and version.
    """
    return {"status": "healthy", "version": settings.app_version}
