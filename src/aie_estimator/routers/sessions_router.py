# src/aie_estimator/routers/sessions_router.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends, status
import structlog

# --- App modules ---
from aie_estimator.dependencies import get_session_store
from aie_estimator.schemas.sessions import CreateSessionResponse, SessionInfoResponse
from aie_estimator.sessions.store import SessionStore

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

router = APIRouter(
    prefix="/api/v1/sessions",
    tags=["Sessions"],
)


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create conversational estimation session",
    description="Provision an empty in-memory conversational estimation session and return its UUID v4 identifier.",
)
async def create_session(
    store: SessionStore = Depends(get_session_store),
) -> CreateSessionResponse:
    """
    Create a new empty conversational estimation session.

    :param store: Injected ``SessionStore`` singleton.

    :return: Response payload containing the newly generated ``session_id``.
    """
    session = store.create()
    log.info("session_created", session_id=session.session_id)
    return CreateSessionResponse(session_id=session.session_id)


@router.get(
    "/{session_id}",
    response_model=SessionInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect conversational session state",
    description="Return current sliding-window message count, distilled project metadata, and timestamps for a session.",
)
async def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> SessionInfoResponse:
    """
    Inspect an active conversational session by identifier.

    :param session_id: Target session UUID string.
    :param store: Injected ``SessionStore`` singleton.

    :return: Inspection payload with window metrics, metadata, and UTC lifecycle timestamps.
    """
    session = store.get_or_404(session_id)
    return SessionInfoResponse(
        session_id=session.session_id,
        message_count=len(session.history.messages),
        max_turns=session.history.max_turns,
        metadata=session.metadata,
        created_at=session.created_at,
        last_accessed_at=session.last_accessed_at,
    )
