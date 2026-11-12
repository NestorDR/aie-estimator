# src/aie_estimator/routers/sessions_router.py

# --- Python Standard Library ---
from typing import Annotated, Final

# --- Third Party Libraries ---
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import AfterValidator, StringConstraints
import structlog

# --- App modules ---
from aie_estimator.attachments.extractor import validate_attachment_count
from aie_estimator.dependencies import get_estimation_service, get_session_store
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    ProviderMode,
    validate_catalog_model,
)
from aie_estimator.schemas.sessions import CreateSessionResponse, SessionInfoResponse
from aie_estimator.services.estimations_service import EstimationService
from aie_estimator.services.providers import GenerationOptions
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


@router.post(
    "/{session_id}/estimate",
    response_model=EstimationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate conversational session estimation",
    description=(
        "Execute a multi-turn estimation turn inside an active session using multipart/form-data fields "
        "and optional file attachments (.pdf, .docx, .md, .txt)."
    ),
)
async def estimate_in_session(
    session_id: str,
    transcript: Annotated[str, StringConstraints(strip_whitespace=True, min_length=20, max_length=2000), Form()],
    project_type: Annotated[ProjectType, Form()],
    detail_level: Annotated[DetailLevel, Form()],
    output_format: Annotated[OutputFormat, Form()],
    model: Annotated[str | None, Form(), AfterValidator(validate_catalog_model)] = None,
    engine: Annotated[ProviderMode, Form()] = "litellm",
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    store: SessionStore = Depends(get_session_store),
    service: EstimationService = Depends(get_estimation_service),
) -> EstimationResponse:
    """
    Execute a multi-turn software project estimation turn within an existing session.

    :param session_id: Target session UUID string.
    :param transcript: Current turn user transcript (20 to 2,000 characters after trimming).
    :param project_type: Target project category.
    :param detail_level: Granularity level for the estimation breakdown.
    :param output_format: Presentation format for the response text.
    :param model: Optional catalog model or profile alias override.
    :param engine: Provider dispatch mode (``"litellm"`` or ``"native"``).
    :param attachments: Optional list of uploaded specification files (up to 5 files).
    :param store: Injected ``SessionStore`` singleton.
    :param service: Injected ``EstimationService`` orchestrator.

    :return: Generated estimation text with prompt version and telemetry metadata.
    """
    session = store.get_or_404(session_id)

    named_uploads = [(upload.filename, upload) for upload in (attachments or []) if upload.filename]
    validate_attachment_count(len(named_uploads))

    raw_attachments: list[tuple[str, bytes]] = [(filename, await upload.read()) for filename, upload in named_uploads]

    log.info(
        "session_estimation_request_received",
        session_id=session_id,
        model=model,
        engine=engine,
        attachment_count=len(raw_attachments),
    )

    result = await service.generate_session_estimation(
        session=session,
        transcript=transcript,
        project_type=project_type,
        detail_level=detail_level,
        output_format=output_format,
        opts=GenerationOptions(model=model, engine=engine),
        attachments=raw_attachments,
    )

    return EstimationResponse.from_service_result(result)
