# src/aie_estimator/routers/ingestion_router.py

"""
POST /runs   &   GET /jobs/{job_id}

The router is intentionally thin: it validates the source against the catalog,
records the job row, dispatches the orchestrator as a BackgroundTask, and
returns 202 immediately. All ingestion logic lives in
``app.ingestion.orchestrator`` — the router never imports parsers.
"""

# --- Python Standard Library ---
import asyncio
from typing import Final
from uuid import UUID

# --- Third Party Libraries ---
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext
import structlog

# --- App modules ---
from aie_estimator.dependencies import (
    get_catalog,
    get_filesystem_loader,
    get_jobs_repository,
    get_parser_registry,
    get_quarantined_records_repository,
)
from aie_estimator.ingestion.catalog.models import DataCatalog
from aie_estimator.ingestion.loaders.filesystem import FileSystemLoader
from aie_estimator.ingestion.orchestrator import IngestionRejected, ingest_source, require_included_source
from aie_estimator.ingestion.parsers.registry import ParserRegistry
from aie_estimator.persistence.repositories.jobs import JobsRepository
from aie_estimator.persistence.repositories.quarantine import QuarantinedRecordsRepository
from aie_estimator.schemas.ingestion import (
    IngestionJobResponse,
    IngestionRunRequest,
    IngestionRunResponse,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)
_TRACER: Final[trace.Tracer] = trace.get_tracer(__name__)

router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["Ingestion"],
)


async def _run_ingestion_job(
    *,
    job_id: UUID,
    source_name: str,
    catalog: DataCatalog,
    loader: FileSystemLoader,
    registry: ParserRegistry,
    jobs_repo: JobsRepository,
    quarantine_repo: QuarantinedRecordsRepository,
    parent_span_context: SpanContext | None,
) -> None:
    """
    Execute an offline ingestion job in the background with an isolated OTel root span.

    :param job_id: Persisted ingestion job UUID.
    :param source_name: Catalog source identifier to ingest.
    :param catalog: Validated DataCatalog snapshot resolved for the run.
    :param loader: Filesystem loader resolving source files.
    :param registry: ParserRegistry providing format parsers.
    :param jobs_repo: Async repository tracking job lifecycle state.
    :param quarantine_repo: Async repository persisting quarantined records.
    :param parent_span_context: Optional triggering HTTP SpanContext for trace linking.
    """
    links = [Link(parent_span_context)] if parent_span_context is not None and parent_span_context.is_valid else []
    with _TRACER.start_as_current_span("ingestion.run", links=links) as span:
        span.set_attribute("ingestion.job_id", str(job_id))
        span.set_attribute("ingestion.source_name", source_name)

        await jobs_repo.mark_running(job_id)
        log.info("ingestion_job_started", job_id=str(job_id), source_name=source_name)

        try:
            result = await asyncio.to_thread(
                ingest_source,
                catalog=catalog,
                source_name=source_name,
                loader=loader,
                registry=registry,
            )
            if result.quarantined:
                await quarantine_repo.upsert_many(
                    job_id=job_id,
                    source_name=source_name,
                    records=result.quarantined,
                )
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            log.exception(
                "ingestion_job_failed",
                job_id=str(job_id),
                source_name=source_name,
                error=repr(exc),
                exc_info=exc,
            )
            await jobs_repo.mark_failed(job_id, error_message=str(exc))
            return

        documents_count = len(result.documents)
        quarantined_count = len(result.quarantined)
        span.set_attribute("ingestion.documents_count", documents_count)
        span.set_attribute("ingestion.quarantined_count", quarantined_count)
        await jobs_repo.mark_completed(
            job_id,
            documents_count=documents_count,
            validation_report=result.report,
        )
        log.info(
            "ingestion_job_completed",
            job_id=str(job_id),
            source_name=source_name,
            documents_count=documents_count,
            quarantined_count=quarantined_count,
        )


@router.post(
    "/runs",
    response_model=IngestionRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger offline ingestion run",
    description="Validate a catalog source, create an ingestion job, and queue background execution.",
)
async def create_ingestion_run(
    request: IngestionRunRequest,
    background_tasks: BackgroundTasks,
    catalog: DataCatalog = Depends(get_catalog),
    loader: FileSystemLoader = Depends(get_filesystem_loader),
    registry: ParserRegistry = Depends(get_parser_registry),
    jobs_repo: JobsRepository = Depends(get_jobs_repository),
    quarantine_repo: QuarantinedRecordsRepository = Depends(get_quarantined_records_repository),
) -> IngestionRunResponse:
    """
    Queue an offline ingestion run for an ``include`` catalog source.

    :param request: Validated run request naming the catalog source.
    :param background_tasks: FastAPI BackgroundTasks queue.
    :param catalog: Per-run validated DataCatalog snapshot.
    :param loader: Filesystem loader rooted at ``Settings.ingestion_data_root``.
    :param registry: Configured ParserRegistry instance.
    :param jobs_repo: Async JobsRepository bound to the lifespan connection pool.

    :return: Immediate 202 Accepted response containing the created ``job_id``.
    :raises HTTPException: HTTP 404 if the source name is absent from the catalog.
    :raises IngestionRejected: If the source decision is not ``include``.
    """
    try:
        require_included_source(catalog, request.source_name)
    except IngestionRejected as exc:
        if exc.reason == "unknown_source":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"reason": exc.reason, "source_name": request.source_name},
            ) from exc
        raise

    job = await jobs_repo.create(request.source_name)
    parent_span_context = trace.get_current_span().get_span_context()
    log.info("ingestion_run_requested", job_id=str(job.job_id), source_name=job.source_name)

    background_tasks.add_task(
        _run_ingestion_job,
        job_id=job.job_id,
        source_name=request.source_name,
        catalog=catalog,
        loader=loader,
        registry=registry,
        jobs_repo=jobs_repo,
        quarantine_repo=quarantine_repo,
        parent_span_context=parent_span_context,
    )
    return IngestionRunResponse(
        job_id=job.job_id,
        source_name=job.source_name,
        status=job.status,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect offline ingestion job",
    description="Return the persisted status and metrics of an offline ingestion job.",
)
async def get_ingestion_job(
    job_id: UUID,
    jobs_repo: JobsRepository = Depends(get_jobs_repository),
) -> IngestionJobResponse:
    """
    Retrieve the status and outcome of an offline ingestion job.

    :param job_id: Target ingestion job UUID.
    :param jobs_repo: Async JobsRepository bound to the lifespan connection pool.

    :return: Persisted job status and execution metrics.
    :raises HTTPException: HTTP 404 if no job exists with ``job_id``.
    """
    job = await jobs_repo.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found",
        )
    return IngestionJobResponse.model_validate(job)
