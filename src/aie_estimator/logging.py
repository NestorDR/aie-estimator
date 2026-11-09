# src/aie_estimator/logging.py

# --- Python Standard Library ---
import logging
import logging.handlers
from pathlib import Path

# --- Third Party Libraries ---
import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

# --- App modules ---
from aie_estimator.settings import Environment, Settings
from aie_estimator.telemetry import inject_opentelemetry_context


def configure_logging(settings: Settings) -> None:
    """
    Configure unified structlog and standard library logging handlers.

    Sets up console and optional rotating file handlers with structlog formatting,
    and bridges standard library loggers (Uvicorn, FastAPI) to structlog.

    :param settings: Application settings instance controlling log level, format, and file logging.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    use_json = settings.json_logging or (settings.app_env == Environment.PROD)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        inject_opentelemetry_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        CallsiteParameterAdder(
            {
                CallsiteParameter.FILENAME,
                CallsiteParameter.LINENO,
                CallsiteParameter.FUNC_NAME,
            }
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        cache_logger_on_first_use=settings.app_env == Environment.PROD,
    )

    console_renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(colors=True)
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, console_renderer],
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    handlers: list[logging.Handler] = [console_handler]

    if settings.enable_file_logging:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / ("app.log" if settings.app_env == Environment.PROD else "dev.log")

        file_renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(colors=False)
        file_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, file_renderer],
        )
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress verbose noisy namespaces
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("psycopg").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(log_level)

