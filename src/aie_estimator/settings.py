# src/aie_estimator/settings.py

# --- Python Standard Library ---
from enum import Enum
from functools import lru_cache
import importlib.metadata
import logging
import logging.handlers
from os import environ
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from structlog.processors import CallsiteParameter, CallsiteParameterAdder

# --- App modules ---
from aie_estimator.schemas.models import MODEL_CATALOG

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


class Environment(str, Enum):
    """
    Defines the possible environments for the application itself (e.g., dev, test, prod).
    """

    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables for a specific environment.
    """

    # Application environment and version
    app_env: Environment = Environment.DEV
    app_version: str = "0.0.0"  # It will be updated with the version from the installed package metadata

    # Logging settings
    log_level: str = "INFO"
    json_logging: bool = False
    enable_file_logging: bool = False

    # Server settings
    server_host: str = "localhost"
    server_port: int = 8000

    # AI settings
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"

    # Cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 86400

    @property
    def llm_provider(self) -> str:
        """Derive the default provider from llm_model via MODEL_CATALOG.

        :return: Provider identifier ('openai' or 'anthropic').
        """
        spec = MODEL_CATALOG.get(self.llm_model)
        return spec.provider if spec is not None else "openai"

    # Configure env file handling
    model_config = SettingsConfigDict(
        env_file=f".env.{app_env.value}",
        env_file_encoding="utf-8",
        # To ignore additional variables
        extra="ignore",
        # Variable name case transformation
        case_sensitive=False,
        # Allow mapping environment variable names
        populate_by_name=True,
    )

    def __init__(self, **kwargs):
        """
        Initialize settings with a dynamic .env file path based on the environment.

        The initialization process:
        1. Determines which environment to use (from kwargs or environment variables)
        2. Loads the corresponding .env file (e.g., .env.dev for DEV environment)
        3. Initializes configuration with values from environment variables and .env file

        Usage examples:
            # Explicit environment selection
            test_settings = Settings(app_env=Environment.DEV)
            # Override default port during initialization (e.g., in test fixtures)
            test_settings = Settings(app_env=Environment.TEST, server_port=8080)

        :param kwargs: Keyword arguments to override settings during instantiation.
        
        :raises ValueError: If an unrecognized environment identifier is provided.
        :raises ValidationError: If setting validation fails.
        """
        environment = (kwargs.get("app_env") or environ.get("APP_ENV", Environment.DEV.value)).lower()
        if isinstance(environment, str):
            environment = Environment(environment)  # Convert to Environment enum

        # Set the .env file path
        env_file = f".env.{environment.value}"
        script_dir = Path(__file__).resolve().parent
        env_path = script_dir / env_file

        if env_path.exists():
            kwargs["_env_file"] = str(env_path)  # Update kwargs with correct env_file
            log.info("loading_environment_file", env_path=str(env_path))
        else:
            log.warning("environment_file_not_found", env_path=str(env_path))

        try:
            # Initialize the base class with the identified environment
            super().__init__(**kwargs)

            # Configure structlog immediately after base settings initialization
            self.configure_logging()

            try:
                # Fetch the version from the installed package metadata
                installed_version = importlib.metadata.version("aie_estimator")
                if installed_version:
                    self.app_version = installed_version

            except importlib.metadata.PackageNotFoundError:
                pass

            log.info("settings_loaded", app_env=self.app_env.value, app_version=self.app_version)

        except ValidationError as e:
            log.exception("settings_initialization_failed", exc_info=e)
            raise

        except Exception as e:
            log.exception("unexpected_initialization_error", exc_info=e)
            raise Exception("Unexpected error during initialization.") from e

    def configure_logging(self) -> None:
        """
        Configure unified structlog and standard library logging handlers.

        Sets up console and optional rotating file handlers with structlog formatting,
        and bridges standard library loggers (Uvicorn, FastAPI) to structlog.
        """
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        use_json = self.json_logging or (self.app_env == Environment.PROD)

        shared_processors: list[structlog.types.Processor] = [
            structlog.contextvars.merge_contextvars,
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
            cache_logger_on_first_use=self.app_env == Environment.PROD,
        )

        console_renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer(colors=True)
        console_formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, console_renderer],
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        handlers: list[logging.Handler] = [console_handler]

        if self.enable_file_logging:
            log_dir = Path(__file__).resolve().parents[2] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / ("app.log" if self.app_env == Environment.PROD else "dev.log")

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
        logging.getLogger("uvicorn.access").setLevel(log_level)


@lru_cache
def get_settings() -> Settings:
    """
    Return cached application settings (singleton).

    :return: The process-local singleton Settings instance.
    """
    return Settings()
