# src/aie_estimator/settings.py

# --- Python Standard Library ---
from enum import Enum
from functools import lru_cache
import importlib.metadata
import json
import logging
from os import environ
from pathlib import Path
from typing import override

# --- Third Party Libraries ---
from pydantic import computed_field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """
    Defines the possible environments for the application itself (e.g., development, test, production).
    """

    DEV = "development"
    PROD = "production"
    TEST = "test"


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    :param fmt_dict: Mapping between key names in the outputted JSON log and keys from the LogRecord.

    :raise UnknownFormattingKeyException: If the key doesn't exist in LogRecord.
    """

    def __init__(self, fmt_dict: dict, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fmt_dict = fmt_dict

    @override
    def usesTime(self) -> bool:
        """
        Returns whether asctime is requested from LogRecord.
        """
        return "asctime" in self.fmt_dict.values()

    def _build_message_dict(self, record: logging.LogRecord) -> dict:
        """
        Return dict with requested LogRecord attributes.
        """
        try:
            return {k: record.__dict__[v] for k, v in self.fmt_dict.items()}

        except KeyError as e:
            print(f"Unknown key in log format: {e.args[0]}")
            return {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        """
        Adapted from the parent class to output as JSON instead of string.
        """
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record)

        message_dict = self._build_message_dict(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


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
    server_ip: str = "127.0.0.1"
    server_port: int = 8000

    # AI settings
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

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
        2. Loads the corresponding .env file (e.g., .env.development for DEV environment)
        3. Initializes configuration with values from environment variables and .env file

        Usage examples:
            # Explicit environment selection
            test_settings = Settings(app_env=Environment.DEV)
            # Override default port during initialization (e.g., in test fixtures)
            test_settings = Settings(app_env=Environment.TEST, server_port=8080)
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
            logger.info(f"Loading environment variables from: {env_path}")
        else:
            logger.warning(f"Environment file {env_path} not found. Using system environment variables only.")

        try:
            # Initialize the base class with the identified environment
            super().__init__(**kwargs)

            try:
                # Fetch the version from the installed package metadata
                installed_version = importlib.metadata.version("aie_estimator")
                if installed_version:
                    self.app_version = installed_version

            except importlib.metadata.PackageNotFoundError:
                pass

            logger.info(f"Settings loaded for environment: {self.app_env} (Version: {self.app_version})")

        except ValidationError as e:
            logger.exception("Initialization failed. Missing required environment variables or invalid values.", exc_info=e)
            raise

        except Exception as e:
            msg = "Unexpected error during initialization."
            logger.exception(msg, exc_info=e)
            raise Exception(msg) from e

    @computed_field
    def log_config(self) -> dict:
        str_fmt = "%(asctime)s - %(name)s : %(levelname)s - %(filename)s - %(lineno)d - %(message)s"

        json_fmt = {
            "level": "levelname",
            "message": "message",
            "logger": "name",
            "timestamp": "asctime",
            "filename": "filename",
            "line": "lineno",
        }

        handlers = ["console"]
        if self.enable_file_logging:
            handlers.append("file")

        logger_config = {
            "level": self.log_level,
            "handlers": [],
            "propagate": True,
        }

        loggers = [
            "aie_estimator",
            "asyncio",
            "psycopg",
            "uvicorn",
            "uvicorn.access",
            "uvicorn.error",
        ]

        _config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "complex": {"format": str_fmt},
                "json": {"()": "aie_estimator.settings.JsonFormatter", "fmt_dict": json_fmt},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": "json" if self.json_logging else "complex",
                    "stream": "ext://sys.stdout",
                },
            },
            # root logger will use the handlers, all other loggers will propagate to it
            "root": {
                "level": self.log_level,
                "handlers": handlers,
            },
            "loggers": {x: logger_config.copy() for x in loggers},
        }

        _config["loggers"]["asyncio"]["level"] = "WARNING"
        _config["loggers"]["psycopg"]["level"] = "WARNING"
        _config["loggers"]["uvicorn.access"]["level"] = "WARNING"

        if self.enable_file_logging:
            log_base_path = Path(__file__).resolve().parent.parent.parent  # go up 3 levels
            log_folder = log_base_path / "logs"
            log_folder.mkdir(parents=True, exist_ok=True)

            log_files = {Environment.PROD: "app.log", Environment.TEST: "test.log"}
            log_filename = log_files.get(Environment(self.app_env), "dev.log")
            log_file_path = log_folder / log_filename

            handler_class = "handlers.RotatingFileHandler" if self.app_env == Environment.PROD else "FileHandler"
            handler_params = {
                "level": self.log_level,
                "formatter": "complex",
                "filename": log_file_path,
                **(
                    {
                        "maxBytes": 524288,  # 512 KB
                        "backupCount": 3,
                        "encoding": "utf-8",
                    }
                    if self.app_env == Environment.PROD
                    else {}
                ),
            }
            _config["handlers"]["file"] = {"class": f"logging.{handler_class}", **handler_params}

        return _config


@lru_cache
def get_settings() -> Settings:
    """
    Return cached application settings (singleton).

    :return: The process-local singleton Settings instance.
    """
    return Settings()
