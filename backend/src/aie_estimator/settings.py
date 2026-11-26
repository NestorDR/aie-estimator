# src/aie_estimator/settings.py

# --- Python Standard Library ---
from enum import Enum
from functools import lru_cache
import importlib.metadata
from os import environ
from pathlib import Path
from typing import Final, Literal

# --- Third Party Libraries ---
from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

# --- App modules ---
from aie_estimator.utils.constants import (
    DEFAULT_FAST_PROFILE,
    DEFAULT_SESSION_MAX_ANCHOR_PAIRS,
    DEFAULT_SESSION_MAX_TURNS,
    DEFAULT_SESSION_TTL_SECONDS,
    PROVIDER_PREFIXES,
    SECONDS_PER_DAY,
)
from aie_estimator.utils.embeddings_utils import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()
_TEST_PII_HASH_SALT: Final[str] = "test-pii-hash-salt"


class Environment(str, Enum):
    """
    Defines the possible environments for the application itself (e.g., dev, test, prod).
    """

    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class Settings(BaseSettings):
    """Application settings loaded from environment variables for a specific environment."""

    # Application environment and version
    app_env: Environment = Environment.DEV
    app_version: str = "0.0.0"  # It will be updated with the version from the installed package metadata

    # Logging and observability settings
    log_level: str = "INFO"
    json_logging: bool = False
    enable_file_logging: bool = False
    enable_otel_tracing: bool = False
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"

    # Server settings
    server_host: str = "localhost"
    server_port: int = 8000

    # AI settings
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    scaleway_api_key: SecretStr | None = None
    nebius_api_key: SecretStr | None = None
    llm_model: str = DEFAULT_FAST_PROFILE
    active_prompt_version: str = "v3"
    conversational_prompt_version: str = "v3"
    litellm_local_model_cost_map: bool = True
    llm_profiles_path: Path = Path(__file__).resolve().parent / "llm_profiles.yml"

    # Embedding settings (RAG pipeline)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = Field(default=DEFAULT_EMBEDDING_DIMENSIONS, gt=0)
    embedding_batch_size: int = Field(default=100, gt=0)

    # Cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Exact cache settings
    enable_exact_cache: bool = True
    exact_cache_ttl: int = SECONDS_PER_DAY

    # Semantic cache settings
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_log_only: bool = False
    semantic_cache_ttl: int = SECONDS_PER_DAY

    # Persistence settings
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, gt=0)
    postgres_user: str = "estimator"
    postgres_password: SecretStr | None = None
    postgres_db: str = "estimator"

    # Offline catalog & ingestion settings
    catalog_path: Path = Path("data/catalog/catalog.yaml")
    ingestion_data_root: Path = Path("data/seed")

    # PII detection & GDPR pseudonymization settings
    pii_hash_salt: SecretStr | None = None
    pii_pseudonymization_enabled: bool = True
    pii_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    pii_spacy_model: str = "en_core_web_sm"
    pii_language: str = "en"
    pii_faker_locale: str = "en_US"

    # Attachment ingestion settings
    max_attachments_per_turn: int = Field(default=5, gt=0)
    max_attachment_bytes: int = Field(default=5_242_880, gt=0)  # 5 MB
    max_attachment_chars: int = Field(default=60_000, gt=0)
    max_total_attachment_chars: int = Field(default=60_000, gt=0)
    supported_attachment_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md"],
    )

    # Conversational session settings
    session_max_turns: int = Field(default=DEFAULT_SESSION_MAX_TURNS, gt=0)
    session_ttl_seconds: int = Field(default=DEFAULT_SESSION_TTL_SECONDS, gt=0)
    session_max_anchor_pairs: int = Field(default=DEFAULT_SESSION_MAX_ANCHOR_PAIRS, ge=0)

    # History compression settings
    enable_history_compression: bool = True
    anchor_strategy: Literal["heuristic", "llm"] = "heuristic"
    compression_model: str = DEFAULT_FAST_PROFILE

    # Metadata extraction settings
    enable_metadata_extraction: bool = True
    metadata_extractor_model: str = DEFAULT_FAST_PROFILE

    # Actor-Critic-Boss orchestration settings
    acb_max_iterations: int = Field(default=2, ge=1, le=5)
    acb_critic_model: str = DEFAULT_FAST_PROFILE

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
            log.info("environment_file_loaded", env_path=str(env_path))
        else:
            log.warning("environment_file_not_found", env_path=str(env_path))

        try:
            # Initialize the base class with the identified environment
            super().__init__(**kwargs)

            # Guarantee local cost map fallback and deterministic PII test salt in test environment
            if self.app_env == Environment.TEST:
                self.litellm_local_model_cost_map = True
                if self.pii_hash_salt is None:
                    self.pii_hash_salt = SecretStr(_TEST_PII_HASH_SALT)

            # Mirror secret API keys into os.environ for LiteLLM and external SDKs
            self.mirror_secrets_to_environ()

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

    def mirror_secrets_to_environ(self) -> None:
        """
        Mirror loaded API key secrets into os.environ for third-party libraries (e.g., LiteLLM) if not already set.
        """
        for provider in PROVIDER_PREFIXES:
            env_var = f"{provider.upper()}_API_KEY"
            secret_val: SecretStr | None = getattr(self, env_var.lower(), None)
            if secret_val is not None and not environ.get(env_var):
                environ[env_var] = secret_val.get_secret_value()

        if self.litellm_local_model_cost_map and not environ.get("LITELLM_LOCAL_MODEL_COST_MAP"):
            environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    def get_database_kwargs(self) -> dict[str, str | int] | None:
        """
        Return PostgreSQL connection keyword arguments when credentials are configured.

        :return: Connection kwargs dictionary when postgres_password is set, or None otherwise.
        """
        if self.postgres_password is None:
            return None
        return {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "password": self.postgres_password.get_secret_value(),
            "dbname": self.postgres_db,
        }

    def configure_logging(self) -> None:
        """
        Configure unified structlog and standard library logging handlers.

        Delegates to aie_estimator.logging.configure_logging.
        """
        from aie_estimator.logging import configure_logging

        configure_logging(self)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached application settings (singleton).

    :return: The process-local singleton Settings instance.
    """
    return Settings()
