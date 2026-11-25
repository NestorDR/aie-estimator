# tests/test_settings.py

# --- Python Standard Library ---
import json
import logging
import logging.handlers
from os import environ
from pathlib import Path
from typing import Any
from unittest.mock import patch

# --- Third Party Libraries ---
from pydantic import SecretStr, ValidationError
import pytest
import structlog

# --- App modules ---
from aie_estimator.settings import Environment, Settings, get_settings


@pytest.mark.parametrize(
    ("kwargs", "expected_exception", "match_pattern"),
    [
        ({"app_env": "invalid_environment_name"}, ValueError, r"is not a valid"),
        ({"server_port": "invalid_port"}, ValidationError, r"valid integer"),
        ({"max_attachments_per_turn": 0}, ValidationError, r"greater than 0"),
        ({"max_attachment_bytes": 0}, ValidationError, r"greater than 0"),
        ({"max_attachment_chars": -1}, ValidationError, r"greater than 0"),
        ({"max_total_attachment_chars": 0}, ValidationError, r"greater than 0"),
        ({"session_max_turns": 0}, ValidationError, r"greater than 0"),
        ({"session_ttl_seconds": 0}, ValidationError, r"greater than 0"),
    ],
    ids=[
        "invalid_env_string",
        "invalid_port_type",
        "max_attachments_per_turn_non_positive",
        "max_attachment_bytes_non_positive",
        "max_attachment_chars_negative",
        "max_total_attachment_chars_non_positive",
        "session_max_turns_non_positive",
        "session_ttl_seconds_non_positive",
    ],
)
def test_settings_initialization_validation_errors(
    kwargs: dict[str, Any],
    expected_exception: type[Exception],
    match_pattern: str,
) -> None:
    """
    GIVEN invalid configuration arguments
    WHEN Settings is instantiated
    THEN the expected exception is raised matching concise keywords.

    :param kwargs: Keyword arguments to pass to Settings constructor.
    :param expected_exception: Expected exception class.
    :param match_pattern: Concise regex pattern targeting distinctive error keywords.
    """
    with pytest.raises(expected_exception, match=match_pattern):
        Settings(**kwargs)


@pytest.mark.parametrize(
    ("env", "json_logging", "expected_is_json"),
    [
        (Environment.DEV, False, False),
        (Environment.DEV, True, True),
        (Environment.PROD, False, True),
    ],
    ids=["dev_plain_console_and_file", "dev_forced_json", "prod_json_default"],
)
def test_configure_logging_renderer_selection(
    capsys: pytest.CaptureFixture[str],
    env: Environment,
    json_logging: bool,
    expected_is_json: bool,
) -> None:
    """
    GIVEN an application environment and json_logging flag
    WHEN configure_logging is executed on Settings with file logging enabled
    THEN console output and file formatter produce JSON or plain text as configured without writing files.

    :param capsys: Pytest standard output/error capture fixture.
    :param env: Target application environment.
    :param json_logging: Whether JSON logging is forced.
    :param expected_is_json: Whether emitted logs should be valid JSON.
    """
    with (
        patch("logging.handlers.RotatingFileHandler", autospec=True) as mock_file_handler_cls,
        patch("pathlib.Path.mkdir") as mock_mkdir,
    ):
        mock_file_handler = mock_file_handler_cls.return_value
        mock_file_handler.level = logging.NOTSET

        try:
            settings = Settings(
                app_env=env,
                json_logging=json_logging,
                enable_file_logging=True,
            )
            settings.configure_logging()

            mock_mkdir.assert_called()
            mock_file_handler_cls.assert_called()
            call_kwargs = mock_file_handler_cls.call_args.kwargs
            expected_filename = "app.log" if env == Environment.PROD else "dev.log"
            assert call_kwargs["filename"].name == expected_filename
            assert call_kwargs["maxBytes"] == 5 * 1024 * 1024
            assert call_kwargs["backupCount"] == 3
            assert call_kwargs["encoding"] == "utf-8"

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2

            capsys.readouterr()
            structlog.get_logger("test_renderer").info("renderer_probe")
            captured = capsys.readouterr()
            combined_output = "\n".join(part for part in (captured.out.strip(), captured.err.strip()) if part)
            log_line = combined_output.splitlines()[-1]
            assert "renderer_probe" in log_line

            mock_file_handler.setFormatter.assert_called()
            file_formatter = mock_file_handler.setFormatter.call_args[0][0]
            assert isinstance(file_formatter, structlog.stdlib.ProcessorFormatter)
            if expected_is_json:
                assert json.loads(log_line)["event"] == "renderer_probe"
            else:
                with pytest.raises(json.JSONDecodeError):
                    json.loads(log_line)
                assert "\x1b[" not in log_line
        finally:
            logging.getLogger().handlers.clear()


def test_configure_logging_captures_callsite_parameters(capsys: pytest.CaptureFixture[str]) -> None:
    """
    GIVEN configured logging with CallsiteParameterAdder in shared_processors
    WHEN a structured log event is emitted
    THEN filename, lineno, and func_name are captured and present in the log output.

    :param capsys: Pytest standard output/error capture fixture.
    """
    settings = Settings(
        app_env=Environment.DEV,
        json_logging=True,
        enable_file_logging=False,
    )
    settings.configure_logging()

    test_logger = structlog.get_logger("test_callsite")
    test_logger.info("callsite_probe")

    captured = capsys.readouterr()
    log_text = captured.out or captured.err
    assert "callsite_probe" in log_text
    assert "filename" in log_text
    assert "test_settings.py" in log_text
    assert "lineno" in log_text
    assert "func_name" in log_text


@pytest.mark.parametrize(
    ("attr_name", "expected_default"),
    [
        ("enable_exact_cache", True),
        ("exact_cache_ttl", 86400),
        ("enable_semantic_cache", True),
        ("semantic_cache_threshold", 0.92),
        ("semantic_cache_log_only", False),
        ("semantic_cache_ttl", 86400),
        ("redis_host", "localhost"),
        ("redis_port", 6379),
        ("active_prompt_version", "v3"),
        ("conversational_prompt_version", "v3"),
        ("enable_otel_tracing", False),
        ("max_attachments_per_turn", 5),
        ("max_attachment_bytes", 5_242_880),
        ("max_attachment_chars", 60_000),
        ("max_total_attachment_chars", 60_000),
        ("supported_attachment_extensions", [".pdf", ".docx", ".txt", ".md"]),
        ("enable_metadata_extraction", True),
        ("metadata_extractor_model", "reasoner-fast"),
    ],
    ids=[
        "enable_exact_cache",
        "exact_cache_ttl",
        "enable_semantic_cache",
        "semantic_cache_threshold",
        "semantic_cache_log_only",
        "semantic_cache_ttl",
        "redis_host",
        "redis_port",
        "active_prompt_version",
        "conversational_prompt_version",
        "enable_otel_tracing",
        "max_attachments_per_turn",
        "max_attachment_bytes",
        "max_attachment_chars",
        "max_total_attachment_chars",
        "supported_attachment_extensions",
        "enable_metadata_extraction",
        "metadata_extractor_model",
    ],
)
def test_settings_default_values(attr_name: str, expected_default: Any) -> None:
    """
    GIVEN default Settings initialization under test environment
    WHEN an instance is created without explicit parameter overrides
    THEN each configuration attribute resolves to its canonical default value.

    :param attr_name: Name of the Settings attribute to verify.
    :param expected_default: Expected default value.
    """
    settings = Settings(app_env=Environment.TEST)
    assert getattr(settings, attr_name) == expected_default


@pytest.mark.parametrize(
    ("env_var", "env_val", "attr_name", "expected_val"),
    [
        # Exact cache
        ("ENABLE_EXACT_CACHE", "false", "enable_exact_cache", False),
        ("ENABLE_EXACT_CACHE", "true", "enable_exact_cache", True),
        ("EXACT_CACHE_TTL", "3600", "exact_cache_ttl", 3600),
        # Semantic cache
        ("ENABLE_SEMANTIC_CACHE", "false", "enable_semantic_cache", False),
        ("SEMANTIC_CACHE_THRESHOLD", "0.88", "semantic_cache_threshold", 0.88),
        ("SEMANTIC_CACHE_LOG_ONLY", "true", "semantic_cache_log_only", True),
        ("SEMANTIC_CACHE_TTL", "7200", "semantic_cache_ttl", 7200),
        # Redis connection
        ("REDIS_HOST", "redis.internal", "redis_host", "redis.internal"),
        ("REDIS_PORT", "6380", "redis_port", 6380),
        # General & Telemetry
        ("ACTIVE_PROMPT_VERSION", "v2", "active_prompt_version", "v2"),
        ("CONVERSATIONAL_PROMPT_VERSION", "v3", "conversational_prompt_version", "v3"),
        ("ENABLE_OTEL_TRACING", "true", "enable_otel_tracing", True),
        ("ENABLE_OTEL_TRACING", "false", "enable_otel_tracing", False),
        # Multi-provider API keys
        ("SCALEWAY_API_KEY", "test-scaleway-key", "scaleway_api_key", "test-scaleway-key"),
        ("NEBIUS_API_KEY", "test-nebius-key", "nebius_api_key", "test-nebius-key"),
        # Attachment ingestion
        ("MAX_ATTACHMENTS_PER_TURN", "3", "max_attachments_per_turn", 3),
        ("MAX_ATTACHMENT_BYTES", "1048576", "max_attachment_bytes", 1_048_576),
        ("MAX_ATTACHMENT_CHARS", "8000", "max_attachment_chars", 8000),
        ("MAX_TOTAL_ATTACHMENT_CHARS", "20000", "max_total_attachment_chars", 20_000),
        ("SUPPORTED_ATTACHMENT_EXTENSIONS", '[".pdf",".md"]', "supported_attachment_extensions", [".pdf", ".md"]),
        # Metadata extraction
        ("ENABLE_METADATA_EXTRACTION", "false", "enable_metadata_extraction", False),
        ("METADATA_EXTRACTOR_MODEL", "gpt-6-luna", "metadata_extractor_model", "gpt-6-luna"),
    ],
    ids=[
        "enable_exact_cache_false",
        "enable_exact_cache_true",
        "exact_cache_ttl",
        "enable_semantic_cache_false",
        "semantic_cache_threshold",
        "semantic_cache_log_only_true",
        "semantic_cache_ttl",
        "redis_host",
        "redis_port",
        "active_prompt_version",
        "conversational_prompt_version",
        "enable_otel_tracing_true",
        "enable_otel_tracing_false",
        "scaleway_api_key",
        "nebius_api_key",
        "max_attachments_per_turn",
        "max_attachment_bytes",
        "max_attachment_chars",
        "max_total_attachment_chars",
        "supported_attachment_extensions",
        "enable_metadata_extraction_false",
        "metadata_extractor_model",
    ],
)
def test_settings_environment_variable_overrides(
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    env_val: str,
    attr_name: str,
    expected_val: Any,
) -> None:
    """
    GIVEN an environment variable override
    WHEN Settings is initialized
    THEN the target attribute is correctly populated with the converted value.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param env_var: Environment variable name.
    :param env_val: String value set in the environment.
    :param attr_name: Settings attribute name to check.
    :param expected_val: Expected parsed value or secret string value.
    """
    monkeypatch.setenv(env_var, env_val)
    settings = Settings(app_env=Environment.TEST)
    actual_val = getattr(settings, attr_name)
    if isinstance(actual_val, SecretStr):
        assert actual_val.get_secret_value() == expected_val
    else:
        assert actual_val == expected_val


def test_mirror_secrets_to_environ_sets_missing_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN a Settings instance with configured secret keys and unset environment variables
    WHEN mirror_secrets_to_environ is called
    THEN the values are populated into os.environ.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    settings = Settings(
        app_env=Environment.TEST,
        openai_api_key=SecretStr("sk-test-openai"),
        mistral_api_key=SecretStr("sk-test-mistral"),
    )
    settings.mirror_secrets_to_environ()

    assert environ.get("OPENAI_API_KEY") == "sk-test-openai"
    assert environ.get("MISTRAL_API_KEY") == "sk-test-mistral"


def test_mirror_secrets_to_environ_does_not_overwrite_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an environment variable already present in os.environ
    WHEN mirror_secrets_to_environ is called with a conflicting setting
    THEN the existing os.environ value is preserved.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "existing-preset-key")

    settings = Settings(
        app_env=Environment.TEST,
    )
    settings.mistral_api_key = SecretStr("new-attempted-key")
    settings.mirror_secrets_to_environ()

    assert environ.get("MISTRAL_API_KEY") == "existing-preset-key"


def test_settings_litellm_local_model_cost_map_test_env() -> None:
    """
    GIVEN an instantiation of Settings with app_env=Environment.TEST
    WHEN the instance initializes
    THEN litellm_local_model_cost_map is automatically forced to True.
    """
    settings = Settings(app_env=Environment.TEST)
    assert settings.litellm_local_model_cost_map is True


def test_settings_llm_profiles_path_default_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    GIVEN default Settings and an environment variable override for LLM_PROFILES_PATH
    WHEN Settings is instantiated
    THEN the default resolves to package llm_profiles.yml and the override reflects the specified Path.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Pytest temporary directory fixture.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.llm_profiles_path.name == "llm_profiles.yml"
    assert default_settings.llm_profiles_path.is_file()

    custom_path = tmp_path / "custom_profiles.yml"
    monkeypatch.setenv("LLM_PROFILES_PATH", str(custom_path))
    overridden_settings = Settings(app_env=Environment.TEST)
    assert overridden_settings.llm_profiles_path == custom_path


def test_monkeypatch_environment_mutation_clears_get_settings_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN cached get_settings singleton
    WHEN environment variable is mutated or deleted via monkeypatch
    THEN get_settings cache is automatically invalidated without manual cache_clear.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    initial_settings = get_settings()
    assert initial_settings.active_prompt_version == "v3"

    monkeypatch.setenv("ACTIVE_PROMPT_VERSION", "v2")
    updated_settings = get_settings()
    assert updated_settings.active_prompt_version == "v2"
    assert updated_settings is not initial_settings

    monkeypatch.delenv("ACTIVE_PROMPT_VERSION", raising=False)
    reset_settings = get_settings()
    assert reset_settings.active_prompt_version == "v3"
    assert reset_settings is not updated_settings


def test_settings_session_defaults_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN default Settings and SESSION_MAX_TURNS / SESSION_TTL_SECONDS env overrides
    WHEN Settings is instantiated
    THEN defaults are 6 turns and 86400 seconds and env overrides take effect.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.session_max_turns == 6
    assert default_settings.session_ttl_seconds == 86_400

    monkeypatch.setenv("SESSION_MAX_TURNS", "10")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")
    overridden = Settings(app_env=Environment.TEST)
    assert overridden.session_max_turns == 10
    assert overridden.session_ttl_seconds == 3600


def test_settings_history_compression_defaults_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN default Settings and history compression env overrides
    WHEN Settings is instantiated
    THEN defaults match expected values and env overrides take effect.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.enable_history_compression is True
    assert default_settings.anchor_strategy == "heuristic"
    assert default_settings.compression_model == "reasoner-fast"
    assert default_settings.session_max_anchor_pairs == 4

    monkeypatch.setenv("ENABLE_HISTORY_COMPRESSION", "false")
    monkeypatch.setenv("ANCHOR_STRATEGY", "llm")
    monkeypatch.setenv("COMPRESSION_MODEL", "reasoner-deep")
    monkeypatch.setenv("SESSION_MAX_ANCHOR_PAIRS", "2")
    overridden = Settings(app_env=Environment.TEST)
    assert overridden.enable_history_compression is False
    assert overridden.anchor_strategy == "llm"
    assert overridden.compression_model == "reasoner-deep"
    assert overridden.session_max_anchor_pairs == 2


def test_settings_persistence_defaults_and_kwargs_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN default test settings and discrete POSTGRES_* environment overrides
    WHEN Settings is instantiated and get_database_kwargs is called
    THEN postgres_password defaults to None in test and configured values return masked kwargs.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.postgres_password is None
    assert default_settings.get_database_kwargs() is None

    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "est_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss:word/1 +a")
    monkeypatch.setenv("POSTGRES_DB", "est_db")
    overridden = Settings(app_env=Environment.TEST)
    assert isinstance(overridden.postgres_password, SecretStr)
    assert overridden.get_database_kwargs() == {
        "host": "postgres",
        "port": 5432,
        "user": "est_user",
        "password": "p@ss:word/1 +a",
        "dbname": "est_db",
    }
    assert "p@ss:word/1 +a" not in repr(overridden)


def test_settings_catalog_paths_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    GIVEN default test settings and INGESTION_DATA_ROOT / CATALOG_PATH overrides
    WHEN Settings is instantiated
    THEN defaults resolve to data/seed and data/catalog/catalog.yaml and env overrides take effect.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param tmp_path: Temporary directory fixture.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.ingestion_data_root == Path("data/seed")
    assert default_settings.catalog_path == Path("data/catalog/catalog.yaml")

    custom_root = tmp_path / "custom_seed"
    custom_catalog = tmp_path / "custom.yaml"
    monkeypatch.setenv("INGESTION_DATA_ROOT", str(custom_root))
    monkeypatch.setenv("CATALOG_PATH", str(custom_catalog))
    overridden = Settings(app_env=Environment.TEST)
    assert overridden.ingestion_data_root == custom_root
    assert overridden.catalog_path == custom_catalog


def test_settings_pii_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN default test settings, non-test settings, and PII_* environment overrides
    WHEN Settings is instantiated
    THEN test mode injects a deterministic redacted salt, non-test leaves it None, and env overrides apply.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.delenv("PII_HASH_SALT", raising=False)
    test_settings = Settings(app_env=Environment.TEST)
    assert test_settings.pii_pseudonymization_enabled is True
    assert test_settings.pii_confidence_threshold == 0.7
    assert test_settings.pii_spacy_model == "en_core_web_sm"
    assert test_settings.pii_language == "en"
    assert test_settings.pii_faker_locale == "en_US"
    assert isinstance(test_settings.pii_hash_salt, SecretStr)
    assert test_settings.pii_hash_salt.get_secret_value() == "test-pii-hash-salt"
    assert "test-pii-hash-salt" not in repr(test_settings)

    # Isolate from any local unversioned .env.prod so the non-test default (None) is deterministic.
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p.name == ".env.prod" else original_exists(p))
    prod_settings = Settings(app_env=Environment.PROD)
    assert prod_settings.pii_hash_salt is None

    monkeypatch.setenv("PII_HASH_SALT", "prod-secret-salt")
    monkeypatch.setenv("PII_PSEUDONYMIZATION_ENABLED", "false")
    monkeypatch.setenv("PII_CONFIDENCE_THRESHOLD", "0.85")
    monkeypatch.setenv("PII_SPACY_MODEL", "es_core_news_md")
    monkeypatch.setenv("PII_LANGUAGE", "es")
    monkeypatch.setenv("PII_FAKER_LOCALE", "es_ES")
    overridden = Settings(app_env=Environment.TEST)
    assert overridden.pii_hash_salt is not None
    assert overridden.pii_hash_salt.get_secret_value() == "prod-secret-salt"
    assert overridden.pii_pseudonymization_enabled is False
    assert overridden.pii_confidence_threshold == 0.85
    assert overridden.pii_spacy_model == "es_core_news_md"
    assert overridden.pii_language == "es"
    assert overridden.pii_faker_locale == "es_ES"

    with pytest.raises(ValidationError, match=r"less than or equal to 1"):
        Settings(app_env=Environment.TEST, pii_confidence_threshold=1.5)
