# tests/test_settings.py

# --- Python Standard Library ---
import logging
import logging.handlers
from os import environ
from typing import Any
from unittest.mock import patch

# --- Third Party Libraries ---
from pydantic import SecretStr, ValidationError
import pytest
import structlog

# --- App modules ---
from aie_estimator.schemas.models import (
    DEFAULT_CLAUDE_LLM,
    DEFAULT_FAST_PROFILE,
    DEFAULT_GPT_LLM,
)
from aie_estimator.settings import Environment, Settings, get_settings


def test_get_settings_returns_cached_singleton() -> None:
    """
    GIVEN no prior calls or existing cached settings
    WHEN get_settings is invoked multiple times
    THEN the exact same Settings singleton instance is returned.
    """
    first_instance = get_settings()
    second_instance = get_settings()

    assert isinstance(first_instance, Settings)
    assert first_instance is second_instance


def test_get_settings_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    GIVEN an environment variable override for APP_ENV
    WHEN get_settings.cache_clear() is executed and get_settings is called
    THEN a reloaded Settings instance reflecting the new environment is returned.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()

    reloaded_settings = get_settings()
    assert reloaded_settings.app_env == Environment.TEST


@pytest.mark.parametrize(
    ("env_str", "expected_env"),
    [
        ("dev", Environment.DEV),
        ("prod", Environment.PROD),
        ("test", Environment.TEST),
    ],
    ids=["env_dev", "env_prod", "env_test"],
)
def test_settings_initialization_valid_env_strings(env_str: str, expected_env: Environment) -> None:
    """
    GIVEN valid environment string representations
    WHEN Settings is instantiated with app_env
    THEN Settings is successfully initialized with the corresponding Environment enum member.

    :param env_str: String representation of the target environment.
    :param expected_env: Expected Environment enum member.
    """
    settings = Settings(app_env=env_str)
    assert settings.app_env == expected_env


@pytest.mark.parametrize(
    ("kwargs", "expected_exception", "match_pattern"),
    [
        ({"app_env": "invalid_environment_name"}, ValueError, r"is not a valid"),
        ({"server_port": "invalid_port"}, ValidationError, r"valid integer"),
    ],
    ids=["invalid_env_string", "invalid_port_type"],
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
    ("env", "json_logging", "expected_console_renderer", "expected_file_renderer", "expected_file_colors"),
    [
        (Environment.DEV, False, structlog.dev.ConsoleRenderer, structlog.dev.ConsoleRenderer, False),
        (Environment.DEV, True, structlog.processors.JSONRenderer, structlog.processors.JSONRenderer, None),
        (Environment.PROD, False, structlog.processors.JSONRenderer, structlog.processors.JSONRenderer, None),
    ],
    ids=["dev_plain_console_and_file", "dev_forced_json", "prod_json_default"],
)
def test_configure_logging_renderer_selection(
    env: Environment,
    json_logging: bool,
    expected_console_renderer: type,
    expected_file_renderer: type,
    expected_file_colors: bool | None,
) -> None:
    """
    GIVEN an application environment and json_logging flag
    WHEN configure_logging is executed on Settings with file logging enabled
    THEN the console and rotating file handlers receive the expected structlog renderers without writing files.

    :param env: Target application environment.
    :param json_logging: Whether JSON logging is forced.
    :param expected_console_renderer: Expected terminal renderer type.
    :param expected_file_renderer: Expected file renderer type.
    :param expected_file_colors: Expected ANSI color flag on file renderer.
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

            # Verify filesystem directory creation was intercepted
            mock_mkdir.assert_called()

            # Verify RotatingFileHandler constructor parameters
            mock_file_handler_cls.assert_called()
            call_kwargs = mock_file_handler_cls.call_args.kwargs
            expected_filename = "app.log" if env == Environment.PROD else "dev.log"
            assert call_kwargs["filename"].name == expected_filename
            assert call_kwargs["maxBytes"] == 5 * 1024 * 1024
            assert call_kwargs["backupCount"] == 3
            assert call_kwargs["encoding"] == "utf-8"

            # Verify root logger handlers
            root_logger = logging.getLogger()
            handlers = root_logger.handlers
            assert len(handlers) == 2

            stream_handler = next(h for h in handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
            assert isinstance(stream_handler.formatter, structlog.stdlib.ProcessorFormatter)
            assert isinstance(stream_handler.formatter.processors[-1], expected_console_renderer)

            # Verify formatter assigned to mock file handler
            mock_file_handler.setFormatter.assert_called()
            file_formatter = mock_file_handler.setFormatter.call_args[0][0]
            assert isinstance(file_formatter, structlog.stdlib.ProcessorFormatter)
            assert isinstance(file_formatter.processors[-1], expected_file_renderer)
            if expected_file_colors is not None:
                assert getattr(file_formatter.processors[-1], "_colors", None) is expected_file_colors
        finally:
            logging.getLogger().handlers.clear()


@pytest.mark.parametrize(
    ("model_name", "expected_provider"),
    [
        (DEFAULT_GPT_LLM, "openai"),
        (DEFAULT_FAST_PROFILE, "litellm"),
        (DEFAULT_CLAUDE_LLM, "anthropic"),
    ],
    ids=["gpt6_luna_openai", "reasoner_fast_litellm", "claude_haiku_anthropic"],
)
def test_settings_llm_provider_derived_from_model(model_name: str, expected_provider: str) -> None:
    """
    GIVEN a model configured in Settings.llm_model
    WHEN the llm_provider property is accessed
    THEN it resolves to the provider defined in MODEL_CATALOG.

    :param model_name: Target model identifier.
    :param expected_provider: Expected provider identifier ('openai' or 'anthropic').
    """
    settings = Settings(app_env=Environment.TEST, llm_model=model_name)
    assert settings.llm_provider == expected_provider


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


def test_settings_server_host_configuration() -> None:
    """
    GIVEN default Settings initialization and custom override
    WHEN server_host is inspected
    THEN established defaults and overrides are verified.
    """
    default_settings = Settings(app_env=Environment.TEST)
    assert default_settings.server_host == "localhost"

    custom_settings = Settings(app_env=Environment.TEST, server_host="10.0.0.5")
    assert custom_settings.server_host == "10.0.0.5"


def test_settings_cache_configuration_defaults() -> None:
    """
    GIVEN default Settings initialization
    WHEN cache settings are inspected
    THEN redis_host, redis_port, redis_url, and cache_ttl reflect established defaults.
    """
    settings = Settings(app_env=Environment.TEST)
    assert settings.redis_host == "localhost"
    assert settings.redis_port == 6379
    assert settings.cache_ttl == 86400


@pytest.mark.parametrize(
    ("custom_host", "custom_port", "custom_ttl"),
    [
        ("10.0.0.1", 6380, 3600),
        ("cache.internal", 6379, 7200),
    ],
    ids=["custom_ip_port", "internal_host"],
)
def test_settings_cache_configuration_overrides(
    custom_host: str,
    custom_port: int,
    custom_ttl: int,
) -> None:
    """
    GIVEN custom redis_host, redis_port, and cache_ttl constructor overrides
    WHEN Settings is instantiated
    THEN the configured values override defaults and redis_url reflects them.

    :param custom_host: Custom Redis host.
    :param custom_port: Custom Redis port.
    :param custom_ttl: Custom cache time-to-live in seconds.
    """
    settings = Settings(
        app_env=Environment.TEST,
        redis_host=custom_host,
        redis_port=custom_port,
        cache_ttl=custom_ttl,
    )
    assert settings.redis_host == custom_host
    assert settings.redis_port == custom_port
    assert settings.cache_ttl == custom_ttl


@pytest.mark.parametrize(
    ("key_attribute", "env_variable", "test_value"),
    [
        ("mistral_api_key", "MISTRAL_API_KEY", "test-mistral-key"),
        ("scaleway_api_key", "SCALEWAY_API_KEY", "test-scaleway-key"),
        ("nebius_api_key", "NEBIUS_API_KEY", "test-nebius-key"),
    ],
    ids=["mistral_key", "scaleway_key", "nebius_key"],
)
def test_settings_additional_api_keys_initialization(
    key_attribute: str,
    env_variable: str,
    test_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an environment variable for a multi-provider LLM API key
    WHEN Settings is initialized
    THEN the key is loaded as a SecretStr with the corresponding secret value.

    :param key_attribute: Attribute name on Settings instance.
    :param env_variable: Environment variable name.
    :param test_value: Raw secret string value.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv(env_variable, test_value)
    settings = Settings(app_env=Environment.TEST)
    loaded_secret: SecretStr | None = getattr(settings, key_attribute)

    assert loaded_secret is not None
    assert isinstance(loaded_secret, SecretStr)
    assert loaded_secret.get_secret_value() == test_value


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

