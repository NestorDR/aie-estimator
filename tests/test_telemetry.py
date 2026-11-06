# tests/test_telemetry.py

# --- Python Standard Library ---
from typing import Any
from unittest.mock import MagicMock, patch

# --- Third Party Libraries ---
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags
import pytest

# --- App modules ---
from aie_estimator.settings import Environment, Settings
from aie_estimator.telemetry import (
    inject_opentelemetry_context,
    instrument_fastapi_app,
    setup_telemetry,
)


def test_inject_opentelemetry_context_with_active_span() -> None:
    """
    GIVEN an active OpenTelemetry span with valid trace and span IDs
    WHEN inject_opentelemetry_context is called on an event dictionary
    THEN trace_id and span_id are added in hexadecimal format.
    """
    trace_id = 0x12345678123456781234567812345678
    span_id = 0x8765432187654321
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    mock_span = NonRecordingSpan(span_context)

    with patch.object(trace, "get_current_span", return_value=mock_span):
        event_dict: dict[str, Any] = {"event": "test_event"}
        enriched = inject_opentelemetry_context(MagicMock(), "info", event_dict)

        assert enriched["trace_id"] == f"{trace_id:032x}"
        assert enriched["span_id"] == f"{span_id:016x}"
        assert enriched["event"] == "test_event"


def test_inject_opentelemetry_context_with_inactive_span() -> None:
    """
    GIVEN no active recording span
    WHEN inject_opentelemetry_context is called on an event dictionary
    THEN trace_id and span_id are not added to the dictionary.
    """
    with patch.object(trace, "get_current_span", return_value=trace.INVALID_SPAN):
        event_dict: dict[str, Any] = {"event": "test_event"}
        enriched = inject_opentelemetry_context(MagicMock(), "info", event_dict)

        assert "trace_id" not in enriched
        assert "span_id" not in enriched
        assert enriched["event"] == "test_event"


def test_setup_telemetry_disabled() -> None:
    """
    GIVEN settings with enable_otel_tracing disabled
    WHEN setup_telemetry is executed
    THEN None is returned and no tracer provider is registered.
    """
    settings = Settings(app_env=Environment.TEST, enable_otel_tracing=False)
    result = setup_telemetry(settings)
    assert result is None


def test_setup_telemetry_enabled() -> None:
    """
    GIVEN settings with enable_otel_tracing enabled
    WHEN setup_telemetry is executed with mocked network dependencies
    THEN a TracerProvider is created and LiteLLM is instrumented.
    """
    settings = Settings(
        app_env=Environment.TEST,
        enable_otel_tracing=True,
        otel_exporter_otlp_endpoint="http://localhost:4317",
    )
    with (
        patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter") as mock_exporter,
        patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_processor,
        patch("openinference.instrumentation.litellm.LiteLLMInstrumentor") as mock_litellm,
    ):
        provider = setup_telemetry(settings)
        assert provider is not None
        mock_exporter.assert_called_once_with(endpoint="http://localhost:4317", insecure=True)
        mock_processor.assert_called_once()
        mock_litellm.return_value.instrument.assert_called_once_with(tracer_provider=provider)


@pytest.mark.parametrize("enabled", [True, False], ids=["tracing_enabled", "tracing_disabled"])
def test_instrument_fastapi_app(enabled: bool) -> None:
    """
    GIVEN a FastAPI application and a configurable tracing setting
    WHEN instrument_fastapi_app is called
    THEN FastAPIInstrumentor.instrument_app is invoked only if tracing is enabled.

    :param enabled: Boolean flag indicating if tracing is enabled.
    """
    settings = Settings(app_env=Environment.TEST, enable_otel_tracing=enabled)
    mock_app = MagicMock()

    with patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor:
        instrument_fastapi_app(mock_app, settings)
        if enabled:
            mock_instrumentor.instrument_app.assert_called_once_with(mock_app)
        else:
            mock_instrumentor.instrument_app.assert_not_called()
