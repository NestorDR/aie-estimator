# src/aie_estimator/telemetry.py

# --- Python Standard Library ---
from typing import TYPE_CHECKING, Any, Final

# --- Third Party Libraries ---
from opentelemetry import trace
import structlog

# --- App modules ---
if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from aie_estimator.settings import Settings

_SERVICE_NAME: Final[str] = "aie-estimator"


def inject_opentelemetry_context(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """
    Inject active OpenTelemetry trace_id and span_id into structlog event context.

    :param logger: The wrapped logger instance.
    :param method_name: The name of the logger method invoked.
    :param event_dict: The event dictionary to enrich with trace identifiers.
    
    :return: The enriched event dictionary.
    """
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = f"{span_context.trace_id:032x}"
        event_dict["span_id"] = f"{span_context.span_id:016x}"
    return event_dict


def setup_telemetry(settings: Settings) -> TracerProvider | None:
    """
    Initialize OpenTelemetry TracerProvider and instrument LiteLLM if enabled.

    :param settings: Application settings controlling telemetry configuration.
    :return: The initialized TracerProvider instance, or None if tracing is disabled.
    """
    if not settings.enable_otel_tracing:
        return None

    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": _SERVICE_NAME,
            "service.version": settings.app_version,
            "deployment.environment": settings.app_env.value,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    LiteLLMInstrumentor().instrument(tracer_provider=provider)
    return provider


def instrument_fastapi_app(app: Any, settings: Settings) -> None:
    """
    Instrument FastAPI application with OpenTelemetry middleware if enabled.

    :param app: The FastAPI application instance to instrument.
    :param settings: Application settings controlling telemetry configuration.
    """
    if not settings.enable_otel_tracing:
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)
