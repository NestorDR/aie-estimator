# src/aie_estimator/services/litellm_provider.py

# --- Python Standard Library ---
from functools import lru_cache
from os import environ
from typing import Any, Final

# --- Third Party Libraries ---
import litellm
from litellm import Router
import structlog
import yaml

# --- App modules ---
from aie_estimator.schemas.models import DEFAULT_FAST_PROFILE
from aie_estimator.services import LLMServiceError, format_provider_response
from aie_estimator.services.model_catalog import get_model_catalog, get_profiles_config_path

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


@lru_cache(maxsize=1)
def get_litellm_router() -> Router:
    """
    Initialize or return the cached singleton LiteLLM Router instance.

    Loads router configuration, enforces zero-telemetry and parameter normalization guardrails,
    and dynamically filters models whose environment credentials are missing or empty.

    :return: Configured LiteLLM Router instance.
    :raises LLMServiceError: If configuration cannot be loaded or no models have valid credentials.
    """
    config_path = get_profiles_config_path()
    if not config_path.is_file():
        log.error("litellm_profiles_file_not_found", config_path=str(config_path))
        raise LLMServiceError(f"LiteLLM profiles configuration file not found at {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except Exception as e:
        log.exception("litellm_profiles_read_failed", config_path=str(config_path), error=repr(e))
        raise LLMServiceError(f"Failed to read LiteLLM profiles configuration: {e}") from e

    # Apply general engine and zero-telemetry guardrails
    general_settings = config.get("general_settings", {})
    if general_settings.get("disable_telemetry", True):
        litellm.telemetry = False
    litellm.drop_params = general_settings.get("drop_params", True)
    litellm.modify_params = general_settings.get("modify_params", True)

    router_settings = config.get("router_settings", {})
    model_list = config.get("model_list", [])

    # Filter out models missing credentials in os.environ
    filtered_model_list: list[dict[str, Any]] = []
    for entry in model_list:
        litellm_params = entry.get("litellm_params", {})
        api_key_ref = litellm_params.get("api_key")

        if isinstance(api_key_ref, str) and api_key_ref.startswith("os.environ/"):
            env_var = api_key_ref.split("/", 1)[1]
            if not environ.get(env_var):
                log.info(
                    "litellm_model_skipped_missing_credentials",
                    model_name=entry.get("model_name"),
                    target_model=litellm_params.get("model"),
                    missing_env_var=env_var,
                )
                continue

        filtered_model_list.append(entry)

    if not filtered_model_list:
        log.error("litellm_router_no_valid_credentials", total_configured_models=len(model_list))
        raise LLMServiceError(
            "No models in Router have valid credentials configured in the environment. "
            "Please configure at least one provider API key."
        )

    log.info(
        "litellm_router_initialized",
        active_models_count=len(filtered_model_list),
        routing_strategy=router_settings.get("routing_strategy", "cost-based-routing"),
    )

    return Router(
        model_list=filtered_model_list,
        routing_strategy=router_settings.get("routing_strategy", "cost-based-routing"),
        num_retries=router_settings.get("num_retries", 2),
        timeout=router_settings.get("timeout", 30.0),
        fallbacks=router_settings.get("fallbacks"),
    )


async def call_litellm_router(
    messages: str | list[dict[str, Any]],
    system_prompt: str | None,
    model: str,
    max_tokens: int,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """
    Send an asynchronous estimation request through the LiteLLM Router with cost extraction and error handling.

    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional system instructions and reference context.
    :param model: Target model identifier or logical profile.
    :param max_tokens: Maximum output token budget.
    :param temperature: Optional sampling temperature.
    :param reasoning_effort: Optional qualitative reasoning effort tier ('low', 'medium', 'high').

    :return: Standardized response dictionary containing estimation, model, provider, finish_reason, usage, and cost_usd.
    :raises LLMServiceError: If the LiteLLM router call fails.
    """
    router = get_litellm_router()
    catalog = get_model_catalog()
    model_spec = catalog.get(model)
    target_profile = model if (model_spec and model_spec.is_profile) else DEFAULT_FAST_PROFILE

    formatted_messages: list[dict[str, Any]] = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})

    if isinstance(messages, str):
        formatted_messages.append({"role": "user", "content": messages})
    else:
        formatted_messages.extend(messages)

    kwargs: dict[str, Any] = {
        "model": target_profile,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = await router.acompletion(**kwargs)
    except Exception as e:
        log.exception(
            "litellm_router_call_failed",
            model=model,
            target_profile=target_profile,
            error_type=type(e).__name__,
            error=repr(e),
        )
        raise LLMServiceError(f"LiteLLM router call failed: {e}") from e

    choice = response.choices[0]

    # For a `Chat Completions` response, the documented `finish_reason` values are:
    #     - stop          : the model finished naturally or reached a stop sequence.
    #     - length        : the token limit was reached.
    #     - tool_calls    : the model requested a tool call.
    #     - content_filter: content was omitted due to content filtering.
    #     - function_call : a legacy value for a function call; deprecated.
    #
    # Visit [OpenAI Chat Completions API reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), which documents `finish_reason` and its values. [[1]](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
    finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
    estimation_text = getattr(choice.message, "content", "") or ""

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    hidden_params = getattr(response, "_hidden_params", {}) or {}
    raw_cost = hidden_params.get("response_cost")
    cost_usd = float(raw_cost) if raw_cost is not None else None
    actual_model = getattr(response, "model", None) or model

    # Derive provider identifier
    if model_spec and model_spec.is_profile:
        provider = "litellm"
    else:
        custom_provider = hidden_params.get("custom_llm_provider")
        provider = custom_provider or (model_spec.provider if model_spec else "litellm")

    log.info(
        "litellm_response_received",
        requested_model=model,
        actual_model=actual_model,
        provider=provider,
        finish_reason=finish_reason,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return format_provider_response(
        estimation=estimation_text,
        model=actual_model,
        provider=provider,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
