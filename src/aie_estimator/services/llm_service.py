# src/aie_estimator/services/llm_service.py

# --- Python Standard Library ---
from dataclasses import dataclass
import random
import time
from typing import Any, Final

# --- Third Party Libraries ---
from anthropic import Anthropic
from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
import structlog

# --- App modules ---
from aie_estimator.context.examples import format_examples_for_prompt, select_examples
from aie_estimator.schemas.estimations import ExampleFormat, PreprocessingMode
from aie_estimator.schemas.models import MODEL_CATALOG
from aie_estimator.services.cache import EstimationCache, get_estimation_cache
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

SYSTEM_INSTRUCTION_PREFIX: Final[str] = (
    "You are an expert software engineering estimator. Your responsibility is to analyze client meeting "
    "transcripts and generate realistic, well-structured effort estimations including task breakdowns, hour "
    "estimates, team composition, and overall project duration."
    "Below are historical reference estimations to guide your estimation format, detail level, and sizing:\n"
)

DEFAULT_MAX_TOKENS: Final[int] = 4000
EXTRACTION_MAX_TOKENS: Final[int] = 1500


class LLMServiceError(Exception):
    """Raised when the LLM provider call fails."""


# region Client Lifecycle

_openai_client: OpenAI | None = None
_anthropic_client: Anthropic | None = None


def get_openai_client() -> OpenAI:
    """
    Retrieve or initialize the cached OpenAI API client instance.

    :return: Authenticated OpenAI client.
    :raises LLMServiceError: If the OpenAI API key is missing.
    """
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        if not api_key:
            raise LLMServiceError("OpenAI API key is not configured in the application environment.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_anthropic_client() -> Anthropic:
    """
    Retrieve or initialize the cached Anthropic API client instance.

    :return: Authenticated Anthropic client with configured retries.
    :raises LLMServiceError: If the Anthropic API key is missing.
    """
    global _anthropic_client
    if _anthropic_client is None:
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        if not api_key:
            raise LLMServiceError("Anthropic API key is not configured in the application environment.")
        _anthropic_client = Anthropic(api_key=api_key, max_retries=2)
    return _anthropic_client


def _reset_clients() -> None:
    """
    Reset cached client instances to force re-initialization.

    Used primarily in testing environments when settings or environment variables are overridden.
    """
    global _openai_client, _anthropic_client
    _openai_client = None
    _anthropic_client = None


# endregion Client Lifecycle


# region Prompt building blocks
# ---------------------------------------------------------------------------
# The two ACTIVE_OUTPUT_PROMPT variants live side by side so you
# can switch between them by editing the ACTIVE_OUTPUT_PROMPT assignment below.
# Uvicorn `--reload` picks up the change automatically.
# ---------------------------------------------------------------------------

PROMPT_OUTPUT_BASIC = "Generate an estimation for the project described above."

PROMPT_OUTPUT_STRUCTURED = """
Generate the estimation with this exact structure:

## Project summary
[2-3 sentences describing the project scope and goals]

## Task breakdown
| Task | Hours | Cost (EUR) |
[one row per task; cost = hours * 62.50 EUR for developer tasks]

## Totals
- Total hours: [number]
- Total cost: [number] EUR
- Recommended team: [composition]
- Estimated duration: [weeks]

## Risks and assumptions
- [3-5 bullet points covering technical risks, scope assumptions, and external dependencies]
"""

# Switch: change the right-hand side to PROMPT_OUTPUT_STRUCTURED
ACTIVE_OUTPUT_PROMPT = PROMPT_OUTPUT_BASIC

INLINE_CLEANING_BLOCK = """
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation.
"""

# System prompt to instruct the LLM to extract structured data (e.g., JSON) from unstructured text.
EXTRACTION_SYSTEM_PROMPT = """
You are an analyst. Read the meeting transcription and produce a clean, deduplicated bullet list of 
functional requirements, non-functional requirements, integrations, constraints and explicit deadlines. 
Ignore fillers, divagations and off-topic remarks. Output Markdown only.
"""


# endregion Prompt building blocks


@dataclass
class GenerationOptions:
    """Parameters specific to each request that determine the construction of the prompt and the call to the LLM."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget: int | None = None


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 3,
    use_examples: bool = True,
    inline_cleaning: bool = False,
) -> str:
    """
    System prompt construction.
    Assemble the CAG system prompt containing estimator role instructions and few-shot context examples.

    :param example_format: Output format used to serialize few-shot examples.
    :param num_examples: Number of historical reference examples to embed.
    :param use_examples: Whether few-shot context examples should be injected.
    :param inline_cleaning: Whether inline transcript cleaning instructions should be applied.

    :return: Formatted system prompt string with embedded historical estimations.
    """
    role = (
        "You are a senior software consultant with 15+ years of experience in project "
        "estimation. Your task is to produce a detailed software project estimation based "
        "on a meeting transcription provided by the user."
    )
    rates = (
        "Use a developer rate of approximately 62.50 EUR/hour (500 EUR/day) and a designer "
        "rate of approximately 50 EUR/hour (400 EUR/day). Provide realistic, well-justified "
        "numbers."
    )

    examples_block = ""
    if use_examples and num_examples > 0:
        rendered = format_examples_for_prompt(select_examples(num_examples), example_format)
        if rendered:
            examples_block = (
                "Below are reference estimations from previous projects. Use them as a guide "
                "for structure, level of detail, and realistic pricing. Adapt the content to "
                "match the specific project described in the transcription.\n\n" + rendered
            )

    cleaning_block = INLINE_CLEANING_BLOCK if inline_cleaning else ""

    sections = [role, cleaning_block, rates, ACTIVE_OUTPUT_PROMPT, examples_block]
    return "\n\n".join(s for s in sections if s)


# region Two-phase preprocessing (phase 1: requirement extraction)


def extract_requirements(
    transcription: str,
    opts: GenerationOptions,
) -> tuple[str, dict]:
    """
    Run phase-1 LLM call to extract clean requirements from raw transcription.

    :param transcription: Raw meeting transcript text.
    :param opts: Generation options configuring model and execution settings.

    :return: Tuple of clean requirements text and token usage dictionary (keys: 'input' and 'output', values: token counts for downstream accounting).
    :raises LLMServiceError: If the requirement extraction LLM call fails.
    """
    settings = get_settings()
    model = opts.model or settings.llm_model
    model_spec = MODEL_CATALOG.get(model)
    provider = model_spec.provider if model_spec is not None else settings.llm_provider

    log.info("extracting_requirements", provider=provider, model=model)

    if provider == "openai":
        result = _call_openai(
            messages=transcription,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            model=model,
            max_tokens=EXTRACTION_MAX_TOKENS,
        )
    else:
        result = _call_anthropic(
            messages=transcription,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            model=model,
            max_tokens=EXTRACTION_MAX_TOKENS,
            thinking_budget=None,
        )

    return result["estimation"], {
        "input": result["usage"]["input_tokens"],
        "output": result["usage"]["output_tokens"],
    }


# endregion Two-phase preprocessing (phase 1: requirement extraction)

# region Provider wrappers


def call_with_retry(
    client: OpenAI,
    messages: str | list[dict[str, Any]],
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.3,
    effort: str = "medium",
    max_retries: int = 3,
) -> Any:
    """
    Execute a request against OpenAI Responses API with exponential backoff and jitter retry logic.

    :param client: Authenticated OpenAI client instance.
    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional instructions and context to be injected to the model.
    :param model: Target model identifier.
    :param max_tokens: Maximum output token budget.
    :param temperature: Randomness parameter for non-reasoning schemas.
    :param effort: Reasoning effort for reasoning-capable schemas.
    :param max_retries: Maximum number of attempts for transient failures.

    :return: The OpenAI response object or None if attempts are exhausted.
    :raises BadRequestError: If unrecoverable, a malformed request occurs.
    :raises RateLimitError: If rate limits persist after all retries.
    :raises APIConnectionError: If network connectivity fails after all retries.
    :raises InternalServerError: If upstream server errors persist after all retries.
    """
    spec = MODEL_CATALOG.get(model)
    is_reasoning_model = spec.is_reasoning if spec is not None else False

    request_params: dict[str, Any] = {
        "model": model,
        "input": messages,
        "max_output_tokens": max_tokens,
        "store": False,
    }

    if system_prompt:
        request_params["instructions"] = system_prompt

    if is_reasoning_model:
        request_params["reasoning"] = {"effort": effort}
    elif temperature is not None:
        request_params["temperature"] = temperature

    for attempt in range(max_retries):
        try:
            return client.responses.create(**request_params)

        except (BadRequestError, RateLimitError, InternalServerError, APIConnectionError) as e:
            if isinstance(e, BadRequestError) and "temperature" in str(e) and "temperature" in request_params:
                request_params.pop("temperature", None)

            if attempt >= max_retries - 1:
                raise

            jitter: float = random.SystemRandom().uniform(0, 1)
            wait: float = (2**attempt) + jitter
            log.warning(f"Transient error from OpenAI API (attempt {attempt + 1}/{max_retries}). Retrying in {wait:.2f}s: {e!r}")
            time.sleep(wait)

    return None


def _call_openai(
    messages: str | list[dict[str, Any]],
    system_prompt: str | None,
    model: str,
    max_tokens: int,
    temperature: float = 0.3,
    effort: str = "medium",
) -> dict[str, Any]:
    """
    Send an estimation request to the OpenAI Responses API with retries and structured logging.

    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional system instructions and reference context.
    :param model: Target model identifier.
    :param max_tokens: Maximum output token budget.
    :param temperature: Sampling temperature for non-reasoning models.
    :param effort: Reasoning effort for reasoning models ('low', 'medium', 'high').

    :return: Standardized response dictionary containing estimation, model, provider, finish_reason, and usage.
    :raises LLMServiceError: If the API call fails after all retry attempts.
    """
    client = get_openai_client()
    try:
        response = call_with_retry(
            client=client,
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            effort=effort,
        )
    except Exception as e:
        log.error(
            "llm_provider_call_failed",
            provider="openai",
            model=model,
            error_type=type(e).__name__,
            error=repr(e),
            exc_info=e,
        )
        raise LLMServiceError(f"OpenAI call failed: {e}") from e

    if response is None:
        log.error("llm_provider_call_exhausted", provider="openai", model=model)
        raise LLMServiceError("OpenAI request failed after retry attempts.")

    if response.status != "completed":
        if response.status == "incomplete":
            detail = f"Incomplete response from OpenAI: {getattr(response, 'incomplete_details', None)}"
        elif response.status == "failed":
            detail = f"OpenAI request failed: {getattr(response, 'error', None)}"
        else:
            detail = f"OpenAI response status not completed: {response.status}"
        log.error("llm_response_not_completed", provider="openai", model=model, status=response.status, detail=detail)
        raise LLMServiceError(detail)

    finish_reason = "stop"  # Responses API 'completed' status maps to 'stop' for evaluation compatibility
    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

    log.info(
        "llm_response_received",
        provider="openai",
        model=response.model,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return {
        "estimation": response.output_text,
        "model": response.model,
        "provider": "openai",
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }


def _call_anthropic(
    messages: str | list[dict[str, Any]],
    system_prompt: str | None,
    model: str,
    max_tokens: int,
    thinking_budget: int | None = None,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """
    Send an estimation request to the Anthropic Messages API with structured logging and retry capability.

    :param messages: Single string prompt or multi-turn list of message objects.
    :param system_prompt: Optional system instructions and reference context.
    :param model: Target model identifier.
    :param max_tokens: Maximum output token budget.
    :param thinking_budget: Extended thinking token budget, or None to disable.
    :param temperature: Sampling temperature (ignored and unset when thinking_budget is configured).
    :return: Standardized response dictionary containing estimation, model, provider, finish_reason, and usage.
    :raises LLMServiceError: If the API call fails or encounters an unhandled provider error.
    """
    client = get_anthropic_client()

    formatted_messages = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if thinking_budget is not None:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        # Anthropic requires max_tokens > thinking_budget; pad with headroom for the answer.
        kwargs["max_tokens"] = max(max_tokens, thinking_budget + 1024)
    elif temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = client.messages.create(**kwargs)
    except Exception as e:
        log.error(
            "llm_provider_call_failed",
            provider="anthropic",
            model=model,
            error_type=type(e).__name__,
            error=repr(e),
            exc_info=e,
        )
        raise LLMServiceError(f"Anthropic call failed: {e}") from e

    if response.stop_reason == "max_tokens":
        log.warning("llm_response_truncated", provider="anthropic", model=model, stop_reason=response.stop_reason)

    finish_reason = response.stop_reason or "stop"

    # Extract the first text block (skipping any thinking blocks that precede it)
    text_block = next(
        (block for block in response.content if getattr(block, "type", None) == "text"),
        None,
    )
    estimation_text = getattr(text_block, "text", "") if text_block else ""

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens

    log.info(
        "llm_response_received",
        provider="anthropic",
        model=response.model,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return {
        "estimation": estimation_text,
        "model": response.model,
        "provider": "anthropic",
        "finish_reason": finish_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    }


# endregion Provider wrappers


def generate_estimation(transcription: str, opts: GenerationOptions | None = None) -> dict[str, str]:
    """
    Generate a software estimation from a meeting transcription using the configured LLM.

    :param transcription: Meeting transcription text to analyze.
    :param opts: Optional generation configuration overrides.

    :return: Dictionary containing estimation text, model, provider, finish reason, and token usage.
    :raises LLMServiceError: If an unrecoverable failure occurs, call the LLM provider.
    """
    opts = opts or GenerationOptions()
    settings = get_settings()

    t0 = time.perf_counter()

    prep_usage = {"input": 0, "output": 0}
    extracted_requirements: str | None = None
    user_input = transcription

    if opts.preprocessing == "two_phase":
        extracted_requirements, prep_usage = extract_requirements(transcription, opts)
        user_input = extracted_requirements

    system_prompt = build_system_prompt(
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        inline_cleaning=(opts.preprocessing == "inline_cleaning"),
    )

    model = opts.model or settings.llm_model
    model_spec = MODEL_CATALOG.get(model)
    provider = model_spec.provider if model_spec is not None else settings.llm_provider

    cache = get_estimation_cache()
    cache_key = EstimationCache.make_key(
        system_prompt=system_prompt,
        user_message=user_input,
        model=model,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
        preprocessing=opts.preprocessing,
    )

    cached_result = cache.get(cache_key)
    if cached_result is not None:
        cached_result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        cached_result["cache_hit"] = True
        log.info("estimation_served_from_cache", model=model, latency_ms=cached_result["latency_ms"])
        return cached_result

    log.info(
        "generating_estimation",
        provider=provider,
        model=model,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        max_tokens=opts.max_tokens,
        thinking_budget=opts.thinking_budget,
    )

    try:
        if provider == "openai":
            if opts.thinking_budget is not None:
                log.warning("thinking_budget_ignored_for_provider", provider="openai")
            result = _call_openai(
                messages=user_input,
                system_prompt=system_prompt,
                model=model,
                max_tokens=opts.max_tokens,
            )
        else:
            result = _call_anthropic(
                messages=user_input,
                system_prompt=system_prompt,
                model=model,
                max_tokens=opts.max_tokens,
                thinking_budget=opts.thinking_budget,
            )

    except LLMServiceError:
        raise

    except Exception as e:
        log.error("llm_call_failed", error=repr(e), provider=provider)
        raise LLMServiceError(f"LLM call failed: {e!r}") from e

    result["usage"]["preprocessing_input_tokens"] = prep_usage["input"]
    result["usage"]["preprocessing_output_tokens"] = prep_usage["output"]
    result["preprocessing"] = opts.preprocessing
    result["extracted_requirements"] = extracted_requirements
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["cache_hit"] = False

    cache.set(cache_key, result)

    return result
