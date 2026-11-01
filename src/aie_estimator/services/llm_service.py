# src/aie_estimator/services/llm_service.py

# --- Python Standard Library ---
from dataclasses import dataclass, replace
import time
from typing import Any, Final

# --- Third Party Libraries ---
import structlog

# --- App modules ---
from aie_estimator.context.examples import format_examples_for_prompt, select_examples
from aie_estimator.schemas.estimations import (
    ExampleFormat,
    ExecutionEngine,
    PreprocessingMode,
    ReasoningEffort,
)
from aie_estimator.schemas.models import get_model_catalog
from aie_estimator.services import DEFAULT_MAX_TOKENS, LLMServiceError
from aie_estimator.services.cache import EstimationCache, get_estimation_cache
from aie_estimator.services.litellm_provider import call_litellm_router
from aie_estimator.services.native_providers import call_anthropic, call_openai
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

SYSTEM_INSTRUCTION_PREFIX: Final[str] = (
    "You are an expert software engineering estimator. Your responsibility is to analyze client meeting "
    "transcripts and generate realistic, well-structured effort estimations including task breakdowns, hour "
    "estimates, team composition, and overall project duration."
    "Below are historical reference estimations to guide your estimation format, detail level, and sizing:\n"
)

EXTRACTION_MAX_TOKENS: Final[int] = 1500

# region Prompt building blocks
# ---------------------------------------------------------------------------
# The two ACTIVE_OUTPUT_PROMPT variants live side by side so you
# can switch between them by editing the ACTIVE_OUTPUT_PROMPT assignment below.
# Uvicorn `--reload` picks up the change automatically.
# ---------------------------------------------------------------------------

PROMPT_OUTPUT_BASIC: Final[str] = "Generate an estimation for the project described above."

PROMPT_OUTPUT_STRUCTURED: Final[str] = """
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
ACTIVE_OUTPUT_PROMPT: Final[str] = PROMPT_OUTPUT_STRUCTURED

INLINE_CLEANING_BLOCK: Final[str] = """
The transcription you receive is from a real meeting and may contain:
- Informal small talk you must ignore
- Implicit requirements you must surface explicitly
- Contradictions where you must trust the most recent statement
- Non-technical jargon you must interpret

Extract ONLY the functional and technical requirements relevant to the estimation.
"""

# System prompt to instruct the LLM to extract structured data (e.g., JSON) from unstructured text.
EXTRACTION_SYSTEM_PROMPT: Final[str] = """
You are an analyst. Read the meeting transcription and produce a clean, deduplicated bullet list of 
functional requirements, non-functional requirements, integrations, constraints and explicit deadlines. 
Ignore fillers, divagations and off-topic remarks. Output Markdown only.
"""


# endregion Prompt building blocks


@dataclass
class GenerationOptions:
    """Request-specific parameters that determine the construction of the prompt and the call to the LLM."""

    preprocessing: PreprocessingMode = "none"
    example_format: ExampleFormat = "markdown"
    num_examples: int = 3
    use_examples: bool = True
    model: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning_effort: ReasoningEffort | None = None
    engine: ExecutionEngine = "litellm"

    @property
    def effective_model(self) -> str:
        """
        Return the requested model override or the default model from settings.

        :return: Resolved model or logical profile identifier.
        """
        return self.model or get_settings().llm_model

    @property
    def provider(self) -> str:
        """
        Resolve the upstream provider identifier for the effective model.

        :return: Provider identifier string.
        """
        spec = get_model_catalog().get(self.effective_model)
        return spec.provider if spec is not None else get_settings().llm_provider


async def _dispatch_llm_call(
    messages: str,
    system_prompt: str,
    opts: GenerationOptions,
) -> dict[str, Any]:
    """
    Dispatch an LLM completion request to the configured execution engine and provider.

    :param messages: User input prompt string.
    :param system_prompt: System prompt instructions.
    :param opts: Generation options specifying engine, model, provider, max_tokens, and reasoning_effort.

    :return: Standardized provider response dictionary.
    :raises LLMServiceError: If the provider is unsupported or the upstream call fails.
    """
    model = opts.effective_model
    provider = opts.provider
    effort = opts.reasoning_effort

    if opts.engine == "litellm":
        return await call_litellm_router(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=opts.max_tokens,
            reasoning_effort=effort,
        )
    if provider == "openai":
        return await call_openai(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=opts.max_tokens,
            effort=effort or "medium",
        )
    if provider == "anthropic":
        return await call_anthropic(
            messages=messages,
            system_prompt=system_prompt,
            model=model,
            max_tokens=opts.max_tokens,
            reasoning_effort=effort,
        )

    log.error("unsupported_provider_for_native_engine", provider=provider, model=model)
    raise LLMServiceError(f"Unsupported provider '{provider}' for legacy native engine.")


async def extract_requirements(
    transcription: str,
    opts: GenerationOptions,
) -> tuple[str, dict[str, int]]:
    """
    Extract clean, structured requirements from a raw meeting transcript using a focused extraction prompt.

    :param transcription: Raw meeting transcript text.
    :param opts: Generation options used for the call.

    :return: Tuple containing the extracted requirements string and a token usage dictionary.
    :raises LLMServiceError: If the extraction LLM call fails.
    """
    log.info("extracting_requirements", model=opts.effective_model, provider=opts.provider)

    extraction_opts = replace(opts, max_tokens=EXTRACTION_MAX_TOKENS, reasoning_effort=None)
    response = await _dispatch_llm_call(
        messages=transcription,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        opts=extraction_opts,
    )

    requirements = response["estimation"]
    usage = {
        "input": response["usage"]["input_tokens"],
        "output": response["usage"]["output_tokens"],
    }

    log.info(
        "requirements_extracted",
        extracted_length=len(requirements),
        input_tokens=usage["input"],
        output_tokens=usage["output"],
    )
    return requirements, usage


def build_system_prompt(
    example_format: ExampleFormat = "markdown",
    num_examples: int = 3,
    use_examples: bool = True,
    inline_cleaning: bool = False,
) -> str:
    """
    Assemble the full system prompt by combining base instructions, CAG reference examples, and optional inline cleaning guidance.

    :param example_format: Output format used to render the CAG reference examples.
    :param num_examples: Number of reference examples to include from the canonical bank.
    :param use_examples: Toggle whether to include the reference examples block.
    :param inline_cleaning: Toggle whether to append transcript-cleaning guidance.

    :return: Assembled system prompt string.
    """
    parts: list[str] = [SYSTEM_INSTRUCTION_PREFIX]

    if use_examples and num_examples > 0:
        examples = select_examples(num_examples)
        parts.append(format_examples_for_prompt(examples, fmt=example_format))

    if inline_cleaning:
        parts.append(INLINE_CLEANING_BLOCK)

    parts.append(ACTIVE_OUTPUT_PROMPT)
    return "\n\n".join(parts)


async def generate_estimation(
    transcription: str,
    opts: GenerationOptions | None = None,
) -> dict[str, Any]:
    """
    Coordinate the end-to-end software project estimation workflow.

    :param transcription: Meeting transcript text describing the client project.
    :param opts: Optional generation options controlling preprocessing, model selection, CAG examples, and token limits.

    :return: Response dictionary containing estimation text, model metadata, token usage, latency, and cache hit indicator.
    :raises LLMServiceError: If any LLM provider invocation encounters an error.
    """
    if opts is None:
        opts = GenerationOptions()

    t0 = time.perf_counter()
    extracted_requirements: str | None = None
    prep_usage: dict[str, int] = {"input": 0, "output": 0}

    user_input = transcription

    if opts.preprocessing == "two_phase":
        extracted_requirements, prep_usage = await extract_requirements(transcription, opts)
        user_input = extracted_requirements

    system_prompt = build_system_prompt(
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        inline_cleaning=(opts.preprocessing == "inline_cleaning"),
    )

    model = opts.effective_model
    provider = opts.provider
    effort = opts.reasoning_effort

    cache = get_estimation_cache()
    cache_key = EstimationCache.make_key(
        system_prompt=system_prompt,
        user_message=user_input,
        model=model,
        max_tokens=opts.max_tokens,
        reasoning_effort=effort,
        preprocessing=opts.preprocessing,
        engine=opts.engine,
    )

    cached_result = await cache.get(cache_key)
    if cached_result is not None:
        cached_result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        cached_result["cache_hit"] = True
        log.info("estimation_served_from_cache", model=model, latency_ms=cached_result["latency_ms"])
        return cached_result

    log.info(
        "generating_estimation",
        engine=opts.engine,
        provider=provider,
        model=model,
        preprocessing=opts.preprocessing,
        example_format=opts.example_format,
        num_examples=opts.num_examples,
        use_examples=opts.use_examples,
        max_tokens=opts.max_tokens,
        reasoning_effort=effort,
    )

    try:
        result = await _dispatch_llm_call(user_input, system_prompt, opts)

    except LLMServiceError:
        raise

    except Exception as e:
        log.exception("llm_call_failed", error=repr(e), provider=provider, engine=opts.engine)
        raise LLMServiceError(f"LLM call failed: {e!r}") from e

    result.setdefault("cost_usd", None)
    result["usage"]["preprocessing_input_tokens"] = prep_usage["input"]
    result["usage"]["preprocessing_output_tokens"] = prep_usage["output"]
    result["preprocessing"] = opts.preprocessing
    result["extracted_requirements"] = extracted_requirements
    result["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    result["cache_hit"] = False

    await cache.set(cache_key, result)

    return result
