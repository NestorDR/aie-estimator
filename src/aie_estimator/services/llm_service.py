# src/aie_estimator/services/llm_service.py

# --- Python Standard Library ---
import logging
import random
import time
from typing import Any, Final

# --- Third Party Libraries ---
from fastapi import HTTPException, status
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# --- App modules ---
from aie_estimator.config import config
from aie_estimator.context.examples import ESTIMATION_EXAMPLES

logger: Final[logging.Logger] = logging.getLogger(__name__)

SYSTEM_INSTRUCTION_PREFIX: Final[str] = (
    "You are an expert software engineering estimator. Your responsibility is to analyze client meeting "
    "transcripts and generate realistic, well-structured effort estimations including task breakdowns, hour "
    "estimates, team composition, and overall project duration."
    "Below are historical reference estimations to guide your estimation format, detail level, and sizing:\n"
)


def build_system_prompt() -> str:
    """
    Assemble the CAG system prompt containing the estimator role instructions and few-shot context examples.

    :return: Formatted system prompt string with embedded historical estimations.
    """
    examples_text: list[str] = []
    for index, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        formatted_example = (
            f"--- Reference Example {index} ---\nMeeting Summary:\n{example['meeting_summary']}\n\nEstimation:\n{example['estimation']}\n"
        )
        examples_text.append(formatted_example)

    return f"{SYSTEM_INSTRUCTION_PREFIX}\n" + "\n".join(examples_text)


def call_with_retry(
    client: OpenAI,
    message: str,
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    effort: str = "medium",
    max_retries: int = 3,
) -> Any:
    """
    Execute a request against OpenAI Responses API with exponential backoff and jitter retry logic.

    :param client: Authenticated OpenAI client instance.
    :param message: The user transcript input string.
    :param system_prompt: Optional instructions and context to be injected to the model.
    :param model: Target model identifier.
    :param temperature: Randomness parameter for non-reasoning models.
    :param effort: Reasoning effort for reasoning-capable models.
    :param max_retries: Maximum number of attempts for transient failures.

    :return: The OpenAI response object or None if attempts are exhausted.
    :raises BadRequestError: If unrecoverable, a malformed request occurs.
    :raises RateLimitError: If rate limits persist after all retries.
    :raises APIConnectionError: If network connectivity fails after all retries.
    :raises InternalServerError: If upstream server errors persist after all retries.
    """
    is_reasoning_model = model.startswith(("o1", "o3")) or "-6-" in model

    request_params: dict[str, Any] = {
        "model": model,
        "input": message,
        "max_output_tokens": 1500,
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

        except (BadRequestError, RateLimitError, InternalServerError, APIConnectionError) as exc:
            if isinstance(exc, BadRequestError) and "temperature" in str(exc) and "temperature" in request_params:
                request_params.pop("temperature", None)

            if attempt >= max_retries - 1:
                raise

            jitter: float = random.SystemRandom().uniform(0, 1)
            wait: float = (2**attempt) + jitter
            logger.warning(f"Transient error from OpenAI API (attempt {attempt + 1}/{max_retries}). Retrying in {wait:.2f}s: {exc!r}")
            time.sleep(wait)

    return None


def estimate(transcription: str) -> dict[str, str]:
    """
    Generate a project estimation by calling the configured LLM provider using Responses API with CAG context.

    :param transcription: Meeting transcript text to estimate.

    :return: Dictionary containing estimation text, model identifier, and provider name.
    :raises HTTPException: If provider credentials are missing or an API call error occurs.
    """
    if config.llm_provider.lower() != "openai":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"LLM provider '{config.llm_provider}' is not supported yet. Only 'openai' is configured.",
        )

    if not config.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key is not configured in the application environment.",
        )

    system_prompt = build_system_prompt()
    user_message = f"Please estimate the software project discussed in this meeting transcript:\n\n{transcription}"

    try:
        client = OpenAI(api_key=config.openai_api_key.get_secret_value())
        logger.info(f"Dispatching estimation request to OpenAI Responses API using model: {config.llm_model}")

        response = call_with_retry(
            client=client,
            message=user_message,
            system_prompt=system_prompt,
            model=config.llm_model,
        )

        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="OpenAI request failed after retry attempts.",
            )

        if response.status != "completed":
            if response.status == "incomplete":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Incomplete response from OpenAI: {response.incomplete_details}",
                )
            if response.status == "failed":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"OpenAI request failed: {response.error}",
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"OpenAI response status not completed: {response.status}",
            )

        return {
            "estimation": response.output_text,
            "model": response.model,
            "provider": config.llm_provider,
        }

    except AuthenticationError as exc:
        logger.exception(f"OpenAI authentication failed: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OpenAI authentication failed. Check your API key.",
        ) from exc

    except RateLimitError as exc:
        logger.exception(f"OpenAI rate limit reached: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="OpenAI rate limit exceeded. Please retry later.",
        ) from exc

    except BadRequestError as exc:
        logger.exception(f"OpenAI malformed request: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OpenAI malformed request: {exc!s}",
        ) from exc

    except APIConnectionError as exc:
        logger.exception(f"OpenAI connection error: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to connect to OpenAI API service.",
        ) from exc

    except InternalServerError as exc:
        logger.exception(f"OpenAI internal error: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OpenAI internal server error. Please retry shortly.",
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"Unexpected error during estimation generation: {exc}", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Estimation generation error: {exc!s}",
        ) from exc
