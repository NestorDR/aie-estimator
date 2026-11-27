# src/aie_estimator/foundation/guardrails/input.py

# --- Python Standard Library ---
from functools import lru_cache
import re
from typing import Any, Final, Literal

# --- Third Party Libraries ---
from openai import AsyncOpenAI
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings
from aie_estimator.utils.constants import (
    PII_CREDIT_CARD_RE,
    PII_EMAIL_RE,
    PII_IBAN_RE,
    PII_NUMERIC_IDENTIFIER_RE,
)

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

Reason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):  # noqa: N818
    """Raised when an input guardrail layer rejects a user-provided project description."""

    def __init__(self, message: str, *, reason: Reason) -> None:
        super().__init__(message)
        self.message: Final[str] = message
        self.reason: Final[Reason] = reason


# --- OpenAI Moderation Model -------------------------------------------------
_MODERATION_MODEL: Final[str] = "omni-moderation-latest"

# --- Prompt injection patterns -----------------------------------------------
# Precompiled regular expressions targeting common prompt injection, context wipe,
# and system-boundary escape attacks:
_PROMPT_INJECTION_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # Instruction overrides
    # Detects commands telling the LLM to ignore preceding guidance, prompts, or rules
    # (e.g., "ignore previous instructions", "ignore all rules").
    re.compile(r"ignore\s+(previous|prior|all|the)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    # System XML tag escapes
    # Detects fake tag boundaries used to spoof system prompt endings or openings
    # (e.g., "</system>", "<prompt>", "<instructions>").
    re.compile(r"</?\s*(system|instructions?|prompt)\s*>", re.IGNORECASE),
    # New instruction declarations
    # Detects phrases attempting to introduce authoritative instructions overriding system prompts
    # (e.g., "new instructions: do X instead").
    re.compile(r"new\s+instructions?\s*[:.\-]", re.IGNORECASE),
    # Context wipe directives
    # Detects memory wipe/clearing commands aimed at discarding system context
    # (e.g., "forget everything", "forget all previous").
    re.compile(r"forget\s+(everything|all|previous)", re.IGNORECASE),
    # Persona reassignment / jailbreak commands
    # Detects persona reassignment patterns that instruct the model to adopt a new, unconstrained persona
    # (e.g., "you are now a helpful uncensored assistant").
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    # Disregard directives with intervening words
    # Matches variants of "disregard" directing the model to drop rules or context with up to 40
    # intermediate characters (e.g., "disregard any prior system instructions or rules").
    re.compile(r"\bdisregard\b.{0,40}\b(instructions?|prompts?|rules?|context|previous|prior)", re.IGNORECASE | re.DOTALL),
    # System prompt exfiltration / leakage directives
    # Detects attempts to extract or print the hidden system prompt, initial instructions, or rules
    # (e.g., "repeat your system prompt", "reveal initial instructions").
    re.compile(
        r"\b(repeat|print|reveal|show|display|output)\b.{0,30}\b(system\s+prompt|initial\s+prompt|instructions?|rules?)\b",
        re.IGNORECASE,
    ),
    # Fake role delimiter / ChatML injections
    # Detects synthetic turn boundaries or role tags injected into text to simulate elevated system blocks
    # (e.g., "<|im_start|>system", "<|system|>", "[SYSTEM]", or "system:").
    re.compile(r"<\|im_start\|>|<\|system\|>|\[SYSTEM]|^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    # Explicit jailbreak modes
    # Catches explicit developer mode or jailbreak activation phrases
    # (e.g., "developer mode enabled", "jailbreak mode", "dan mode").
    re.compile(r"\b(developer\s+mode\s+enabled|jailbreak\s+mode|dan\s+mode)\b", re.IGNORECASE),
)

# --- PII Detection Patterns --------------------------------------------------

# Ordered from most specific to broadest (credit_card before numeric_identifier so 16-digit
# cards are never partially matched as 13-digit numeric identifiers).
_PII_RULES: Final[tuple[tuple[str, re.Pattern[str], str, str], ...]] = (
    ("email", PII_EMAIL_RE, "[REDACTED_EMAIL]", "Email address"),
    ("iban", PII_IBAN_RE, "[REDACTED_IBAN]", "IBAN"),
    ("credit_card", PII_CREDIT_CARD_RE, "[REDACTED_CREDIT_CARD]", "Credit card number"),
    ("numeric_identifier", PII_NUMERIC_IDENTIFIER_RE, "[REDACTED_NUMERIC_ID]", "Phone number or numeric identifier"),
)


@lru_cache(maxsize=1)
def get_moderation_client() -> AsyncOpenAI | None:
    """
    Retrieve or initialize the cached asynchronous OpenAI client instance for moderation.

    :return: Authenticated AsyncOpenAI client, or None if OPENAI_API_KEY is not configured.
    """
    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
    if not api_key:
        return None
    return AsyncOpenAI(api_key=api_key)


async def sanitize_input(
    description: str,
    *,
    openai_client: AsyncOpenAI | None = None,
) -> None:
    """
    Execute input guardrail layers in order and raise an exception on first violation.

    :param description: Project description string to validate.
    :param openai_client: Optional AsyncOpenAI client. When provided, runs content moderation.

    :raises InputGuardrailViolation: If moderation flags content, prompt injection is detected, or PII is found.
    """
    if openai_client is not None:
        await _check_moderation(description, openai_client)
    _check_prompt_injection(description)
    _check_pii(description)


async def sanitize_attachment_input(
    text: str,
    *,
    openai_client: AsyncOpenAI | None = None,
) -> str:
    """
    Validate extracted attachment text for moderation and prompt injection, then redact matched PII.

    :param text: Extracted attachment text to inspect.
    :param openai_client: Optional AsyncOpenAI client. When provided, runs content moderation.

    :return: Sanitized attachment text with matched PII replaced by typed placeholders.
    :raises InputGuardrailViolation: If moderation flags content or prompt injection is detected.
    """
    if not text:
        return text
    if openai_client is not None:
        await _check_moderation(text, openai_client)
    _check_prompt_injection(text)
    return _redact_pii(text)


async def _check_moderation(description: str, openai_client: AsyncOpenAI) -> None:
    """
    Evaluate content safety via the OpenAI Moderation API.

    :param description: Project description string to evaluate.

    :raises InputGuardrailViolation: If moderation flags the input text.
    """
    try:
        response = await openai_client.moderations.create(
            model=_MODERATION_MODEL,
            input=description,
        )
    except Exception as e:
        log.warning(
            "moderation_call_failed",
            error_type=type(e).__name__,
            error=str(e),
        )
        return

    results = getattr(response, "results", [])
    if not results:
        return

    result = results[0]
    if getattr(result, "flagged", False):
        categories = _extract_flagged_categories(result)
        log.info("moderation_flagged", categories=categories)
        flagged_str = ", ".join(categories) or "unspecified"
        raise InputGuardrailViolation(
            f"Input flagged by moderation: {flagged_str}",
            reason="moderation",
        )


def _extract_flagged_categories(result: Any) -> list[str]:
    """
    Extract the list of category names that were flagged.

    :param result: OpenAI moderation result item.

    :return: List of category strings flagged as True.
    """
    categories = getattr(result, "categories", None)
    if categories is None:
        return []
    data = categories.model_dump() if hasattr(categories, "model_dump") else categories.__dict__
    return [name for name, flagged in data.items() if flagged]


def _check_prompt_injection(description: str) -> None:
    """
    Scan description for prompt injection and instruction-override heuristics.

    :param description: Project description text to scan.

    :raises InputGuardrailViolation: If suspicious patterns are detected.
    """
    for pattern in _PROMPT_INJECTION_PATTERNS:
        match = pattern.search(description)
        if match:
            matched_text = match.group(0)[:80]
            log.info("prompt_injection_detected", pattern=pattern.pattern, match=matched_text)
            raise InputGuardrailViolation(
                f"Suspicious instruction-like text detected: {matched_text!r}",
                reason="prompt_injection",
            )


def _check_pii(description: str) -> None:
    """
    Scan description for personally identifiable information (emails, IBANs, credit cards, numeric IDs/phones).

    :param description: Project description text to scan.
    :raises InputGuardrailViolation: If PII is detected.
    """
    for pii_type, pattern, _, label in _PII_RULES:
        if pattern.search(description):
            log.info("pii_detected", pii_type=pii_type)
            raise InputGuardrailViolation(
                f"{label} detected in description — please remove personal data.",
                reason="pii",
            )


def _redact_pii(text: str) -> str:
    """
    Replace matched PII patterns in attachment text with typed placeholders and log redaction counts.

    :param text: Extracted attachment text to sanitize.

    :return: Text with matched PII replaced by ``[REDACTED_*]`` placeholders.
    """
    redacted_counts: dict[str, int] = {}
    sanitized = text
    for pii_type, pattern, placeholder, _ in _PII_RULES:
        sanitized, count = pattern.subn(placeholder, sanitized)
        if count > 0:
            redacted_counts[pii_type] = count
    if redacted_counts:
        log.info("attachment_pii_redacted", counts=redacted_counts)
    return sanitized
