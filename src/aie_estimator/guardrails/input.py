# src/aie_estimator/guardrails/input.py

# --- Python Standard Library ---
from functools import lru_cache
import re
from typing import Any, Final, Literal

# --- Third Party Libraries ---
from openai import AsyncOpenAI
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

Reason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):  # noqa: N818
    """Raised when an input guardrail layer rejects a user-provided project description."""

    def __init__(self, message: str, *, reason: Reason) -> None:
        super().__init__(message)
        self.message: Final[str] = message
        self.reason: Final[Reason] = reason


_MODERATION_MODEL: Final[str] = "omni-moderation-latest"

# --- Prompt injection patterns ----------------------------------------------
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
    re.compile(r"<\|im_start\|>|<\|system\|>|\[SYSTEM\]|^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    # Explicit jailbreak modes
    # Catches explicit developer mode or jailbreak activation phrases
    # (e.g., "developer mode enabled", "jailbreak mode", "dan mode").
    re.compile(r"\b(developer\s+mode\s+enabled|jailbreak\s+mode|dan\s+mode)\b", re.IGNORECASE),
)

# --- PII Detection Patterns --------------------------------------------------

# Email Addresses
# Checks standard RFC 5322-compliant email formats (local-part@domain.tld).
# Requires at least 2 alphabetic characters for the top-level domain to avoid
# matching code identifiers or object references (e.g. 'obj.property').
_EMAIL_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# International Bank Account Numbers (IBAN)
# Checks international bank account numbers across all SEPA and international formats.
# Pattern: Two uppercase ISO country code letters, two check digits, followed by
# 10 to 30 alphanumeric Basic Bank Account Number (BBAN) characters bounded by word boundaries.
_IBAN_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

# Numeric Identifiers and Telephone Numbers
# Checks international phone formats (prefixed with '+' and 1-3 country code digits) or
# 10-to-13-digit numeric identifiers with optional spaces, dots, or hyphens (covering national
# telephone numbers, 12-digit AWS account IDs, 13-digit epoch timestamps, and long numeric IDs).
# Guard against false positives: Requires at least 10 digits to avoid matching
# calendar dates (YYYY-MM-DD), shorter timestamps, IP addresses, or semantic versions (e.g., 1.16.1).
_NUMERIC_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d")

# Credit Card Numbers
# Checks major credit card formats:
# - Visa (starts with 4, 16 digits)
# - MasterCard (starts with 51-55 or 22-27, 16 digits)
# - American Express (starts with 34 or 37, 15 digits)
# - Discover (starts with 6011 or 65, 16 digits)
# - Generic 16-digit cards formatted in 4 groups of 4 digits separated by spaces or hyphens.
_CREDIT_CARD_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|2[2-7]\d{2}|6(?:011|5\d{2}))[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    r"|"
    r"\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b"
    r"|"
    r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"
)

# Ordered from most specific to broadest (credit_card before numeric_identifier so 16-digit
# cards are never partially matched as 13-digit numeric identifiers).
_PII_RULES: Final[tuple[tuple[str, re.Pattern[str], str, str], ...]] = (
    ("email", _EMAIL_RE, "[REDACTED_EMAIL]", "Email address"),
    ("iban", _IBAN_RE, "[REDACTED_IBAN]", "IBAN"),
    ("credit_card", _CREDIT_CARD_RE, "[REDACTED_CREDIT_CARD]", "Credit card number"),
    ("numeric_identifier", _NUMERIC_IDENTIFIER_RE, "[REDACTED_NUMERIC_ID]", "Phone number or numeric identifier"),
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


async def check_input(
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


async def check_attachment_input(
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
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "moderation_call_failed",
            error_type=type(exc).__name__,
            error=str(exc),
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
