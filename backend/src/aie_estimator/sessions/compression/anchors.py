# src/aie_estimator/sessions/compression/anchors.py

# --- Python Standard Library ---
from dataclasses import dataclass, field
import re
from typing import Final, Literal

# --- Third Party Libraries ---
from pydantic import BaseModel, Field
import structlog

# --- App modules ---
from aie_estimator.services.providers import GenerationOptions, call_llm
from aie_estimator.sessions.models import Message
from aie_estimator.utils.constants import DEFAULT_FAST_PROFILE

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

AnchorStrategy = Literal["heuristic", "llm"]

_ANCHOR_MAX_TOKENS: Final[int] = 500

_HEURISTIC_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("nda", re.compile(r"\b(nda|non[- ]?disclosure|under embargo|legal hold)\b", re.IGNORECASE)),
    ("signed_contract", re.compile(r"\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b", re.IGNORECASE)),
    ("scope_frozen", re.compile(r"\b(scope|backlog)\s+(is\s+)?(frozen|locked|final|fixed)\b", re.IGNORECASE)),
    ("budget_locked", re.compile(r"\b(budget|cap|ceiling)\s+(is\s+)?(locked|fixed|approved|capped)\s+at\b", re.IGNORECASE)),
    ("compliance", re.compile(r"\b(hipaa|gdpr|sox|pci[- ]?dss|fda|iso[- ]?27001)\b", re.IGNORECASE)),
    ("deadline_hard", re.compile(r"\bhard\s+deadline\b|\bmust\s+go\s+live\s+by\b", re.IGNORECASE)),
    ("contractual", re.compile(r"\bcontractually\s+(bound|required|obliged)\b", re.IGNORECASE)),
    ("explicit_commitment", re.compile(r"\b(we|the (client|customer))\s+(agreed|committed)\s+to\b", re.IGNORECASE)),
)

_LLM_ANCHOR_SYSTEM_PROMPT: Final[str] = (
    "You classify whether a single user message in a software estimation chat contains a durable "
    "commitment that must survive history compression verbatim (signed contract/SOW, NDA/embargo, "
    "frozen scope, locked budget cap, regulatory compliance mandate, or hard contractual deadline). "
    "Be conservative: ordinary feature requests or questions are NOT anchors."
)


@dataclass(frozen=True, slots=True)
class AnchorMatch:
    """Outcome of an anchor check including matched rule names and optional LLM telemetry."""

    is_anchor: bool
    matched_rules: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class _AnchorClassification(BaseModel):
    """Structured LLM binary classification for durable commitment detection."""

    is_anchor: bool = Field(
        description=(
            "True only if the turn contains a durable commitment, legal/compliance constraint, scope, "
            "budget, or deadline commitment that the conversation must not forget across turns."
        )
    )
    reason: str = Field(default="", max_length=200)  # Currently, it is used only to explore its value.


class AnchorDetector:
    """Detect durable commitment turns that should survive sliding-window compression verbatim."""

    def __init__(
        self,
        *,
        strategy: AnchorStrategy = "heuristic",
        model: str = DEFAULT_FAST_PROFILE,
    ) -> None:
        """
        Initialize the anchor detector with a heuristic or LLM classification strategy.

        :param strategy: ``"heuristic"`` for deterministic regex matching or ``"llm"`` for structured LLM classification.
        :param model: Logical profile or model identifier used when ``strategy == "llm"``.
        """
        self.strategy: Final[AnchorStrategy] = strategy
        self.model: Final[str] = model

    def _detect_heuristic(self, message: Message) -> AnchorMatch:
        """
        Evaluate compiled commitment regexes against ``message.content``.

        :param message: User message being evaluated for anchor promotion.

        :return: ``AnchorMatch`` with matched rule identifiers.
        """
        matched = [name for name, pattern in _HEURISTIC_PATTERNS if pattern.search(message.content)]
        return AnchorMatch(is_anchor=bool(matched), matched_rules=matched)

    async def detect(self, message: Message) -> AnchorMatch:
        """
        Determine whether ``message`` carries a durable commitment worth anchoring.

        :param message: User message peeled from the sliding window.

        :return: ``AnchorMatch`` indicating promotion status, rule labels, and any LLM usage.
        """
        if self.strategy == "heuristic":
            return self._detect_heuristic(message)

        try:
            response = await call_llm(
                user_prompt=f"Classify this user turn:\n\n{message.content}",
                system_prompt=_LLM_ANCHOR_SYSTEM_PROMPT,
                opts=GenerationOptions(model=self.model, max_tokens=_ANCHOR_MAX_TOKENS, reasoning_effort="low"),
                response_model=_AnchorClassification,
            )
            classified = _AnchorClassification.model_validate(response.get("result"))
        except Exception as e:  # noqa: BLE001
            log.warning(
                "history_anchor_detection_failed",
                fallback="heuristic",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            return self._detect_heuristic(message)

        usage = response.get("usage") or {}
        return AnchorMatch(
            is_anchor=classified.is_anchor,
            matched_rules=["llm_classifier"] if classified.is_anchor else [],
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=response.get("cost_usd"),
        )
