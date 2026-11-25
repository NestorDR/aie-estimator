# src/aie_estimator/ingestion/pii/analyzer.py

"""
Pluggable PII analyzer protocol and Presidio + spaCy implementation.

Isolates ``presidio-analyzer`` and ``spaCy`` model loading behind a minimal ``PiiAnalyzer``
protocol so unit tests can inject deterministic stubs without paying model initialization
cost, while production runs share a cached ``AnalyzerEngine`` with deterministic overlap
resolution and confidence-threshold filtering.
"""

# --- Python Standard Library ---
from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Protocol, runtime_checkable

# --- Third Party Libraries ---
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.pii.recognizers import build_domain_recognizers

# Presidio-recognized entity categories (NUMERIC_IDENTIFIER is an online guardrail-only pattern).
DEFAULT_ANALYZER_ENTITIES: Final[tuple[str, ...]] = tuple(flag.value for flag in PiiFlag if flag is not PiiFlag.NUMERIC_IDENTIFIER)


@dataclass(frozen=True, slots=True)
class PiiSpan:
    """Immutable character span representing a single detected PII entity."""

    start: int
    end: int
    entity_type: str
    score: float


@runtime_checkable
class PiiAnalyzer(Protocol):
    """Contract for synchronous PII entity detection over plain text."""

    def analyze(self, text: str) -> list[PiiSpan]:
        """
        Detect non-overlapping PII entity spans within ``text``.

        :param text: Input text to inspect.

        :return: Non-overlapping PiiSpan list ordered by ascending start offset.
        """
        ...  # pragma: no cover


def resolve_overlapping_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """
    Resolve overlapping spans deterministically by preferring higher confidence, then longer length.

    :param spans: Raw detected PII spans that may overlap.

    :return: Non-overlapping spans sorted by ascending ``start`` offset.
    """
    # Rank higher-confidence and longer spans first so specific domain recognizers beat generic matches.
    ranked = sorted(spans, key=lambda span: (-span.score, -(span.end - span.start), span.start))
    selected: list[PiiSpan] = []
    for candidate in ranked:
        # Half-open [start, end) intersection check: adjacent spans (end == start) are allowed.
        if any(not (candidate.end <= kept.start or candidate.start >= kept.end) for kept in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda span: span.start)
    return selected


# Keep a single loaded spaCy pipeline warm in memory (~300 ms load saved per run) without unbounded RAM growth.
@lru_cache(maxsize=1)
def _build_presidio_engine(spacy_model: str, language: str) -> AnalyzerEngine:
    """
    Build and cache a Presidio ``AnalyzerEngine`` wired to ``spacy_model`` and domain recognizers.

    :param spacy_model: Installed spaCy model package name (e.g., ``en_core_web_sm``).
    :param language: ISO 639-1 language code for analysis (e.g., ``en``).

    :return: Configured Presidio AnalyzerEngine singleton for ``(spacy_model, language)``.
    """
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": spacy_model}],
        }
    )
    # Pass supported_languages explicitly so Presidio does not attempt to load its default ~382 MB en_core_web_lg model.
    engine = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=[language],
    )
    for recognizer in build_domain_recognizers(supported_language=language):
        engine.registry.add_recognizer(recognizer)
    return engine


class PresidioPiiAnalyzer:
    """``PiiAnalyzer`` implementation backed by Presidio, spaCy, and domain recognizers."""

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_sm",
        language: str = "en",
        score_threshold: float = 0.7,
        engine: AnalyzerEngine | None = None,
    ) -> None:
        """
        Initialize the Presidio-backed PII analyzer.

        :param spacy_model: Installed spaCy model package name.
        :param language: ISO 639-1 language code.
        :param score_threshold: Minimum confidence score required to retain a detection.
        :param engine: Optional pre-built AnalyzerEngine (useful for dependency injection/tests).
        """
        self._engine = engine if engine is not None else _build_presidio_engine(spacy_model, language)
        self._language = language
        self._score_threshold = score_threshold

    def analyze(self, text: str) -> list[PiiSpan]:
        """
        Analyze ``text`` for non-overlapping PII spans meeting ``score_threshold``.

        :param text: Input text to analyze.

        :return: Non-overlapping PiiSpan list sorted by ascending start offset.
        """
        if not text or not text.strip():
            return []

        raw_results = self._engine.analyze(
            text=text,
            language=self._language,
            entities=list(DEFAULT_ANALYZER_ENTITIES),
            score_threshold=self._score_threshold,
        )
        spans = [
            PiiSpan(
                start=item.start,
                end=item.end,
                entity_type=item.entity_type,
                score=float(item.score),
            )
            for item in raw_results
        ]
        return resolve_overlapping_spans(spans)
