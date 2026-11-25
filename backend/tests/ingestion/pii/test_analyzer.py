# tests/ingestion/pii/test_analyzer.py

"""Unit tests for Presidio domain recognizers, overlap resolution, and PresidioPiiAnalyzer."""

# --- Python Standard Library ---
from types import SimpleNamespace
from unittest.mock import MagicMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.pii.analyzer import (
    PiiAnalyzer,
    PiiSpan,
    PresidioPiiAnalyzer,
    resolve_overlapping_spans,
)


def test_resolve_overlapping_spans_prefers_higher_score_then_longer() -> None:
    """
    GIVEN overlapping PiiSpan detections with differing scores and lengths
    WHEN resolve_overlapping_spans is invoked
    THEN the higher-confidence or longer non-overlapping spans are retained in ascending order.
    """
    spans = [
        PiiSpan(start=0, end=10, entity_type=PiiFlag.NUMERIC_IDENTIFIER.value, score=0.6),
        PiiSpan(start=0, end=16, entity_type=PiiFlag.BUDGET_ID.value, score=0.95),
        PiiSpan(start=20, end=28, entity_type=PiiFlag.PERSON.value, score=0.85),
        PiiSpan(start=20, end=32, entity_type=PiiFlag.ORGANIZATION.value, score=0.85),
    ]

    resolved = resolve_overlapping_spans(spans)

    assert [(s.start, s.end, s.entity_type) for s in resolved] == [
        (0, 16, PiiFlag.BUDGET_ID.value),
        (20, 32, PiiFlag.ORGANIZATION.value),
    ]


def test_presidio_analyzer_detects_builtin_and_domain_entities() -> None:
    """
    GIVEN English text containing a person name, email, budget ID, client code, and known client organization
    WHEN PresidioPiiAnalyzer.analyze is invoked with en_core_web_sm
    THEN all domain and built-in entities are detected above threshold without overlap.
    """
    analyzer = PresidioPiiAnalyzer(spacy_model="en_core_web_sm", language="en", score_threshold=0.7)
    assert isinstance(analyzer, PiiAnalyzer)

    text = "Laura Miller at Acme Spain Ltd. confirmed BUDGET-2024-0001 (client code CLI-0042) via laura.miller@acme.com."
    spans = analyzer.analyze(text)
    detected_types = {span.entity_type for span in spans}

    assert PiiFlag.PERSON.value in detected_types
    assert PiiFlag.ORGANIZATION.value in detected_types
    assert PiiFlag.BUDGET_ID.value in detected_types
    assert PiiFlag.CLIENT_CODE.value in detected_types
    assert PiiFlag.EMAIL_ADDRESS.value in detected_types

    # Verify spans are strictly non-overlapping and sorted ascending.
    for prev, curr in zip(spans, spans[1:], strict=False):
        assert prev.end <= curr.start


def test_presidio_analyzer_handles_blank_text_and_delegates_to_engine() -> None:
    """
    GIVEN a mock Presidio engine returning detected spans
    WHEN PresidioPiiAnalyzer.analyze is called
    THEN empty/blank text short-circuits without calling the engine and valid spans are returned.
    """
    mock_engine = MagicMock()
    mock_engine.analyze.return_value = [
        SimpleNamespace(start=10, end=22, entity_type=PiiFlag.PERSON.value, score=0.90),
    ]
    analyzer = PresidioPiiAnalyzer(engine=mock_engine)

    assert analyzer.analyze("   ") == []
    mock_engine.analyze.assert_not_called()

    spans = analyzer.analyze("Sprint by Laura Miller")
    assert len(spans) == 1
    assert spans[0].start == 10
    assert spans[0].end == 22


@pytest.mark.parametrize(
    ("text", "expected_entity"),
    [
        ("Ref BUDGET-2025-1234 approved.", PiiFlag.BUDGET_ID.value),
        ("Client CLT-INT-A047 onboarded.", PiiFlag.CLIENT_CODE.value),
        ("Contract with BetaNorte Distribution signed.", PiiFlag.ORGANIZATION.value),
    ],
    ids=["budget_id", "client_code", "known_client_org"],
)
def test_presidio_analyzer_domain_recognizers(text: str, expected_entity: str) -> None:
    """
    GIVEN text containing a single domain-specific identifier or known client
    WHEN PresidioPiiAnalyzer.analyze is invoked
    THEN the corresponding custom recognizer detects the expected entity type.

    :param text: Sample input sentence.
    :param expected_entity: Expected canonical PiiFlag entity type string.
    """
    analyzer = PresidioPiiAnalyzer(spacy_model="en_core_web_sm", language="en", score_threshold=0.7)
    spans = analyzer.analyze(text)
    assert any(span.entity_type == expected_entity for span in spans)
