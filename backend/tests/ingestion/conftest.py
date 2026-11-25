# tests/ingestion/conftest.py

"""Shared deterministic stubs for offline ingestion unit tests."""

# --- Python Standard Library ---
from collections.abc import Callable

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.ingestion.pii.analyzer import PiiSpan


class StubPiiAnalyzer:
    """Deterministic substring-matching PiiAnalyzer stub avoiding spaCy load in unit tests."""

    def __init__(self, needles: dict[str, str]) -> None:
        """
        Initialize the stub with literal-to-entity-type mappings.

        :param needles: Mapping of exact plaintext substrings to canonical PiiFlag entity names.
        """
        self._needles = needles

    def analyze(self, text: str) -> list[PiiSpan]:
        """
        Return deterministic PiiSpan matches for all configured literals in ``text``.

        :param text: Input text to scan.

        :return: List of detected PiiSpan instances.
        """
        spans: list[PiiSpan] = []
        for literal, entity_type in self._needles.items():
            start = 0
            while True:
                idx = text.find(literal, start)
                if idx == -1:
                    break
                spans.append(PiiSpan(start=idx, end=idx + len(literal), entity_type=entity_type, score=0.95))
                start = idx + len(literal)
        return spans


@pytest.fixture
def make_stub_analyzer() -> Callable[[dict[str, str]], StubPiiAnalyzer]:
    """
    Provide a factory fixture constructing deterministic ``StubPiiAnalyzer`` instances.

    :return: Callable taking a literal-to-entity-type dict and returning a StubPiiAnalyzer.
    """
    return StubPiiAnalyzer
