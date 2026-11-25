# src/aie_estimator/ingestion/parsers/registry.py

"""
Tiny parser registry - maps a format string to a :class:`Parser` instance.

The registry is a simple dict in disguise. We keep it as a class to make the
"only one parser per format" invariant enforceable and to give the orchestrator
a single object to depend on.
"""

# --- App modules ---
from aie_estimator.ingestion.parsers.budget_json import BudgetJsonParser
from aie_estimator.ingestion.parsers.protocol import Parser
from aie_estimator.ingestion.parsers.transcript_txt import TranscriptTxtParser


class ParserRegistry:
    """Registry mapping each supported source format string to a single Parser instance."""

    def __init__(self) -> None:
        """Initialize an empty format-to-parser registry."""
        self._by_format: dict[str, Parser] = {}

    def register(self, parser: Parser) -> None:
        """
        Register a parser for each format declared in ``parser.supported_formats``.

        :param parser: Parser implementation satisfying the Parser protocol.

        :raises ValueError: If any declared format already has a registered parser.
        """
        for fmt in parser.supported_formats:
            if fmt in self._by_format:
                existing = type(self._by_format[fmt]).__name__
                raise ValueError(f"Format '{fmt}' already has a registered parser ({existing}).")
            self._by_format[fmt] = parser

    def get(self, fmt: str) -> Parser:
        """
        Retrieve the parser registered for ``fmt``.

        :param fmt: Source format identifier (for example ``json`` or ``txt``).

        :return: Registered Parser instance.
        :raises KeyError: If no parser is registered for ``fmt``.
        """
        if fmt not in self._by_format:
            raise KeyError(f"No parser registered for format '{fmt}'.")
        return self._by_format[fmt]


def default_registry() -> ParserRegistry:
    """
    Build a ParserRegistry preloaded with the built-in ``json`` and ``txt`` parsers.

    :return: Configured ParserRegistry instance.
    """
    registry = ParserRegistry()
    registry.register(BudgetJsonParser())
    registry.register(TranscriptTxtParser())
    return registry
