# tests/context/test_examples.py

# --- Python Standard Library ---
import json
from typing import cast

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.context.examples import CANONICAL_EXAMPLES, ExampleFormat, format_examples_for_prompt


def test_format_examples_empty_list_returns_empty_string() -> None:
    """
    GIVEN an empty list of canonical examples
    WHEN format_examples_for_prompt is invoked with any format
    THEN an empty string is returned.
    """
    assert format_examples_for_prompt([], fmt="markdown") == ""
    assert format_examples_for_prompt([], fmt="json") == ""
    assert format_examples_for_prompt([], fmt="narrative") == ""


def test_format_examples_markdown() -> None:
    """
    GIVEN canonical examples and format="markdown"
    WHEN format_examples_for_prompt is executed
    THEN serialized example headers and meeting summaries are present in markdown.
    """
    result = format_examples_for_prompt(CANONICAL_EXAMPLES[:2], fmt="markdown")
    assert "--- EXAMPLE 1 ---" in result
    assert "Meeting Summary:" in result
    assert "Estimation:" in result


def test_format_examples_json() -> None:
    """
    GIVEN canonical examples and format="json"
    WHEN format_examples_for_prompt is executed
    THEN valid JSON containing meeting summaries and totals is produced.
    """
    result = format_examples_for_prompt(CANONICAL_EXAMPLES[:2], fmt="json")
    assert result.startswith("Reference examples (JSON):\n")
    json_part = result.replace("Reference examples (JSON):\n", "")
    parsed = json.loads(json_part)
    assert len(parsed) == 2
    assert "meeting_summary" in parsed[0]
    assert "totals" in parsed[0]


def test_format_examples_narrative() -> None:
    """
    GIVEN canonical examples and format="narrative"
    WHEN format_examples_for_prompt is executed
    THEN narrative prose describing past engagements is generated.
    """
    result = format_examples_for_prompt(CANONICAL_EXAMPLES[:2], fmt="narrative")
    assert "In a previous engagement (#1)" in result
    assert "In a previous engagement (#2)" in result


def test_format_examples_unsupported_format_raises_value_error() -> None:
    """
    GIVEN an unrecognized format identifier
    WHEN format_examples_for_prompt is called
    THEN ValueError is raised.
    """
    with pytest.raises(ValueError, match=r"Unknown example format"):
        format_examples_for_prompt(CANONICAL_EXAMPLES[:1], fmt=cast(ExampleFormat, "unsupported_format"))
