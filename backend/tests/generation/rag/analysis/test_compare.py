# tests/generation/rag/analysis/test_compare.py
"""
Offline unit tests for the standalone ``scripts/compare.py`` CLI.

Exercises argument parsing (``--1st`` / ``--2nd``), formatted ``stdout``
output, missing API key handling, and empty-text validation without network calls.
"""

# --- Python Standard Library ---
from unittest.mock import AsyncMock

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from scripts import compare


def test_compare_cli_prints_cosine_similarity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    GIVEN a configured embedder returning known orthogonal-free vectors
    WHEN scripts/compare.py main() is invoked with --1st and --2nd
    THEN it prints both texts and the four-decimal cosine similarity and returns 0.
    """
    fake_embedder = AsyncMock()
    fake_embedder.embed_one = AsyncMock(side_effect=[[1.0, 0.0], [1.0, 1.0]])
    monkeypatch.setattr(compare, "get_openai_client", lambda: object())
    monkeypatch.setattr(compare, "get_embedder", lambda _client: fake_embedder)

    exit_code = compare.main(["--1st", "First text", "--2nd", "Second text"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Text 1: First text" in captured.out
    assert "Text 2: Second text" in captured.out
    assert "Cosine similarity: 0.7071" in captured.out


def test_compare_cli_fails_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    GIVEN OPENAI_API_KEY is unset so get_embedder returns None
    WHEN scripts/compare.py main() is invoked
    THEN it writes an error to stderr, writes nothing to stdout, and returns 1.
    """
    monkeypatch.setattr(compare, "get_openai_client", lambda: None)
    monkeypatch.setattr(compare, "get_embedder", lambda _client: None)

    exit_code = compare.main(["--1st", "Hello", "--2nd", "World"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "OPENAI_API_KEY" in captured.err


def test_compare_cli_fails_on_blank_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    GIVEN an embedder that rejects blank text with ValueError
    WHEN scripts/compare.py main() is invoked with whitespace-only input
    THEN it writes the validation error to stderr and returns 1.
    """
    fake_embedder = AsyncMock()
    fake_embedder.embed_one = AsyncMock(side_effect=ValueError("Embedding input texts must not be empty."))
    monkeypatch.setattr(compare, "get_openai_client", lambda: object())
    monkeypatch.setattr(compare, "get_embedder", lambda _client: fake_embedder)

    exit_code = compare.main(["--1st", "   ", "--2nd", "World"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "must not be empty" in captured.err
