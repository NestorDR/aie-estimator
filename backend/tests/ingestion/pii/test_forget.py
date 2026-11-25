# tests/ingestion/pii/test_forget.py

"""Unit tests for GDPR Article 17 forget_entity service and CLI entrypoint."""

# --- Python Standard Library ---
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
import pytest
import structlog.testing

# --- App modules ---
from aie_estimator.ingestion.catalog.models import PiiFlag
from aie_estimator.ingestion.pii.analyzer import PiiSpan
from aie_estimator.ingestion.pii.forget import forget_entity, main
from aie_estimator.ingestion.pii.mapping_store import InMemoryMappingStore
from aie_estimator.ingestion.pii.pseudonymizer import ConsistentPseudonymizer, compute_pii_hash


class _SinglePersonAnalyzer:
    def analyze(self, text: str) -> list[PiiSpan]:
        idx = text.find("Laura Miller")
        if idx == -1:
            return []
        return [PiiSpan(start=idx, end=idx + len("Laura Miller"), entity_type=PiiFlag.PERSON.value, score=0.95)]


@pytest.mark.anyio
async def test_forget_entity_deletes_mapping_invalidates_reversibility_and_logs_without_plaintext() -> None:
    """
    GIVEN a previously pseudonymized person stored in InMemoryMappingStore
    WHEN forget_entity is awaited for that person
    THEN the mapping is removed, a subsequent run generates a new pseudonym, and audit logs omit plaintext.
    """
    store = InMemoryMappingStore()
    salt = "gdpr-test-salt"
    pseudonymizer = ConsistentPseudonymizer(analyzer=_SinglePersonAnalyzer(), mapping_store=store, salt=salt)

    first = await pseudonymizer.pseudonymize_text("Contact Laura Miller.")
    expected_hash = compute_pii_hash("Laura Miller", salt)

    with structlog.testing.capture_logs() as captured:
        deleted = await forget_entity(PiiFlag.PERSON, "Laura Miller", salt=salt, store=store)
        missing = await forget_entity(PiiFlag.PERSON, "Laura Miller", salt=salt, store=store)

    assert deleted is True
    assert missing is False

    events = [entry["event"] for entry in captured]
    assert events == ["ingestion_pii_erasure_completed", "ingestion_pii_erasure_not_found"]
    for entry in captured:
        assert entry["entity_type"] == PiiFlag.PERSON.value
        assert entry["original_hash"] == expected_hash
        assert "Laura Miller" not in repr(entry)

    # Re-pseudonymizing after erasure creates a new mapping entry.
    second = await pseudonymizer.pseudonymize_text("Contact Laura Miller.")
    assert second.applied[0].original_hash == expected_hash
    assert second.applied[0].pseudonym != first.applied[0].pseudonym


@pytest.mark.anyio
async def test_forget_entity_rejects_empty_value_or_salt() -> None:
    """
    GIVEN an empty original_value or blank salt
    WHEN forget_entity is awaited
    THEN ValueError is raised.
    """
    store = InMemoryMappingStore()
    with pytest.raises(ValueError, match=r"must not be empty"):
        await forget_entity(PiiFlag.PERSON, "", salt="valid-salt", store=store)

    with pytest.raises(ValueError, match=r"non-empty"):
        await forget_entity(PiiFlag.PERSON, "Laura Miller", salt="", store=store)


def test_forget_cli_main_invokes_async_runner(capsys: pytest.CaptureFixture[str]) -> None:
    """
    GIVEN valid CLI arguments for entity type and value
    WHEN forget.main is invoked
    THEN _run_cli_forget runs with parsed PiiFlag and returns exit code 0.

    :param capsys: Pytest stdout/stderr capture fixture.
    """
    with patch("aie_estimator.ingestion.pii.forget._run_cli_forget", new=AsyncMock(return_value=True)) as mock_run:
        exit_code = main(["--entity-type", "PERSON", "--value", "Laura Miller"])

    assert exit_code == 0
    mock_run.assert_awaited_once_with(PiiFlag.PERSON, "Laura Miller")
    assert "erased" in capsys.readouterr().out
