# tests/services/test_evaluation.py

# --- Python Standard Library ---

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.schemas.estimations import StructureCheck
from aie_estimator.services.evaluation import _to_int, evaluate_estimation_structure


@pytest.mark.parametrize(
    ("raw_input", "expected_output"),
    [
        ("40", 40),
        ("1.200 EUR", 1200),
        ("  50h  ", 50),
        ("", None),
        ("no-digits-here", None),
    ],
    ids=["simple_digits", "formatted_currency", "whitespace_and_units", "empty_string", "non_numeric"],
)
def test_to_int_parsing(raw_input: str, expected_output: int | None) -> None:
    """
    GIVEN raw string values representing numbers or non-numeric tokens
    WHEN _to_int is invoked
    THEN isolated integers are parsed or None is returned.

    :param raw_input: Raw string representation.
    :param expected_output: Expected parsed integer or None.
    """
    assert _to_int(raw_input) == expected_output


def test_evaluate_estimation_structure_fully_conforming() -> None:
    """
    GIVEN a fully compliant markdown estimation text with matching totals and valid finish reason
    WHEN evaluate_estimation_structure is executed
    THEN all structural flags are True, score is 1.0, and issues list is empty.
    """
    conforming_text = (
        "## Cloud Notification Service Estimation\n\n"
        "| Task | Hours | Cost |\n"
        "|---|---|---|\n"
        "| Architecture & Setup | 20 | 1.000 EUR |\n"
        "| Integration & Testing | 20 | 1.000 EUR |\n\n"
        "Total hours: 40\n"
        "Total cost: 2.000 EUR\n\n"
        "Recommended Team: 2 Senior Backend Developers\n"
        "Estimated Duration: 3 weeks\n"
    )

    check = evaluate_estimation_structure(conforming_text, finish_reason="stop")

    assert isinstance(check, StructureCheck)
    assert check.has_title is True
    assert check.has_breakdown_table is True
    assert check.has_totals_section is True
    assert check.has_team_section is True
    assert check.has_duration_section is True
    assert check.declared_total_hours == 40
    assert check.sum_row_hours == 40
    assert check.hours_match is True
    assert check.declared_total_cost == 2000
    assert check.sum_row_cost == 2000
    assert check.cost_match is True
    assert check.finish_reason_ok is True
    assert check.score == 1.0
    assert check.issues == []


def test_evaluate_estimation_structure_alternative_end_turn_finish_reason() -> None:
    """
    GIVEN an Anthropic finish reason of end_turn
    WHEN evaluate_estimation_structure is executed
    THEN finish_reason_ok evaluates to True.
    """
    minimal_text = "## Title\n| Task | Hours | Cost |\n|---|---|---|\n| API | 10 | 500 |\nTotal hours: 10\nTotal cost: 500\nTeam: 1 Dev\nDuration: 1 week"
    check = evaluate_estimation_structure(minimal_text, finish_reason="end_turn")
    assert check.finish_reason_ok is True


def test_evaluate_estimation_structure_missing_all_elements_and_truncated() -> None:
    """
    GIVEN plain unstructured text and a truncated finish reason
    WHEN evaluate_estimation_structure is executed
    THEN all structural flags are False and issues identify every missing section.
    """
    check = evaluate_estimation_structure("Raw text output with no markdown formatting.", finish_reason="length")

    assert check.has_title is False
    assert check.has_breakdown_table is False
    assert check.has_totals_section is False
    assert check.has_team_section is False
    assert check.has_duration_section is False
    assert check.finish_reason_ok is False
    assert check.hours_match is None
    assert check.cost_match is None
    assert check.score == 0.0

    issue_text = " ".join(check.issues)
    assert "Missing H2 project title" in issue_text
    assert "Missing breakdown table" in issue_text
    assert "Missing totals section" in issue_text
    assert "Missing recommended team section" in issue_text
    assert "Missing estimated duration" in issue_text
    assert "Response truncated or unexpected finish_reason='length'" in issue_text


def test_evaluate_estimation_structure_totals_mismatch() -> None:
    """
    GIVEN a markdown estimation where declared totals diverge from table row sums
    WHEN evaluate_estimation_structure is executed
    THEN hours_match and cost_match are False and discrepancy issues are captured.
    """
    mismatched_text = (
        "## Mismatched Estimation\n\n"
        "| Task | Hours | Cost |\n"
        "|---|---|---|\n"
        "| Task Alpha | 10 | 500 EUR |\n"
        "| Task Beta | 10 | 500 EUR |\n\n"
        "Total hours: 50\n"
        "Total cost: 5.000 EUR\n\n"
        "- 1 Fullstack Engineer\n"
        "Duration: 4 weeks\n"
    )

    check = evaluate_estimation_structure(mismatched_text, finish_reason="stop")

    assert check.hours_match is False
    assert check.cost_match is False
    assert check.sum_row_hours == 20
    assert check.declared_total_hours == 50
    assert check.sum_row_cost == 1000
    assert check.declared_total_cost == 5000

    issue_text = " ".join(check.issues)
    assert "Total hours mismatch" in issue_text
    assert "Total cost mismatch" in issue_text


def test_evaluate_estimation_structure_table_unparseable_rows_and_header_skipping() -> None:
    """
    GIVEN a breakdown table containing separator rows, empty task names, and unparseable values
    WHEN evaluate_estimation_structure processes the rows
    THEN invalid rows are skipped without raising errors.
    """
    table_with_skipped_rows = (
        "## Title\n\n"
        "| Task | Hours | Cost |\n"
        "|:---|:---|:---|\n"
        "|  | 10 | 500 EUR |\n"
        "| Task | 10 | 500 EUR |\n"
        "| Valid Row | 10 | 500 EUR |\n"
        "| Partial None | 10 | TBD |\n"
        "| Unparseable | TBD | TBD |\n\n"
        "Total hours: 10\n"
        "Total cost: 500 EUR\n\n"
        "Recommended Team: 1 Engineer\n"
        "Duration: 1 week\n"
    )

    check = evaluate_estimation_structure(table_with_skipped_rows, finish_reason="stop")

    assert check.sum_row_hours == 10
    assert check.sum_row_cost == 500
    assert check.hours_match is True
    assert check.cost_match is True
