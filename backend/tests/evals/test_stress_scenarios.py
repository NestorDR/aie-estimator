# tests/evals/test_stress_scenarios.py

"""Deterministic structural tests for synthetic CAG stress scenarios and attachment sweep helpers."""

# --- Python Standard Library ---
from dataclasses import FrozenInstanceError

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.domain.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from evals.stress.fixtures.build_pdfs import STRESS_PDF_SIZES_KB
from evals.stress.scenarios import (
    ATTACHMENT_SWEEP_TRANSCRIPT,
    CHECKPOINT_TURNS,
    Scenario,
    ScenarioTurn,
    all_scenarios,
    attachment_canary_facts,
    get_scenario,
)


def test_all_scenarios_returns_three_profiles_in_canonical_order() -> None:
    """GIVEN the stress scenario registry WHEN all_scenarios is called THEN growing, pivot, contradiction are returned in order."""
    scenarios = all_scenarios()

    assert [scenario.name for scenario in scenarios] == ["growing", "pivot", "contradiction"]
    assert CHECKPOINT_TURNS == (1, 3, 6, 10, 20)
    assert ATTACHMENT_SWEEP_TRANSCRIPT.strip() != ""


@pytest.mark.parametrize(
    ("name", "expected_anchor", "expected_turn_count"),
    [
        ("growing", "Nimbus", 20),
        ("pivot", "Helios", 20),
        ("contradiction", "Atlas", 20),
    ],
)
def test_get_scenario_returns_valid_20_turn_profile(
    name: str,
    expected_anchor: str,
    expected_turn_count: int,
) -> None:
    """GIVEN a valid scenario name WHEN get_scenario is called THEN it returns 20 non-empty turns with a turn-1 project_name anchor."""
    scenario = get_scenario(name)

    assert len(scenario.turns) == expected_turn_count
    assert all(turn.transcript.strip() != "" for turn in scenario.turns)
    assert scenario.turns[0].fact_introduced == expected_anchor
    assert scenario.turns[0].fact_field == "project_name"


def test_pivot_scenario_scopes_flutter_replacement_to_technologies() -> None:
    """GIVEN the pivot scenario WHEN inspecting stack-change turns THEN Flutter assertions target fact_field='technologies'."""
    pivot = get_scenario("pivot")
    flutter_turns = [turn for turn in pivot.turns if turn.fact_introduced == "Flutter"]

    assert len(flutter_turns) >= 1
    assert all(turn.fact_field == "technologies" for turn in flutter_turns)


def test_contradiction_scenario_updates_budget_from_30_to_80() -> None:
    """GIVEN the contradiction scenario WHEN inspecting budget turns THEN 30 precedes 80."""
    contradiction = get_scenario("contradiction")
    facts = [turn.fact_introduced for turn in contradiction.turns]

    assert facts.index("30") < facts.index("80")


def test_get_scenario_rejects_unknown_name() -> None:
    """GIVEN an unregistered scenario name WHEN get_scenario is called THEN ValueError is raised."""
    with pytest.raises(ValueError, match=r"Unknown scenario"):
        get_scenario("nonexistent")


def test_scenario_rejects_fewer_than_20_turns() -> None:
    """GIVEN fewer than 20 turns WHEN Scenario is instantiated THEN ValueError is raised."""
    short_turns = (ScenarioTurn("Initial project scope.", fact_introduced="Demo", fact_field="project_name"),)

    with pytest.raises(ValueError, match=r"at least 20 turns"):
        Scenario(
            name="growing",
            project_type=ProjectType.WEB_SAAS,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
            turns=short_turns,
        )


def test_scenario_and_turn_are_immutable() -> None:
    """GIVEN a Scenario and ScenarioTurn WHEN mutating an attribute THEN FrozenInstanceError is raised."""
    scenario = get_scenario("growing")

    with pytest.raises(FrozenInstanceError):
        scenario.name = "pivot"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        scenario.turns[0].transcript = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("size_kb", "expected_count"),
    [(0, 0), *((size_kb, 3) for size_kb in STRESS_PDF_SIZES_KB)],
)
def test_attachment_canary_facts_by_size(size_kb: int, expected_count: int) -> None:
    """GIVEN a calibrated attachment size WHEN attachment_canary_facts is called THEN 0 returns empty and positive sizes return 3 canaries."""
    facts = attachment_canary_facts(size_kb)

    assert len(facts) == expected_count
    if expected_count > 0:
        assert all(f"_{size_kb}KB:" in fact for fact in facts)


def test_attachment_canary_facts_rejects_negative_size() -> None:
    """GIVEN a negative size WHEN attachment_canary_facts is called THEN ValueError is raised."""
    with pytest.raises(ValueError, match=r"non-negative"):
        attachment_canary_facts(-5)
