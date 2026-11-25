# evals/dataset.py

# --- Python Standard Library ---
import json
from pathlib import Path
from typing import Final, Self

# --- Third Party Libraries ---
from pydantic import BaseModel, Field, model_validator
import structlog

# --- App modules ---
from aie_estimator.domain.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from aie_estimator.generation.conversation.tier_resolver import Tier

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

DEFAULT_DATASET_PATH: Final[Path] = Path(__file__).resolve().parent / "golden_dataset.json"


class GoldenCase(BaseModel):
    """Curated estimation scenario and tier-agnostic expectation envelope."""

    id: str = Field(min_length=3, max_length=64)
    description: str = Field(min_length=20, max_length=2000)
    project_type: ProjectType
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE
    tier: Tier | None = None

    expected_tier: Tier | None = None
    expected_out_of_scope: bool = False
    expected_hours_range: tuple[int, int] | None = None
    expected_cost_range_eur: tuple[int, int] | None = None
    expected_duration_weeks_range: tuple[int, int] | None = None
    expected_phase_count_range: tuple[int, int] | None = None
    required_components_any_of: list[str] = Field(default_factory=list)
    expected_technologies_any_of: list[str] = Field(default_factory=list)
    min_confidence_pct: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """
        Ensure all configured lower/upper bound tuples satisfy ``low <= high``.

        :return: Validated GoldenCase instance.
        :raises ValueError: If any range tuple has ``low > high`` or negative bounds.
        """
        for field_name in (
            "expected_hours_range",
            "expected_cost_range_eur",
            "expected_duration_weeks_range",
            "expected_phase_count_range",
        ):
            bounds = getattr(self, field_name)
            if bounds is not None:
                low, high = bounds
                min_allowed = 1 if field_name == "expected_phase_count_range" else 0
                if low < min_allowed or low > high:
                    raise ValueError(f"{field_name} ({low}, {high}) must satisfy {min_allowed} <= low <= high.")
        return self


def load_dataset(path: Path | str | None = None) -> list[GoldenCase]:
    """
    Load and validate the Golden Dataset from JSON.

    :param path: Optional path to a Golden Dataset JSON file; defaults to ``DEFAULT_DATASET_PATH``.

    :return: List of validated ``GoldenCase`` entries.
    """
    target = Path(path) if path is not None else DEFAULT_DATASET_PATH
    raw_entries = json.loads(target.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(entry) for entry in raw_entries]
    log.info("eval_dataset_loaded", path=str(target), case_count=len(cases))
    return cases
