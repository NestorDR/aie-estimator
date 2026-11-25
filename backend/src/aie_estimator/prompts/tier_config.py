# src/aie_estimator/prompts/tier_config.py

# --- Python Standard Library ---
from dataclasses import dataclass
from typing import Final

# --- App modules ---
from aie_estimator.schemas.estimations import (
    BaseTierEstimation,
    DeveloperEstimation,
    ExecutiveEstimation,
    PmEstimation,
)
from aie_estimator.sessions.tier_resolver import Tier


@dataclass(frozen=True)
class TierConfig:
    """Declarative mapping from a stakeholder tier to its prompt template and schema."""

    template: str
    schema: type[BaseTierEstimation]


TIER_CONFIG: Final[dict[Tier, TierConfig]] = {
    Tier.DEVELOPER: TierConfig(
        template="developer.j2",
        schema=DeveloperEstimation,
    ),
    Tier.PM: TierConfig(
        template="pm.j2",
        schema=PmEstimation,
    ),
    Tier.EXECUTIVE: TierConfig(
        template="executive.j2",
        schema=ExecutiveEstimation,
    ),
}

