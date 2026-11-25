# src/aie_estimator/domain/schemas/acb.py

"""
DTOs for the Actor-Critic-Boss audit trail.
"""

# --- Python Standard Library ---
from typing import Final, Literal, Self

# --- Third Party Libraries ---
from pydantic import BaseModel, Field, model_validator

# --- App modules ---
from aie_estimator.domain.schemas.estimations import EstimationResponse

_MAX_CRITIC_ISSUES: Final[int] = 12

CriticIssueCategory = Literal[
    "arithmetic_error",
    "missing_component",
    "metadata_contradiction",
    "scope_omission",
    "tier_mismatch",
    "unrealistic_estimate",
]

CriticIssueSeverity = Literal["critical", "major", "minor"]

CriticVerdict = Literal["accept", "needs_iteration", "reject"]

BossDecision = Literal["accept", "iterate", "synthesize"]


class CriticIssue(BaseModel):
    """Structured defect flagged by the Critic against a candidate estimation."""

    category: CriticIssueCategory
    severity: CriticIssueSeverity
    field_path: str = Field(
        min_length=1,
        max_length=120,
        description="Dotted or bracketed path to the estimation field under review.",
    )
    description: str = Field(
        min_length=5,
        max_length=500,
        description="Concise explanation of the defect found.",
    )
    suggested_fix: str | None = Field(
        default=None,
        max_length=300,
        description="Optional one-sentence hint for resolving the defect.",
    )


class CriticFeedback(BaseModel):
    """Structured Critic audit verdict consumed by the Boss orchestrator."""

    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(
        default_factory=list,
        max_length=_MAX_CRITIC_ISSUES,
        description="Typed list of defects identified during review.",
    )
    confidence_in_review: int = Field(
        ge=0,
        le=100,
        description="Critic confidence score (0-100) for this audit.",
    )

    @model_validator(mode="after")
    def iteration_requires_blocking_issue(self) -> Self:
        """
        Enforce that ``needs_iteration`` has at least one blocking issue and ``reject`` has at least one issue.

        :return: Validated ``CriticFeedback`` instance.
        :raises ValueError: If ``needs_iteration`` lacks a critical/major issue or ``reject`` has no issues.
        """
        if self.verdict == "needs_iteration":
            has_blocking = any(issue.severity in {"critical", "major"} for issue in self.issues)
            if not has_blocking:
                raise ValueError(
                    "verdict 'needs_iteration' requires at least one issue with severity in {critical, major}; "
                    "minor-only issues should accept."
                )
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue describing why.")
        return self


class ACBIteration(BaseModel):
    """Audit record for a single Actor-Critic round within a Boss run."""

    iteration: int = Field(ge=0, description="Zero-based iteration index.")
    decision_after: BossDecision
    critic_verdict: CriticVerdict
    critic_confidence: int = Field(ge=0, le=100)
    issue_summary: list[str] = Field(default_factory=list)


class BossTrace(BaseModel):
    """Full governance audit trail produced by the Boss orchestrator."""

    iterations: list[ACBIteration] = Field(default_factory=list)
    final_decision: BossDecision
    iterations_run: int = Field(ge=0)


class ACBResponse(EstimationResponse):
    """Response payload for Actor-Critic-Boss session estimations."""

    boss_trace: BossTrace = Field(..., description="Structured audit trail of Boss governance decisions.")
