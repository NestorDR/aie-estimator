# src/aie_estimator/schemas/estimations.py

# --- Python Standard Library ---
from enum import Enum
from typing import Final, Literal, Self

# --- Third Party Libraries ---
from pydantic import BaseModel, Field, model_validator

# --- App modules ---
from aie_estimator.model_catalog import ProviderMode, resolve_engine_model
from aie_estimator.sessions.tier_resolver import Tier

OUT_OF_SCOPE_PREFIX: Final[str] = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD: Final[int] = 30


class ProjectType(str, Enum):
    """Project domain categories with a basic profile."""

    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    """Level of detail for the breakdown of the estimate."""

    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    """Target presentation layout for the estimation request."""

    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    """Payload schema for software project estimation requests."""

    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Free-text description or transcription of the project to estimate.",
    )
    project_type: ProjectType = Field(description="Project domain categories with a basic profile.")
    detail_level: DetailLevel = Field(description="How deep the estimation should go.")
    output_format: OutputFormat = Field(description="Shape of the rendered estimation.")
    model: str | None = Field(
        default=None,
        description="Override the default LLM model for this request. Must be compatible with the selected engine.",
    )
    engine: ProviderMode = Field(
        default="litellm",
        description="Provider dispatch mode: 'litellm' for dynamic multi-provider routing, or 'native' for direct vendor SDK calls.",
    )
    tier: Tier | None = Field(
        default=None,
        description="Optional stakeholder tier override ('developer', 'pm', 'executive').",
    )

    @model_validator(mode="after")
    def validate_engine_model_compatibility(self) -> Self:
        """
        Resolve omitted model defaults and enforce engine/model compatibility.

        :return: Self with ``model`` resolved to a valid identifier for ``engine``.
        :raises ValueError: If ``model`` is incompatible with ``engine``.
        """
        self.model = resolve_engine_model(self.engine, self.model)
        return self


class TokenUsage(BaseModel):
    """Token consumption details from the generation LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class Phase(BaseModel):
    """A distinct delivery phase within a software project estimation."""

    name: str = Field(
        min_length=1,
        description="Phase title or identifier.",
    )
    duration_weeks: int = Field(
        ge=1,
        le=52,
        description="Phase duration in weeks.",
    )
    cost_eur: int = Field(
        ge=0,
        description="Estimated monetary cost for the phase in EUR.",
    )
    summary: str = Field(
        min_length=10,
        description="Concise scope and deliverable narrative for this phase.",
    )


class BaseTierEstimation(BaseModel):
    """Shared base contract for all tier-specific structured estimation results."""

    summary: str = Field(
        ...,
        description="Concise synthesis of project scope, risks, and architectural assumptions.",
    )
    confidence_pct: int = Field(
        ge=0,
        le=100,
        description="Confidence percentage score (0-100) reflecting transcript clarity and requirements completeness.",
    )

    @model_validator(mode="after")
    def low_confidence_requires_out_of_scope_prefix(self) -> Self:
        """Validate that estimations with low confidence (< 30) explicitly state that the request is out of scope.

        :return: Self instance if confidence thresholds and prefix constraints are met.
        :raises ValueError: If confidence is low but the summary omits the out-of-scope prefix.
        """
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(OUT_OF_SCOPE_PREFIX):
            raise ValueError(f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to start with '{OUT_OF_SCOPE_PREFIX}'.")
        return self


class HoursRange(BaseModel):
    """Minimum and maximum effort estimate in hours."""

    min_hours: int = Field(ge=0, description="Lower-bound effort estimate in hours.")
    max_hours: int = Field(ge=0, description="Upper-bound effort estimate in hours.")

    @model_validator(mode="after")
    def min_not_exceeds_max(self) -> Self:
        """Validate that ``min_hours`` does not exceed ``max_hours``."""
        if self.min_hours > self.max_hours:
            raise ValueError(f"min_hours ({self.min_hours}) cannot exceed max_hours ({self.max_hours}).")
        return self


class CostRange(BaseModel):
    """Minimum and maximum monetary cost estimate in EUR."""

    min_eur: int = Field(ge=0, description="Lower-bound cost estimate in EUR.")
    max_eur: int = Field(ge=0, description="Upper-bound cost estimate in EUR.")

    @model_validator(mode="after")
    def min_not_exceeds_max(self) -> Self:
        """Validate that ``min_eur`` does not exceed ``max_eur``."""
        if self.min_eur > self.max_eur:
            raise ValueError(f"min_eur ({self.min_eur}) cannot exceed max_eur ({self.max_eur}).")
        return self


class DurationRange(BaseModel):
    """Minimum and maximum delivery duration estimate in calendar weeks."""

    min_weeks: int = Field(ge=1, description="Lower-bound duration estimate in weeks.")
    max_weeks: int = Field(ge=1, description="Upper-bound duration estimate in weeks.")

    @model_validator(mode="after")
    def min_not_exceeds_max(self) -> Self:
        """Validate that ``min_weeks`` does not exceed ``max_weeks``."""
        if self.min_weeks > self.max_weeks:
            raise ValueError(f"min_weeks ({self.min_weeks}) cannot exceed max_weeks ({self.max_weeks}).")
        return self


class ComponentEstimation(BaseModel):
    """Technical component effort and risk breakdown for the developer tier."""

    name: str = Field(min_length=1, description="Technical component name (e.g., Backend API, Auth, Database).")
    estimated_hours: int = Field(ge=0, description="Estimated engineering hours for this component.")
    technical_risks: list[str] = Field(default_factory=list, description="Component-specific technical risks.")
    stack_assumptions: list[str] = Field(default_factory=list, description="Assumptions about stack or integrations.")


class DeveloperEstimation(BaseTierEstimation):
    """Structured estimation output tailored to software engineers and tech leads."""

    components: list[ComponentEstimation] = Field(..., min_length=1, description="Technical component breakdowns.")
    technical_risks: list[str] = Field(default_factory=list, description="Cross-cutting architectural or delivery risks.")
    stack_assumptions: list[str] = Field(default_factory=list, description="Key technology and integration assumptions.")
    uncertainty_drivers: list[str] = Field(
        default_factory=list,
        description="Scope areas where small decisions can shift effort by >20%.",
    )
    total_hours_range: HoursRange = Field(..., description="Aggregated engineering effort range in hours.")


class Milestone(BaseModel):
    """Visible delivery milestone for project management tracking."""

    name: str = Field(min_length=1, description="Milestone title.")
    target_week: int = Field(ge=1, description="Target completion week from project start.")
    deliverable: str = Field(min_length=1, description="Concrete artifact or outcome delivered.")


class TeamComposition(BaseModel):
    """Recommended delivery team staffing."""

    developers: int = Field(ge=0, description="Number of full-time developers.")
    designers: int = Field(ge=0, description="Number of designers.")
    qa_engineers: int = Field(ge=0, default=0, description="Number of QA engineers.")
    tech_lead: bool = Field(default=True, description="Whether a dedicated tech lead is included.")


class PmEstimation(BaseTierEstimation):
    """Structured estimation output tailored to project managers and delivery leads."""

    phases: list[Phase] = Field(..., min_length=1, description="Sequential project implementation phases.")
    milestones: list[Milestone] = Field(default_factory=list, description="Key delivery milestones.")
    team_composition: TeamComposition = Field(..., description="Recommended team staffing.")
    blockers: list[str] = Field(default_factory=list, description="Dependencies or decisions that could block delivery.")
    total_duration_weeks: int = Field(ge=0, description="Overall project duration in elapsed weeks, matching the sum of phases.")
    total_cost_eur: int = Field(ge=0, description="Total project expenditure in EUR, matching the exact sum of phase costs.")

    @model_validator(mode="after")
    def phases_sum_matches_total(self) -> Self:
        """Validate that total_duration_weeks and total_cost_eur equal the sum of individual phases.

        :return: Self instance if totals align with phase components.
        :raises ValueError: If cost or duration sums diverge from declared totals.
        """
        sum_weeks = sum(phase.duration_weeks for phase in self.phases)
        if sum_weeks != self.total_duration_weeks:
            raise ValueError(f"total_duration_weeks ({self.total_duration_weeks}) does not match the sum of phases ({sum_weeks}).")

        sum_cost = sum(phase.cost_eur for phase in self.phases)
        if sum_cost != self.total_cost_eur:
            raise ValueError(f"total_cost_eur ({self.total_cost_eur}) does not match the sum of phase costs ({sum_cost}).")
        return self


class ExecutiveEstimation(BaseTierEstimation):
    """Structured estimation output tailored to executive stakeholders and commercial leads."""

    headline_cost_range: CostRange = Field(..., description="High-level project budget range in EUR.")
    headline_duration_range: DurationRange = Field(..., description="High-level delivery timeline range in weeks.")
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Qualitative confidence rating.")
    top_three_risks: list[str] = Field(..., max_length=3, description="Up to three material business or delivery risks.")
    go_no_go_recommendation: str = Field(..., min_length=1, description="Actionable executive go/no-go guidance.")


TierEstimationResult = DeveloperEstimation | PmEstimation | ExecutiveEstimation


class EstimationResponse(BaseModel):
    """Response schema containing the generated structured estimation and execution metadata."""

    result: TierEstimationResult = Field(..., description="Tier-specific structured project estimation result.")
    tier: Tier = Field(..., description="Resolved stakeholder tier governing prompt and output schema.")
    tier_rule: str = Field(..., description="Identifier of the tier resolution rule that fired.")
    prompt_version: str = Field(..., description="Prompt template version identifier.")
    cached: bool = Field(default=False, description="Whether this response was served from the cache.")
    model: str = Field(..., description="Model identifier used to produce the estimation.")
    provider: str = Field(..., description="LLM provider name used to produce the estimation.")
    cost_usd: float | None = Field(
        default=None,
        description="Calculated execution cost in USD reported by the routing engine.",
    )
    usage: TokenUsage
