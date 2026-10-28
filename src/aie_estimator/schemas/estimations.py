# src/aie_estimator/schemas/estimations.py

# --- Python Standard Library ---
from typing import Literal

# --- Third Party Libraries ---
from pydantic import BaseModel, Field, field_validator

# --- App modules ---
from aie_estimator.schemas.models import SUPPORTED_MODELS
from aie_estimator.settings import get_settings

PreprocessingMode = Literal["none", "inline_cleaning", "two_phase"]
ExampleFormat = Literal["markdown", "json", "narrative"]


class EstimationRequest(BaseModel):
    """Payload schema for software project estimation requests."""

    transcription: str = Field(
        ...,
        min_length=50,
        description="Meeting transcript text containing client requirements and project scope.",
        examples=[
            "In the meeting with the marketing team, the client explained that they need a landing page with a contact form..."],
    )
    # region Added in live session
    preprocessing: PreprocessingMode = Field(
        default="none",
        description="Input preprocessing strategy: none | inline_cleaning | two_phase",
    )
    example_format: ExampleFormat = Field(
        default="markdown",
        description="Format used to render the CAG examples in the system prompt",
    )
    num_examples: int = Field(
        default=3,
        ge=0,
        le=5,
        description="Number of canonical examples to inject (0..N where N=len(CANONICAL_EXAMPLES))",
    )
    use_examples: bool = Field(
        default=True,
        description="Toggle the CAG examples block on/off (overrides num_examples when False)",
    )
    model: str | None = Field(
        default=get_settings().llm_model,
        validate_default=True,
        # json_schema_extra={"enum": list(SUPPORTED_MODELS)},
        description="Override the default LLM model for this request. Must be one of the catalog supported models.",
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        le=16000,
        description="Maximum output tokens for the estimation call",
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        le=16000,
        description="Extended thinking budget (Anthropic only). Ignored for OpenAI.",
    )
    evaluate: bool = Field(
        default=True,
        description="Run the structural evaluation on the generated estimation",
    )
    # endregion Added in live session

    @field_validator("model", mode="after")
    @classmethod
    def validate_model(cls, value: str | None) -> str:
        """Validate that the model is registered in SUPPORTED_MODELS.

        :param value: Optional model identifier from request.

        :return: Validated model identifier, defaulting to settings.llm_model if None.
        :raises ValueError: If a model is not registered in SUPPORTED_MODELS.
        """
        if value is None or value in ("none", ""):
            return get_settings().llm_model
        if value not in SUPPORTED_MODELS:
            supported = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"Input should be one of: {supported}")
        return value


class TokenUsage(BaseModel):
    """Token consumption details from the LLM call(s)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    preprocessing_input_tokens: int = 0
    preprocessing_output_tokens: int = 0


class StructureCheck(BaseModel):
    """Level-1 structural evaluation of the generated estimation."""

    has_title: bool
    has_breakdown_table: bool
    has_totals_section: bool
    has_team_section: bool
    has_duration_section: bool
    declared_total_hours: int | None
    sum_row_hours: int | None
    hours_match: bool | None
    declared_total_cost: float | None
    sum_row_cost: float | None
    cost_match: bool | None
    finish_reason_ok: bool
    score: float
    issues: list[str]


class EstimationResponse(BaseModel):
    """Response schema containing the generated estimation and provider metadata."""

    estimation: str = Field(..., description="Formatted software project effort estimation breakdown.")
    model: str = Field(..., description="Model identifier used to produce the estimation.")
    provider: str = Field(..., description="LLM provider name used to produce the estimation.")

    # region Added in live session
    usage: TokenUsage
    finish_reason: str = Field(..., description="Stop reason reported by the provider")
    preprocessing: PreprocessingMode = "none"
    extracted_requirements: str | None = Field(
        default=None,
        description="Phase-1 output when preprocessing='two_phase'; null otherwise",
    )
    latency_ms: int = Field(..., description="Server-side total latency in milliseconds")
    validation: StructureCheck | None = None
