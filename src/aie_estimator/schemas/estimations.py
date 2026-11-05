# src/aie_estimator/schemas/estimations.py

# --- Python Standard Library ---
from enum import Enum

# --- Third Party Libraries ---
from pydantic import BaseModel, Field, field_validator

# --- App modules ---
from aie_estimator.services.providers.generation_options import ProviderMode
from aie_estimator.services.providers.model_catalog import get_supported_models
from aie_estimator.settings import get_settings


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
        validate_default=True,
        # `json_schema_extra` informs FastAPI's OpenAPI schema generator, which Swagger UI (/docs) reads
        # to render an interactive <select> dropdown menu in the browser.
        json_schema_extra=lambda schema: schema.update(enum=get_supported_models()),
        description="Override the default LLM model for this request. Must be one of the catalog supported models.",
    )
    engine: ProviderMode = Field(
        default="litellm",
        description="Provider dispatch mode: 'litellm' for dynamic multi-provider routing, or 'native' for direct vendor SDK calls.",
    )

    @field_validator("model", mode="after")
    @classmethod
    def validate_model(cls, value: str | None) -> str:
        """Validate that the model is registered in the model catalog or logical profiles.

        :param value: Optional model identifier from request.

        :return: Validated model identifier, defaulting to settings.llm_model if None.
        :raises ValueError: If a model is not registered in the catalog.
        """
        if value is None or value in ("none", ""):
            return get_settings().llm_model
        supported = get_supported_models()
        if value not in supported:
            supported_str = ", ".join(supported)
            raise ValueError(f"Input should be one of: {supported_str}")
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

    text: str = Field(..., description="Generated software project effort estimation text.")
    prompt_version: str = Field(..., description="Prompt template version identifier.")
    model: str = Field(..., description="Model identifier used to produce the estimation.")
    provider: str = Field(..., description="LLM provider name used to produce the estimation.")
    cost_usd: float | None = Field(
        default=None,
        description="Calculated execution cost in USD reported by the routing engine.",
    )
    usage: TokenUsage
