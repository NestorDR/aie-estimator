# src/aie_estimator/ingestion/catalog/models.py

# --- Python Standard Library ---
from datetime import datetime
from enum import Enum, StrEnum
from typing import Final, Literal

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_MIN_RAG_READY_SCORE: Final[int] = 3

AccessLevel = Literal["public", "internal", "confidential"]
SourceFormat = Literal["json", "txt", "xlsx", "docx", "pdf", "csv"]


class CatalogDecision(str, Enum):
    """Explicit governance decision controlling whether a source enters ingestion."""

    INCLUDE = "include"
    REVIEW = "review"
    EXCLUDE = "exclude"


class PiiFlag(StrEnum):
    """Closed vocabulary of PII entity categories declared on catalog sources."""

    EMAIL = "EMAIL"
    IBAN = "IBAN"
    CREDIT_CARD = "CREDIT_CARD"
    NUMERIC_IDENTIFIER = "NUMERIC_IDENTIFIER"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    CLIENT_CODE = "CLIENT_CODE"
    BUDGET_ID = "BUDGET_ID"
    PHONE_NUMBER = "PHONE_NUMBER"


class QualityScore(BaseModel):
    """Per-dimension quality assessment on a 1..5 scale."""

    model_config = ConfigDict(extra="forbid")

    completeness: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    actuality: int = Field(ge=1, le=5)
    reliability: int = Field(ge=1, le=5)

    @property
    def is_rag_ready(self) -> bool:
        """
        Return True when no quality dimension falls below the acceptable threshold.

        :return: True if all four dimensions are at least 3, False otherwise.
        """
        return all(score >= _MIN_RAG_READY_SCORE for score in (self.completeness, self.consistency, self.actuality, self.reliability))


class Sensitivity(BaseModel):
    """PII presence, entity categories, and access classification for a source."""

    model_config = ConfigDict(extra="forbid")

    has_pii: bool
    pii_flags: tuple[PiiFlag, ...] = ()
    access_level: AccessLevel = "internal"


class CatalogSource(BaseModel):
    """Single audited data source entry in the versioned catalog."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    location: str
    owners: tuple[str, ...] = ()
    format: SourceFormat
    volume_estimate: str = ""
    refresh_declared: str = ""
    refresh_observed: str = ""
    quality: QualityScore
    sensitivity: Sensitivity
    lineage: tuple[str, ...] = ()
    decision: CatalogDecision
    decision_reason: str = ""
    last_audited: datetime | None = None

    @field_validator("name")
    @classmethod
    def _validate_snake_case_name(cls, value: str) -> str:
        """
        Enforce lowercase snake_case source identifiers.

        :param value: Candidate source name.

        :return: Validated source name.
        :raises ValueError: If the name is empty or contains non-snake_case characters.
        """
        if not value or not all(char.islower() or char.isdigit() or char == "_" for char in value):
            raise ValueError("CatalogSource.name must be non-empty lowercase snake_case")
        return value

    @model_validator(mode="after")
    def _validate_decision_reason(self) -> CatalogSource:
        """
        Require a non-empty decision_reason whenever decision is not include.

        :return: Validated CatalogSource instance.
        :raises ValueError: If decision is review or exclude without decision_reason.
        """
        if self.decision is not CatalogDecision.INCLUDE and not self.decision_reason.strip():
            raise ValueError(f"decision={self.decision.value} requires a non-empty decision_reason")
        return self


class DataCatalog(BaseModel):
    """Versioned collection of audited data sources."""

    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    sources: tuple[CatalogSource, ...]

    @field_validator("sources")
    @classmethod
    def _validate_unique_names(cls, sources: tuple[CatalogSource, ...]) -> tuple[CatalogSource, ...]:
        """
        Ensure every source in the catalog has a unique name.

        :param sources: Tuple of parsed catalog sources.

        :return: Validated tuple of catalog sources.
        :raises ValueError: If duplicate source names are detected.
        """
        seen: set[str] = set()
        for source in sources:
            if source.name in seen:
                raise ValueError(f"Duplicate source name in catalog: {source.name}")
            seen.add(source.name)
        return sources

    def included_sources(self) -> list[CatalogSource]:
        """
        Return all sources marked with decision=include.

        :return: List of approved CatalogSource entries.
        """
        return [source for source in self.sources if source.decision is CatalogDecision.INCLUDE]

    def find(self, name: str) -> CatalogSource | None:
        """
        Look up a source by its unique logical name.

        :param name: Logical source identifier.

        :return: Matching CatalogSource, or None if not found.
        """
        for source in self.sources:
            if source.name == name:
                return source
        return None
