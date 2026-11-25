# src/aie_estimator/ingestion/parsers/budget_json.py

"""
Parser and strict JSON contract for historical budget schemas.

Each budget is rendered to structured markdown for semantic prose retrieval while
retaining all 10 cleaned fields as plain JSON-serializable values in
``DocumentMetadata.extra``. Keeping plain dicts/lists in ``extra`` lets
``ConsistentPseudonymizer`` scrub nested string leaves in place before downstream
consumers validate the payload through ``CleanedBudget``.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
import json
from typing import Any, ClassVar

# --- Third Party Libraries ---
from pydantic import BaseModel, ConfigDict, Field

# --- App modules ---
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata
from aie_estimator.ingestion.loaders.filesystem import LoadedBlob
from aie_estimator.ingestion.parsers.protocol import ParseContext


class BudgetPhase(BaseModel):
    """Single delivery phase within a cleaned historical budget."""

    model_config = ConfigDict(extra="forbid")

    name: str
    weeks: int | float
    amount: float


class CleanedBudget(BaseModel):
    """Strict Pydantic v2 contract for cleaned, pseudonymized Session 6 budget JSON."""

    model_config = ConfigDict(extra="forbid")

    budget_id: str
    client_name: str
    client_code: str
    currency: str
    total_amount: float
    signed_at: str
    phases: list[BudgetPhase] = Field(default_factory=list)
    contact: str | None = None
    contact_email: str | None = None
    notes: str | None = None


class BudgetJsonParser:
    """Parses historical JSON budget files into structured records and Markdown Document objects."""

    supported_formats: ClassVar[set[str]] = {"json"}

    def decode_record(self, blob: LoadedBlob) -> dict[str, Any]:
        """
        Decode a raw JSON budget blob into a dictionary with source location provenance.

        :param blob: Raw JSON file blob.

        :return: Decoded JSON dictionary annotated with ``_source_location``.
        """
        payload: dict[str, Any] = json.loads(blob.bytes_.decode("utf-8"))
        payload["_source_location"] = blob.relative_path
        return payload

    def render_record(
        self,
        payload: dict[str, Any],
        *,
        source_location: str,
        context: ParseContext,
    ) -> Document:
        """
        Render a structured budget dictionary into a Markdown Document.

        :param payload: Decoded (and optionally cleaned) budget dictionary.
        :param source_location: Relative file path of the source blob.
        :param context: Run-scoped catalog and timestamp metadata.

        :return: Rendered budget Document.
        """
        text = _render_budget_markdown(payload)
        budget_id = payload.get("budget_id") or "unknown"
        # Deterministic ID stays unpseudonymized for incremental re-indexing; downstream
        # RAG consumers must read the pseudonymized budget_id from metadata.extra instead.
        doc_id = f"{context.source.name}:{budget_id}:{source_location}"
        return Document(
            id=doc_id,
            text=text,
            metadata=DocumentMetadata(
                source_name=context.source.name,
                source_version=context.source_version,
                source_location=source_location,
                ingested_at=context.ingested_at,
                # Store plain JSON-serializable dicts/lists (not Pydantic models) so
                # ConsistentPseudonymizer._pseudonymize_metadata_value() can recursively scrub
                # nested string leaves (including phases[].name) before CleanedBudget validation.
                extra={
                    "budget_id": payload.get("budget_id"),
                    "client_name": payload.get("client_name"),
                    "client_code": payload.get("client_code"),
                    "currency": payload.get("currency"),
                    "total_amount": payload.get("total_amount"),
                    "signed_at": payload.get("signed_at"),
                    "phases": list(payload.get("phases") or []),
                    "contact": payload.get("contact"),
                    "contact_email": payload.get("contact_email"),
                    "notes": payload.get("notes"),
                },
            ),
        )

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        """
        Decode a JSON budget payload and render it as structured Markdown.

        :param blob: Raw JSON file blob.
        :param context: Run-scoped catalog and timestamp metadata.

        :return: Single-element iterable containing the rendered budget Document.
        """
        payload = self.decode_record(blob)
        yield self.render_record(payload, source_location=blob.relative_path, context=context)


def _render_budget_markdown(payload: dict[str, Any]) -> str:
    """
    Render a budget JSON dictionary into semantic Markdown sections.

    :param payload: Parsed budget JSON object.

    :return: Structured Markdown string.
    """
    lines: list[str] = [
        f"# Budget {payload.get('budget_id', '?')}",
        "",
        "## Client",
        f"- Name: {payload.get('client_name', '-')}",
        f"- Internal code: {payload.get('client_code', '-')}",
    ]
    if payload.get("contact"):
        lines.append(f"- Contact: {payload['contact']}")
    if payload.get("contact_email"):
        lines.append(f"- Email: {payload['contact_email']}")

    lines.extend(
        [
            "",
            "## Total",
            f"- Amount: {payload.get('total_amount', '?')} {payload.get('currency', '?')}",
            f"- Signed: {payload.get('signed_at', '-')}",
            "",
            "## Phases",
        ]
    )

    phases = payload.get("phases") or []
    if not phases:
        lines.append("- (no phases declared)")
    else:
        for phase in phases:
            lines.append(
                f"- **{phase.get('name', '?')}** — "
                f"{phase.get('weeks', '?')} weeks · "
                f"{phase.get('amount', '?')} {payload.get('currency', '?')}"
            )

    if payload.get("notes"):
        lines.extend(["", "## Notes", str(payload["notes"])])

    return "\n".join(lines).rstrip() + "\n"
