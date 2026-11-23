# src/aie_estimator/ingestion/parsers/budget_json.py

"""
Parser for budget JSON schemas.

The key design choice: each budget is rendered to structured markdown, not
``json.dumps``. The downstream retriever embeds prose, not key-value soup;
markdown gives the embedder semantic anchors (## Client, ## Total, ## Phases)
that a raw JSON dump lacks. The numbers stay; only the framing changes.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
import json
from typing import Any, ClassVar

# --- App modules ---
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata
from aie_estimator.ingestion.loaders.filesystem import LoadedBlob
from aie_estimator.ingestion.parsers.protocol import ParseContext


class BudgetJsonParser:
    """Parses historical JSON budget files into structured Markdown Document objects."""

    supported_formats: ClassVar[set[str]] = {"json"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        """
        Decode a JSON budget payload and render it as structured Markdown.

        :param blob: Raw JSON file blob.
        :param context: Run-scoped catalog and timestamp metadata.

        :return: Single-element iterable containing the rendered budget Document.
        """
        payload: dict[str, Any] = json.loads(blob.bytes_.decode("utf-8"))
        text = _render_budget_markdown(payload)
        budget_id = payload.get("budget_id", "unknown")
        doc_id = f"{context.source.name}:{budget_id}:{blob.relative_path}"
        yield Document(
            id=doc_id,
            text=text,
            metadata=DocumentMetadata(
                source_name=context.source.name,
                source_version=context.source_version,
                source_location=blob.relative_path,
                ingested_at=context.ingested_at,
                extra={
                    "budget_id": payload.get("budget_id"),
                    "client_code": payload.get("client_code"),
                    "currency": payload.get("currency"),
                    "total_amount": payload.get("total_amount"),
                    "signed_at": payload.get("signed_at"),
                },
            ),
        )


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
