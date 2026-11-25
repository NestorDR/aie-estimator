# src/aie_estimator/ingestion/catalog/report.py

# --- App modules ---
from aie_estimator.ingestion.catalog.models import CatalogDecision, DataCatalog


def generate_audit_report(catalog: DataCatalog) -> str:
    """
    Render a human-readable Markdown audit report from a validated DataCatalog.

    :param catalog: Validated DataCatalog instance.

    :return: Markdown audit report string.
    """
    included = catalog.included_sources()
    review = [source for source in catalog.sources if source.decision is CatalogDecision.REVIEW]
    excluded = [source for source in catalog.sources if source.decision is CatalogDecision.EXCLUDE]

    lines = [
        f"# Data Audit Report (v{catalog.version})",
        "",
        (
            f"**Sources audited:** {len(catalog.sources)} | "
            f"**include:** {len(included)} | "
            f"**review:** {len(review)} | "
            f"**exclude:** {len(excluded)}"
        ),
        "",
        "## Included Sources",
        "",
    ]

    if included:
        for source in included:
            rag_status = "ready" if source.quality.is_rag_ready else "not_ready"
            lines.append(
                f"- **{source.name}** (`{source.format}`, access=`{source.sensitivity.access_level}`, "
                f"has_pii={source.sensitivity.has_pii}, rag_ready={rag_status})"
            )
    else:
        lines.append("- None")

    for title, sources in (("Sources Under Review", review), ("Excluded Sources", excluded)):
        if sources:
            lines.extend(["", f"## {title}", ""])
            for source in sources:
                lines.append(f"- **{source.name}** (`{source.format}`) — reason: {source.decision_reason}")

    return "\n".join(lines)
