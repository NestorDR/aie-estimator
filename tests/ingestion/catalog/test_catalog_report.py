# tests/ingestion/catalog/test_catalog_report.py

# --- App modules ---
from aie_estimator.ingestion.catalog.models import DataCatalog
from aie_estimator.ingestion.catalog.report import generate_audit_report


def test_generate_audit_report_includes_all_decision_sections() -> None:
    """
    GIVEN a validated DataCatalog containing include, review, and exclude sources
    WHEN generate_audit_report is called
    THEN the Markdown report contains summary counts, readiness, and non-include reasons.
    """
    catalog = DataCatalog.model_validate(
        {
            "version": "1.0.0",
            "sources": [
                {
                    "name": "budgets_json",
                    "location": "budgets",
                    "format": "json",
                    "quality": {"completeness": 4, "consistency": 3, "actuality": 5, "reliability": 4},
                    "sensitivity": {"has_pii": True, "pii_flags": ["EMAIL"], "access_level": "confidential"},
                    "decision": "include",
                },
                {
                    "name": "transcripts_txt",
                    "location": "transcripts",
                    "format": "txt",
                    "quality": {"completeness": 3, "consistency": 2, "actuality": 4, "reliability": 3},
                    "sensitivity": {"has_pii": True, "pii_flags": ["PERSON"], "access_level": "confidential"},
                    "decision": "review",
                    "decision_reason": "Mixed legacy format",
                },
                {
                    "name": "rate_card_xlsx",
                    "location": "rate_card_2024.xlsx",
                    "format": "xlsx",
                    "quality": {"completeness": 4, "consistency": 5, "actuality": 1, "reliability": 4},
                    "sensitivity": {"has_pii": False, "pii_flags": [], "access_level": "internal"},
                    "decision": "exclude",
                    "decision_reason": "Obsolete 2024 pricing",
                },
            ],
        }
    )

    report = generate_audit_report(catalog)
    assert "# Data Audit Report (v1.0.0)" in report
    assert "**Sources audited:** 3 | **include:** 1 | **review:** 1 | **exclude:** 1" in report
    assert "**budgets_json** (`json`, access=`confidential`, has_pii=True, rag_ready=ready)" in report
    assert "## Sources Under Review" in report
    assert "**transcripts_txt** (`txt`) — reason: Mixed legacy format" in report
    assert "## Excluded Sources" in report
    assert "**rate_card_xlsx** (`xlsx`) — reason: Obsolete 2024 pricing" in report


def test_generate_audit_report_without_included_sources() -> None:
    """
    GIVEN a catalog with zero included sources
    WHEN generate_audit_report is called
    THEN the Included Sources section renders '- None'.
    """
    catalog = DataCatalog.model_validate(
        {
            "version": "1.0.0",
            "sources": [
                {
                    "name": "rate_card_xlsx",
                    "location": "rate_card_2024.xlsx",
                    "format": "xlsx",
                    "quality": {"completeness": 4, "consistency": 5, "actuality": 1, "reliability": 4},
                    "sensitivity": {"has_pii": False, "pii_flags": [], "access_level": "internal"},
                    "decision": "exclude",
                    "decision_reason": "Obsolete 2024 pricing",
                }
            ],
        }
    )

    report = generate_audit_report(catalog)
    assert "## Included Sources\n\n- None" in report
    assert "## Sources Under Review" not in report
