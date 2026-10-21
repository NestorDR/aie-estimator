# src/aie_estimator/context/examples.py

# --- Python Standard Library ---
from typing import Final

ESTIMATION_EXAMPLES: Final[list[dict[str, str]]] = [
    {
        "meeting_summary": (
            "The client needs a web-based inventory management platform with role-based access control, "
            "barcode scanning integration, automated low-stock email alerts, and an analytics dashboard "
            "for warehouse managers."
        ),
        "estimation": """## Estimation: Inventory Management Platform

### Task Breakdown:
1. Architecture & Data Modeling: 24 hours
2. Authentication & Role-Based Access Control: 28 hours
3. Inventory CRUD & Barcode Scanner Integration: 50 hours
4. Low-Stock Notification & Alert Engine: 20 hours
5. Analytics Dashboard & Reporting Views: 36 hours
6. Automated Testing & Quality Assurance: 25 hours

**Total Estimated Hours:** 183 hours
**Recommended Team:** 2 Full-Stack Developers + 1 QA Engineer (part-time)
**Estimated Duration:** 6-8 weeks""",
    },
    {
        "meeting_summary": (
            "The client needs a customer support portal with ticketing workflow, customer self-service "
            "knowledge base search, third-party CRM webhook synchronization, and automated SLA tracking."
        ),
        "estimation": """## Estimation: Customer Support Portal

### Task Breakdown:
1. Database Schema & Ticket Lifecycle State Machine: 32 hours
2. Agent Ticket Management UI & Threaded Comments: 44 hours
3. Knowledge Base Search & Public Article Viewer: 24 hours
4. Webhook Ingestion & CRM Synchronization: 30 hours
5. SLA Escalation Cron Jobs & Metrics Dashboard: 22 hours
6. End-to-End Testing & Deployment Pipeline: 20 hours

**Total Estimated Hours:** 172 hours
**Recommended Team:** 2 Backend Developers + 1 Frontend Developer
**Estimated Duration:** 5-7 weeks""",
    },
]
