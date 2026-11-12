# tests/integration/test_session_pdf_attachment.py

# --- Python Standard Library ---
from collections.abc import Callable
from typing import Any, Final
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

# --- App modules ---
from aie_estimator.sessions.models import MetadataExtractionResult
from tests.conftest import build_pdf_bytes

_TRANSCRIPT_TEXT: Final[str] = "Estimate MyProject based on the attached PDF specification document."
_PDF_SPEC_TEXT: Final[str] = "MyProject requires Kafka event streaming, OAuth2 SSO, and HIPAA audit logging."


@pytest.mark.anyio
async def test_pdf_attachment_enriches_prompt_and_influences_estimate(
    app: FastAPI,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN an active session and a multipart estimation request uploading a valid PDF specification
    WHEN POST /api/v1/sessions/{session_id}/estimate extracts the PDF text and invokes the LLM
    THEN the enriched user prompt contains an <attachments> block with the PDF content and the returned
    EstimationResponse reflects the PDF-driven scope while raw session history stores the enriched turn.

    :param app: Configured FastAPI application fixture.
    :param make_estimation_result: Shared estimation result factory fixture.
    """
    pdf_bytes = build_pdf_bytes(_PDF_SPEC_TEXT)
    pdf_result = make_estimation_result(
        result={
            "summary": "MyProject estimate incorporating PDF specification requirements.",
            "confidence_pct": 92,
            "phases": [
                {
                    "name": "Kafka Streaming & HIPAA Audit Pipeline",
                    "duration_weeks": 9,
                    "cost_eur": 27000,
                    "summary": "Event streaming backbone and HIPAA-compliant audit logging from PDF spec.",
                }
            ],
            "total_duration_weeks": 9,
            "total_cost_eur": 27000,
        }
    )

    with (
        patch("aie_estimator.guardrails.input._check_moderation", new_callable=AsyncMock),
        patch(
            "aie_estimator.services.estimations_service.call_llm",
            new_callable=AsyncMock,
            return_value=pdf_result,
        ) as mock_estimation_llm,
        patch(
            "aie_estimator.sessions.metadata_extractor.call_llm",
            new_callable=AsyncMock,
            return_value={
                "result": MetadataExtractionResult(
                    project_name="MyProject",
                    mentioned_technologies=["Kafka", "OAuth2"],
                    explicit_constraints=["HIPAA"],
                ),
                "usage": {"input_tokens": 70, "output_tokens": 25},
            },
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
            session_id = (await async_client.post("/api/v1/sessions")).json()["session_id"]

            response = await async_client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={
                    "transcript": _TRANSCRIPT_TEXT,
                    "project_type": "data_pipeline",
                    "detail_level": "detailed",
                    "output_format": "phases_table",
                },
                files=[("attachments", ("myproject_spec.pdf", pdf_bytes, "application/pdf"))],
            )
            assert response.status_code == 200
            payload = response.json()

            session_info = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()

    assert payload["result"]["total_duration_weeks"] == 9
    assert payload["result"]["total_cost_eur"] == 27000
    assert payload["result"]["phases"][0]["name"] == "Kafka Streaming & HIPAA Audit Pipeline"

    sent_messages = mock_estimation_llm.await_args.kwargs["messages"]
    assert len(sent_messages) == 1
    assert "<attachments>" in sent_messages[0]["content"]
    assert '<attachment filename="myproject_spec.pdf">' in sent_messages[0]["content"]
    assert _PDF_SPEC_TEXT in sent_messages[0]["content"]

    assert session_info["message_count"] == 2
    assert session_info["metadata"]["project_name"] == "MyProject"
    assert session_info["metadata"]["explicit_constraints"] == ["HIPAA"]
