# tests/integration/test_session_metadata_chaining.py

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

_TURN_1_TRANSCRIPT: Final[str] = "We are building MyProject, a B2B SaaS platform with FastAPI, PostgreSQL, and a 3-developer team."
_TURN_2_TRANSCRIPT: Final[str] = "For MyProject, replace PostgreSQL with MongoDB, reject Microservices, and enforce SOC 2 compliance."


@pytest.mark.anyio
async def test_two_turn_session_metadata_chaining_for_myproject(
    app: FastAPI,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a newly created session and two sequential multipart estimation turns for MyProject
    WHEN Turn 1 establishes initial metadata and Turn 2 updates technologies, constraints, and rejections
    THEN Turn 2 injects Turn 1's <project_metadata> into its system prompt, sends 3 messages to the LLM,
    and GET /api/v1/sessions/{id} reflects the merged ProjectMetadata state.

    :param app: Configured FastAPI application fixture.
    :param make_estimation_result: Shared estimation result factory fixture.
    """
    turn_1_delta = MetadataExtractionResult(
        project_name="MyProject",
        assumed_team_size=3,
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        agreed_scope="B2B SaaS MVP",
        explicit_constraints=["GDPR"],
        rejected_options=[],
    )
    turn_2_delta = MetadataExtractionResult(
        project_name="MyProject",
        assumed_team_size=4,
        mentioned_technologies=["MongoDB"],
        agreed_scope=None,
        explicit_constraints=["SOC 2"],
        rejected_options=["PostgreSQL", "Microservices"],
    )

    with (
        patch("aie_estimator.guardrails.input._check_moderation", new_callable=AsyncMock),
        patch(
            "aie_estimator.services.estimations_service.call_llm",
            new_callable=AsyncMock,
            side_effect=[make_estimation_result(), make_estimation_result()],
        ) as mock_estimation_llm,
        patch(
            "aie_estimator.sessions.metadata_extractor.call_llm",
            new_callable=AsyncMock,
            side_effect=[
                {"result": turn_1_delta, "usage": {"input_tokens": 80, "output_tokens": 30}},
                {"result": turn_2_delta, "usage": {"input_tokens": 80, "output_tokens": 30}},
            ],
        ) as mock_extractor_llm,
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
            create_resp = await async_client.post("/api/v1/sessions")
            assert create_resp.status_code == 201
            session_id = create_resp.json()["session_id"]

            resp_1 = await async_client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={
                    "transcript": _TURN_1_TRANSCRIPT,
                    "project_type": "web_saas",
                    "detail_level": "medium",
                    "output_format": "phases_table",
                },
            )
            assert resp_1.status_code == 200

            info_after_turn_1 = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()
            assert info_after_turn_1["message_count"] == 2
            assert info_after_turn_1["metadata"]["project_name"] == "MyProject"
            assert info_after_turn_1["metadata"]["mentioned_technologies"] == ["FastAPI", "PostgreSQL"]

            resp_2 = await async_client.post(
                f"/api/v1/sessions/{session_id}/estimate",
                data={
                    "transcript": _TURN_2_TRANSCRIPT,
                    "project_type": "web_saas",
                    "detail_level": "detailed",
                    "output_format": "phases_table",
                },
            )
            assert resp_2.status_code == 200

            info_after_turn_2 = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()

    assert mock_estimation_llm.await_count == 2
    assert mock_extractor_llm.await_count == 2

    turn_1_call_kwargs = mock_estimation_llm.await_args_list[0].kwargs
    assert "(No project context yet" in turn_1_call_kwargs["system_prompt"]
    assert len(turn_1_call_kwargs["messages"]) == 1

    turn_2_call_kwargs = mock_estimation_llm.await_args_list[1].kwargs
    assert "<project_metadata>" in turn_2_call_kwargs["system_prompt"]
    assert "MyProject" in turn_2_call_kwargs["system_prompt"]
    assert "FastAPI, PostgreSQL" in turn_2_call_kwargs["system_prompt"]

    turn_2_messages = turn_2_call_kwargs["messages"]
    assert len(turn_2_messages) == 3
    assert [msg["role"] for msg in turn_2_messages] == ["user", "assistant", "user"]
    assert turn_2_messages[0]["content"] == _TURN_1_TRANSCRIPT
    assert _TURN_2_TRANSCRIPT in turn_2_messages[2]["content"]

    assert info_after_turn_2["message_count"] == 4
    merged_meta = info_after_turn_2["metadata"]
    assert merged_meta["project_name"] == "MyProject"
    assert merged_meta["assumed_team_size"] == 4
    assert merged_meta["mentioned_technologies"] == ["FastAPI", "MongoDB"]
    assert merged_meta["agreed_scope"] == "B2B SaaS MVP"
    assert merged_meta["explicit_constraints"] == ["GDPR", "SOC 2"]
    assert merged_meta["rejected_options"] == ["PostgreSQL", "Microservices"]
