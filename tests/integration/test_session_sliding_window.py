# tests/integration/test_session_sliding_window.py

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

_TOTAL_TURNS: Final[int] = 8
_MAX_TURNS: Final[int] = 6
_MAX_MESSAGES_SENT_TO_LLM: Final[int] = (_MAX_TURNS * 2) + 1


@pytest.mark.anyio
async def test_eight_turn_conversation_respects_sliding_window_and_retains_turn_1_metadata(
    app: FastAPI,
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a session with max_turns=6 and an 8-turn sequential conversation where Turn 1 sets MyProject metadata
    WHEN 8 estimation requests are sent to POST /api/v1/sessions/{session_id}/estimate
    THEN the messages array passed to call_llm never exceeds 13 items (12 history + 1 current user turn),
    Turns 1 and 2 are evicted from raw history by Turn 8, and Turn 1's MyProject metadata remains intact.

    :param app: Configured FastAPI application fixture.
    :param make_estimation_result: Shared estimation result factory fixture.
    """
    extractor_responses = [
        {
            "result": (
                MetadataExtractionResult(
                    project_name="MyProject",
                    assumed_team_size=5,
                    mentioned_technologies=["FastAPI"],
                    agreed_scope="Core SaaS platform",
                    explicit_constraints=["ISO 27001"],
                )
                if turn_idx == 1
                else MetadataExtractionResult(mentioned_technologies=[f"TechTurn{turn_idx}"])
            ),
            "usage": {"input_tokens": 60, "output_tokens": 20},
        }
        for turn_idx in range(1, _TOTAL_TURNS + 1)
    ]

    with (
        patch("aie_estimator.guardrails.input._check_moderation", new_callable=AsyncMock),
        patch(
            "aie_estimator.services.estimations_service.call_llm",
            new_callable=AsyncMock,
            side_effect=[make_estimation_result() for _ in range(_TOTAL_TURNS)],
        ) as mock_estimation_llm,
        patch(
            "aie_estimator.sessions.metadata_extractor.call_llm",
            new_callable=AsyncMock,
            side_effect=extractor_responses,
        ),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
            session_id = (await async_client.post("/api/v1/sessions")).json()["session_id"]

            for turn_idx in range(1, _TOTAL_TURNS + 1):
                turn_resp = await async_client.post(
                    f"/api/v1/sessions/{session_id}/estimate",
                    data={
                        "transcript": f"Turn {turn_idx}: Refine the MyProject architecture and delivery milestones.",
                        "project_type": "web_saas",
                        "detail_level": "medium",
                        "output_format": "phases_table",
                    },
                )
                assert turn_resp.status_code == 200

            final_session_info = (await async_client.get(f"/api/v1/sessions/{session_id}")).json()

    observed_message_counts = [len(call.kwargs["messages"]) for call in mock_estimation_llm.await_args_list]
    assert observed_message_counts == [1, 3, 5, 7, 9, 11, 13, 13]
    assert max(observed_message_counts) <= _MAX_MESSAGES_SENT_TO_LLM

    turn_8_kwargs = mock_estimation_llm.await_args_list[-1].kwargs
    turn_8_messages = turn_8_kwargs["messages"]
    assert len(turn_8_messages) == _MAX_MESSAGES_SENT_TO_LLM
    assert "Turn 1:" not in " ".join(msg["content"] for msg in turn_8_messages)
    assert "Turn 2:" in turn_8_messages[0]["content"]
    assert "Turn 8:" in turn_8_messages[-1]["content"]

    assert "<project_metadata>" in turn_8_kwargs["system_prompt"]
    assert "MyProject" in turn_8_kwargs["system_prompt"]
    assert "ISO 27001" in turn_8_kwargs["system_prompt"]

    assert final_session_info["message_count"] == _MAX_TURNS * 2
    assert final_session_info["max_turns"] == _MAX_TURNS
    assert final_session_info["metadata"]["project_name"] == "MyProject"
    assert final_session_info["metadata"]["assumed_team_size"] == 5
    assert final_session_info["metadata"]["agreed_scope"] == "Core SaaS platform"
    assert final_session_info["metadata"]["explicit_constraints"] == ["ISO 27001"]
