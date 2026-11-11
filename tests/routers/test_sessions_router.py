# tests/routers/test_sessions_router.py

# --- Python Standard Library ---
from datetime import datetime
from uuid import UUID

# --- Third Party Libraries ---
from fastapi.testclient import TestClient

# --- App modules ---
from aie_estimator.dependencies import get_session_store
from aie_estimator.sessions.models import ProjectMetadata


def test_create_session_returns_201_with_distinct_uuid_v4(client: TestClient) -> None:
    """
    GIVEN a running API server
    WHEN POST /api/v1/sessions is called twice
    THEN each response returns HTTP 201 with a distinct valid UUID v4 session_id and preserves earlier sessions.

    :param client: Shared TestClient fixture.
    """
    first_resp = client.post("/api/v1/sessions")
    second_resp = client.post("/api/v1/sessions")

    assert first_resp.status_code == 201
    assert second_resp.status_code == 201

    first_id = first_resp.json()["session_id"]
    second_id = second_resp.json()["session_id"]
    assert first_id != second_id
    assert UUID(first_id).version == 4
    assert UUID(second_id).version == 4

    assert client.get(f"/api/v1/sessions/{first_id}").status_code == 200
    assert client.get(f"/api/v1/sessions/{second_id}").status_code == 200


def test_get_session_returns_200_with_state_and_timestamps(client: TestClient) -> None:
    """
    GIVEN an active session containing one conversation turn and updated ProjectMetadata
    WHEN GET /api/v1/sessions/{session_id} is requested
    THEN HTTP 200 is returned with message_count, max_turns, metadata, and UTC timestamps.

    :param client: Shared TestClient fixture.
    """
    create_resp = client.post("/api/v1/sessions")
    session_id = create_resp.json()["session_id"]

    store = get_session_store()
    session = store.get_or_404(session_id)
    session.history.append(user="Estimate BookFlow", assistant="Estimated 6 weeks.")
    session.metadata = ProjectMetadata(
        project_name="BookFlow",
        assumed_team_size=3,
        mentioned_technologies=["FastAPI", "PostgreSQL"],
        agreed_scope="MVP phase 1",
        explicit_constraints=["GDPR"],
        rejected_options=["Microservices"],
    )

    response = client.get(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["message_count"] == 2
    assert payload["max_turns"] == 6
    assert payload["metadata"]["project_name"] == "BookFlow"
    assert payload["metadata"]["rejected_options"] == ["Microservices"]

    created_at = datetime.fromisoformat(payload["created_at"])
    last_accessed_at = datetime.fromisoformat(payload["last_accessed_at"])
    assert last_accessed_at >= created_at


def test_get_unknown_session_returns_404(client: TestClient) -> None:
    """
    GIVEN an unknown session identifier
    WHEN GET /api/v1/sessions/{session_id} is requested
    THEN HTTP 404 Not Found is returned with the centralized error detail.

    :param client: Shared TestClient fixture.
    """
    response = client.get("/api/v1/sessions/non-existent-session-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
