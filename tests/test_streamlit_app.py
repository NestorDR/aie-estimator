# tests/test_streamlit_app.py

# --- Python Standard Library ---
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
import httpx
import pytest
from streamlit.testing.v1 import AppTest

# --- App modules ---
from streamlit_app import (
    MAX_DESCRIPTION_LENGTH,
    MIN_DESCRIPTION_LENGTH,
    create_session,
    format_http_error,
    get_session_info,
    send_session_turn,
)

_APP_PATH: Final[str] = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
_DEFAULT_TIMEOUT: Final[float] = 10.0


def _build_fake_estimation(
    *,
    summary: str = "Comprehensive Project Estimation",
    confidence_pct: int = 85,
    total_cost_eur: int = 25000,
    total_duration_weeks: int = 10,
    phases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a deterministic EstimationResponse dictionary for tests.

    :param summary: Estimation summary text.
    :param confidence_pct: Confidence percentage (0-100).
    :param total_cost_eur: Total estimated cost in EUR.
    :param total_duration_weeks: Total estimated duration in weeks.
    :param phases: Optional list of phase dictionaries.

    :return: Dictionary conforming to ``EstimationResponse``.
    """
    default_phases = [
        {
            "name": "Phase 1",
            "duration_weeks": 4,
            "cost_eur": 10000,
            "summary": "Core Backend Architecture",
        },
        {
            "name": "Phase 2",
            "duration_weeks": 6,
            "cost_eur": 15000,
            "summary": "Frontend Implementation",
        },
    ]
    return {
        "result": {
            "summary": summary,
            "confidence_pct": confidence_pct,
            "phases": default_phases if phases is None else phases,
            "total_duration_weeks": total_duration_weeks,
            "total_cost_eur": total_cost_eur,
        },
        "cached": False,
        "prompt_version": "v2",
        "model": "reasoner-fast",
        "provider": "openrouter",
        "cost_usd": 0.0084,
        "usage": {
            "input_tokens": 300,
            "output_tokens": 150,
            "total_tokens": 450,
            "preprocessing_input_tokens": 10,
            "preprocessing_output_tokens": 5,
        },
    }


def _build_fake_session_info(
    session_id: str = "11111111-1111-4111-8111-111111111111",
    message_count: int = 2,
    max_turns: int = 6,
) -> dict[str, Any]:
    """
    Build a deterministic SessionInfoResponse dictionary for tests.

    :param session_id: Session UUID string.
    :param message_count: Current message count in the sliding window.
    :param max_turns: Configured sliding-window turn capacity.

    :return: Dictionary conforming to ``SessionInfoResponse``.
    """
    return {
        "session_id": session_id,
        "message_count": message_count,
        "max_turns": max_turns,
        "metadata": {
            "project_name": "Portal MVP",
            "assumed_team_size": 3,
            "mentioned_technologies": ["FastAPI", "PostgreSQL"],
            "agreed_scope": "Auth and reporting",
            "explicit_constraints": ["EU hosting"],
            "rejected_options": ["Rails"],
        },
        "created_at": "2026-11-12T10:00:00Z",
        "last_accessed_at": "2026-11-12T10:05:00Z",
    }


def _make_async_client_mock(
    *,
    post_responses: list[MagicMock] | None = None,
    get_responses: list[MagicMock] | None = None,
    post_side_effect: Exception | None = None,
) -> AsyncMock:
    """
    Create an ``httpx.AsyncClient`` async context manager mock.

    :param post_responses: Optional sequence of responses for ``client.post``.
    :param get_responses: Optional sequence of responses for ``client.get``.
    :param post_side_effect: Optional exception to raise on ``client.post``.

    :return: Configured ``AsyncMock`` instance.
    """
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    if post_side_effect is not None:
        mock_client.post.side_effect = post_side_effect
    elif post_responses is not None:
        mock_client.post.side_effect = post_responses
    if get_responses is not None:
        mock_client.get.side_effect = get_responses
    return mock_client


# region Session Client & Error Formatter Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_session_returns_session_id() -> None:
    """
    GIVEN a backend /api/v1/sessions endpoint returning a session_id
    WHEN create_session is executed
    THEN it dispatches an async POST request and returns the provisioned session_id string.
    """
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"session_id": "abc-123"}
    mock_resp.raise_for_status.return_value = None
    mock_client = _make_async_client_mock(post_responses=[mock_resp])

    with patch("httpx.AsyncClient", return_value=mock_client):
        session_id = await create_session(endpoint="http://test/api/v1/sessions")

    assert session_id == "abc-123"
    mock_client.post.assert_called_once_with("http://test/api/v1/sessions")


@pytest.mark.anyio
async def test_get_session_info_fetches_session_payload() -> None:
    """
    GIVEN an active session identifier
    WHEN get_session_info is executed
    THEN it dispatches an async GET request and returns the deserialized SessionInfoResponse.
    """
    fake_info = _build_fake_session_info()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_info
    mock_resp.raise_for_status.return_value = None
    mock_client = _make_async_client_mock(get_responses=[mock_resp])

    with patch("httpx.AsyncClient", return_value=mock_client):
        info = await get_session_info("11111111-1111-4111-8111-111111111111", endpoint="http://test/api/v1/sessions")

    assert info == fake_info
    mock_client.get.assert_called_once_with("http://test/api/v1/sessions/11111111-1111-4111-8111-111111111111")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("extra_kwargs", "expected_extra_data", "expected_files"),
    [
        ({}, {"engine": "litellm"}, None),
        (
            {
                "model": "gpt-6-luna",
                "engine": "native",
                "attachments": [("spec.pdf", b"%PDF-1.4", "application/pdf")],
            },
            {"model": "gpt-6-luna", "engine": "native"},
            [("attachments", ("spec.pdf", b"%PDF-1.4", "application/pdf"))],
        ),
    ],
    ids=["defaults_no_attachments", "overrides_with_attachment"],
)
async def test_send_session_turn_dispatches_multipart_payload(
    extra_kwargs: dict[str, Any],
    expected_extra_data: dict[str, Any],
    expected_files: list[tuple[str, tuple[str, bytes, str]]] | None,
) -> None:
    """
    GIVEN valid conversational turn parameters and optional file attachments
    WHEN send_session_turn is executed
    THEN an async POST request is dispatched with multipart form data and files.

    :param extra_kwargs: Optional overrides (model, engine, attachments).
    :param expected_extra_data: Expected form fields in ``data=``.
    :param expected_files: Expected attachment tuples in ``files=``.
    """
    fake_response = _build_fake_estimation()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_client = _make_async_client_mock(post_responses=[mock_resp])

    transcript = "Build a high performance SaaS inventory management portal with analytics."
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await send_session_turn(
            session_id="sess-1",
            transcript=transcript,
            project_type="web_saas",
            detail_level="medium",
            output_format="phases_table",
            endpoint="http://test/api/v1/sessions",
            **extra_kwargs,
        )

    assert result == fake_response
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args.args[0] == "http://test/api/v1/sessions/sess-1/estimate"
    assert call_args.kwargs["data"] == {
        "transcript": transcript,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
        **expected_extra_data,
    }
    assert call_args.kwargs["files"] == expected_files


@pytest.mark.parametrize(
    ("status_code", "payload", "text", "expected_substrings", "expected_icon"),
    [
        (
            400,
            {"detail": {"reason": "prompt_injection", "message": "Suspicious instruction detected"}},
            "Bad Request",
            ["Request blocked by input guardrail (prompt_injection)", "Suspicious instruction detected"],
            ":material/security:",
        ),
        (
            415,
            {"detail": {"reason": "unsupported_attachment", "filename": "virus.exe", "message": "Unsupported ext"}},
            "Unsupported Media Type",
            ["Unsupported attachment 'virus.exe'", "Unsupported ext"],
            ":material/error:",
        ),
        (
            422,
            {"detail": {"reason": "attachment_extraction_failed", "filename": "bad.pdf", "message": "Corrupt PDF"}},
            "Unprocessable Entity",
            ["Attachment error (bad.pdf)", "Corrupt PDF"],
            ":material/error:",
        ),
        (
            422,
            {"detail": "Validation Error", "errors": [{"loc": ["body", "transcript"], "msg": "too short"}]},
            "Unprocessable Entity",
            ["Validation error:", "body.transcript: too short"],
            ":material/error:",
        ),
        (
            500,
            {"detail": "Unexpected server error"},
            "Internal Server Error",
            ["Service returned 500", "Internal Server Error"],
            ":material/error:",
        ),
    ],
    ids=["guardrail_400", "unsupported_415", "attachment_422", "validation_422", "server_500"],
)
def test_format_http_error_maps_status_codes(
    status_code: int,
    payload: dict[str, Any],
    text: str,
    expected_substrings: list[str],
    expected_icon: str,
) -> None:
    """
    GIVEN an HTTP error response from the backend
    WHEN format_http_error inspects its status code and JSON body
    THEN it returns an actionable formatted message and Material icon.

    :param status_code: HTTP status code.
    :param payload: JSON body dictionary.
    :param text: Raw response text fallback.
    :param expected_substrings: Substrings expected in the formatted message.
    :param expected_icon: Expected Material icon string.
    """
    mock_resp = MagicMock(status_code=status_code, text=text)
    mock_resp.json.return_value = payload

    message, icon = format_http_error(mock_resp)

    assert icon == expected_icon
    for snippet in expected_substrings:
        assert snippet in message


# endregion


# region Headless Streamlit AppTest Suite
# ---------------------------------------------------------------------------


def test_apptest_initial_render_and_sidebar_widgets() -> None:
    """
    GIVEN the conversational Streamlit application module
    WHEN initialized within AppTest
    THEN the title, chat input, sidebar controls, and lazy session state are populated without errors.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    assert not at.exception
    assert at.title[0].value == "Software Estimator"
    assert len(at.chat_input) == 1
    assert len(at.selectbox) >= 4  # Target model, Project type, Detail level, Output format
    assert len(at.button) >= 1  # New conversation button
    assert at.session_state.session_id is None
    assert at.session_state.thread == []
    assert at.session_state.session_info is None


@pytest.mark.parametrize(
    ("transcript", "expected_error"),
    [
        ("Too short", f"at least {MIN_DESCRIPTION_LENGTH} characters"),
        ("x" * (MAX_DESCRIPTION_LENGTH + 1), f"cannot exceed {MAX_DESCRIPTION_LENGTH} characters"),
    ],
    ids=["too_short", "too_long"],
)
def test_apptest_client_validation_transcript_length(transcript: str, expected_error: str) -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a chat transcript outside allowed bounds is submitted
    THEN an error banner is displayed and no session is created.

    :param transcript: Chat transcript violating length constraints.
    :param expected_error: Error text expected in the banner.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.chat_input[0].set_value(transcript).run()

    assert len(at.error) > 0
    assert expected_error in at.error[0].value
    assert at.session_state.session_id is None
    assert at.session_state.thread == []


def test_apptest_valid_chat_turn_creates_session_and_renders_metadata() -> None:
    """
    GIVEN an initialized Streamlit app with mocked session endpoints
    WHEN a valid chat transcript is submitted
    THEN a session is lazily created, the estimation thread renders, and sidebar metadata updates.
    """
    fake_session_id = "11111111-1111-4111-8111-111111111111"
    fake_estimation = _build_fake_estimation()
    fake_info = _build_fake_session_info(session_id=fake_session_id, message_count=2, max_turns=6)

    create_resp = MagicMock()
    create_resp.json.return_value = {"session_id": fake_session_id}
    create_resp.raise_for_status.return_value = None

    turn_resp = MagicMock()
    turn_resp.json.return_value = fake_estimation
    turn_resp.raise_for_status.return_value = None

    info_resp = MagicMock()
    info_resp.json.return_value = fake_info
    info_resp.raise_for_status.return_value = None

    mock_client = _make_async_client_mock(
        post_responses=[create_resp, turn_resp],
        get_responses=[info_resp],
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client):
        at.chat_input[0].set_value("Client requires a full stack web portal with authentication and reporting.").run()

    assert not at.exception
    assert at.session_state.session_id == fake_session_id
    assert len(at.session_state.thread) == 1
    assert at.session_state.session_info == fake_info
    assert any("Prompt version:" in m.value for m in at.markdown)
    assert any("Comprehensive Project Estimation" in i.value for i in at.info)
    assert any(m.label == "Total cost" and "€25,000" in m.value for m in at.metric)
    assert any(m.label == "Total duration" and "10 weeks" in m.value for m in at.metric)
    assert any(m.label == "Confidence" and "85%" in m.value for m in at.metric)
    assert len(at.table) >= 1
    assert any("Model sees last 6 turns." in c.value for c in at.caption)
    assert any("Portal MVP" in m.value for m in at.sidebar.markdown)


def test_apptest_thread_persistence_and_new_conversation_reset() -> None:
    """
    GIVEN a Streamlit app with an active session thread in session state
    WHEN the user clicks New conversation
    THEN the session ID, thread history, and session info are cleared.
    """
    fake_session_id = "11111111-1111-4111-8111-111111111111"
    persisted_estimation = _build_fake_estimation(summary="Persisted Turn Output", total_cost_eur=12000)
    fake_info = _build_fake_session_info(session_id=fake_session_id)

    info_resp = MagicMock()
    info_resp.json.return_value = fake_info
    info_resp.raise_for_status.return_value = None
    mock_client = _make_async_client_mock(get_responses=[info_resp])

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.session_state.session_id = fake_session_id
    at.session_state.thread = [
        {
            "user_text": "Initial web SaaS portal request with auth.",
            "attachment_names": ["spec.md"],
            "estimation": persisted_estimation,
        }
    ]
    at.session_state.session_info = fake_info

    with patch("httpx.AsyncClient", return_value=mock_client):
        at.run()

    assert any("Persisted Turn Output" in i.value for i in at.info)
    assert any("spec.md" in c.value for c in at.caption)

    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.session_state.session_id is None
    assert at.session_state.thread == []
    assert at.session_state.session_info is None


def test_apptest_low_confidence_out_of_scope_warning() -> None:
    """
    GIVEN a thread turn whose estimation has confidence below 30 or Out of scope prefix
    WHEN rendered in the Streamlit application
    THEN a warning alert is rendered instead of an info alert.
    """
    warning_estimation = _build_fake_estimation(
        summary="Out of scope: Requested hardware fabrication is outside CAG capability.",
        confidence_pct=20,
        total_cost_eur=0,
        total_duration_weeks=0,
        phases=[],
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.session_state.thread = [
        {
            "user_text": "Estimate hardware fabrication factory build.",
            "attachment_names": [],
            "estimation": warning_estimation,
        }
    ]
    at.run()

    assert not at.exception
    assert any("Out of scope:" in w.value for w in at.warning)


def test_apptest_404_session_recovery_recreates_and_retries() -> None:
    """
    GIVEN an active session that has expired on the backend (HTTP 404)
    WHEN the user submits a follow-up turn
    THEN the client creates a replacement session, clears prior thread, warns the user, and retries once.
    """
    stale_id = "00000000-0000-4000-8000-000000000000"
    new_id = "22222222-2222-4222-8222-222222222222"
    recovered_estimation = _build_fake_estimation(summary="Recovered Session Estimate")
    new_info = _build_fake_session_info(session_id=new_id)

    expired_resp = MagicMock(status_code=404, text="Session not found")
    expired_resp.json.return_value = {"detail": "Session not found"}
    expired_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found",
        request=MagicMock(),
        response=expired_resp,
    )

    recreate_resp = MagicMock()
    recreate_resp.json.return_value = {"session_id": new_id}
    recreate_resp.raise_for_status.return_value = None

    retry_resp = MagicMock()
    retry_resp.json.return_value = recovered_estimation
    retry_resp.raise_for_status.return_value = None

    info_resp = MagicMock()
    info_resp.json.return_value = new_info
    info_resp.raise_for_status.return_value = None

    mock_client = _make_async_client_mock(
        post_responses=[expired_resp, recreate_resp, retry_resp],
        get_responses=[info_resp],
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()
    at.session_state.session_id = stale_id
    at.session_state.thread = [
        {"user_text": "Old turn", "attachment_names": [], "estimation": _build_fake_estimation()}
    ]
    at.session_state.session_info = _build_fake_session_info(session_id=stale_id)

    with patch("httpx.AsyncClient", return_value=mock_client):
        at.chat_input[0].set_value("Follow-up turn after backend restart with enough length.").run()

    assert not at.exception
    assert at.session_state.session_id == new_id
    assert len(at.session_state.thread) == 1
    assert any("Previous session expired or was reset" in w.value for w in at.warning)
    assert any("Recovered Session Estimate" in i.value for i in at.info)


@pytest.mark.parametrize(
    ("status_code", "payload", "expected_snippet"),
    [
        (
            400,
            {"detail": {"reason": "prompt_injection", "message": "Ignore previous instructions detected"}},
            "Request blocked by input guardrail (prompt_injection)",
        ),
        (
            415,
            {"detail": {"reason": "unsupported_attachment", "filename": "bad.zip", "message": "Unsupported ext"}},
            "Unsupported attachment 'bad.zip'",
        ),
        (
            500,
            {"detail": "Unexpected server error"},
            "Service returned 500",
        ),
    ],
    ids=["guardrail_400", "unsupported_415", "server_500"],
)
def test_apptest_submission_http_status_errors(
    status_code: int,
    payload: dict[str, Any],
    expected_snippet: str,
) -> None:
    """
    GIVEN an active session in the Streamlit app
    WHEN the backend responds with HTTP 400, 415, or 500
    THEN the formatted error alert is displayed in the UI.

    :param status_code: HTTP status code returned by backend.
    :param payload: JSON response body.
    :param expected_snippet: Expected substring in ``at.error``.
    """
    create_resp = MagicMock()
    create_resp.json.return_value = {"session_id": "11111111-1111-4111-8111-111111111111"}
    create_resp.raise_for_status.return_value = None

    err_resp = MagicMock(status_code=status_code, text="Error text")
    err_resp.json.return_value = payload
    err_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "HTTP Error",
        request=MagicMock(),
        response=err_resp,
    )

    info_resp = MagicMock()
    info_resp.json.return_value = _build_fake_session_info()
    info_resp.raise_for_status.return_value = None

    mock_client = _make_async_client_mock(
        post_responses=[create_resp, err_resp],
        get_responses=[info_resp],
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client):
        at.chat_input[0].set_value("Valid length description string to trigger backend error handling.").run()

    assert len(at.error) > 0
    assert expected_snippet in at.error[0].value


def test_apptest_submission_network_request_error() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a network connectivity failure occurs during session creation or turn dispatch
    THEN a connection error banner is rendered.
    """
    mock_client = _make_async_client_mock(
        post_side_effect=httpx.RequestError("Connection refused", request=MagicMock())
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client):
        at.chat_input[0].set_value("Valid length description string to trigger connection error handling.").run()

    assert len(at.error) > 0
    assert any("Could not reach the estimator backend" in err.value for err in at.error)


# endregion
