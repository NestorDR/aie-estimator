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
    send_estimation_request,
)

_APP_PATH: Final[str] = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
_DEFAULT_TIMEOUT: Final[float] = 10.0


# region send_estimation_request Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_send_estimation_request_dispatches_expected_payload() -> None:
    """
    GIVEN valid project estimation parameters
    WHEN send_estimation_request is executed
    THEN an asynchronous HTTP POST request is sent with the JSON payload conforming to EstimationRequest.
    """
    fake_response = {
        "text": "## Estimation Breakdown\n- Task 1: 10h",
        "prompt_version": "v1",
        "model": "reasoner-fast",
        "provider": "openrouter",
        "cost_usd": 0.0035,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 60,
            "total_tokens": 180,
            "preprocessing_input_tokens": 0,
            "preprocessing_output_tokens": 0,
        },
    }

    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await send_estimation_request(
            description="Build a high performance SaaS inventory management portal with analytics.",
            project_type="web_saas",
            detail_level="medium",
            output_format="phases_table",
        )

    assert result == fake_response
    mock_client_instance.post.assert_called_once()
    call_kwargs = mock_client_instance.post.call_args.kwargs
    assert call_kwargs["json"] == {
        "description": "Build a high performance SaaS inventory management portal with analytics.",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
        "engine": "litellm",
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("custom_kwargs", "expected_extra_keys"),
    [
        (
            {"model": "gpt-6-luna", "engine": "native"},
            {"model": "gpt-6-luna", "engine": "native"},
        ),
        (
            {"model": "claude-haiku-5-5", "engine": "litellm"},
            {"model": "claude-haiku-5-5", "engine": "litellm"},
        ),
    ],
    ids=["native_gpt", "litellm_claude"],
)
async def test_send_estimation_request_custom_overrides(
    custom_kwargs: dict[str, Any],
    expected_extra_keys: dict[str, Any],
) -> None:
    """
    GIVEN custom model and engine overrides
    WHEN send_estimation_request is executed
    THEN the dispatched HTTP POST payload serializes the specified overrides.

    :param custom_kwargs: Keyword arguments supplied to send_estimation_request.
    :param expected_extra_keys: Expected serialized override key-value pairs in JSON payload.
    """
    fake_response = {
        "text": "## Custom Estimation",
        "prompt_version": "v1",
        "model": expected_extra_keys.get("model", "default"),
        "provider": "openai",
        "cost_usd": 0.005,
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    }

    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await send_estimation_request(
            description="A comprehensive project description satisfying character bounds.",
            project_type="mobile_app",
            detail_level="detailed",
            output_format="line_items",
            **custom_kwargs,
        )

    assert result == fake_response
    call_kwargs = mock_client_instance.post.call_args.kwargs
    for key, value in expected_extra_keys.items():
        assert call_kwargs["json"][key] == value


@pytest.mark.anyio
async def test_send_estimation_request_raises_on_http_error() -> None:
    """
    GIVEN an HTTP error response from the backend
    WHEN send_estimation_request is executed
    THEN an httpx.HTTPStatusError exception is raised.
    """
    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance), pytest.raises(
        httpx.HTTPStatusError,
        match=r"Server Error",
    ):
        await send_estimation_request(
            description="A sufficiently detailed description that passes character length constraints.",
            project_type="web_saas",
            detail_level="summary",
            output_format="narrative",
        )


@pytest.mark.anyio
async def test_send_estimation_request_raises_on_request_error() -> None:
    """
    GIVEN a network connection failure
    WHEN send_estimation_request is executed
    THEN an httpx.RequestError exception is raised.
    """
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.RequestError("Connection refused", request=MagicMock())
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance), pytest.raises(
        httpx.RequestError,
        match=r"Connection refused",
    ):
        await send_estimation_request(
            description="A sufficiently detailed description that passes character length constraints.",
            project_type="internal_tool",
            detail_level="medium",
            output_format="phases_table",
        )


# endregion


# region Headless Streamlit AppTest Suite
# ---------------------------------------------------------------------------


def test_apptest_initial_render_and_sidebar_widgets() -> None:
    """
    GIVEN the Streamlit application module
    WHEN initialized within AppTest
    THEN the default title, form widgets, text area, select boxes, and session state are populated.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    assert not at.exception
    assert at.title[0].value == "Software Estimator"
    assert len(at.text_area) == 1
    assert f"Between {MIN_DESCRIPTION_LENGTH} and {MAX_DESCRIPTION_LENGTH} characters." in at.text_area[0].help
    assert len(at.selectbox) >= 2  # Project type, Output format, and Sidebar model
    assert len(at.button) >= 1  # Form submit button
    assert at.session_state.latest_estimation is None


def test_apptest_client_validation_short_description() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a project description with fewer than 20 characters is submitted
    THEN an error banner is displayed and no network request is executed.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.text_area[0].set_value("Too short")
    at.button[0].click().run()

    assert len(at.error) > 0
    assert f"at least {MIN_DESCRIPTION_LENGTH} characters" in at.error[0].value
    assert at.session_state.latest_estimation is None


def test_apptest_client_validation_long_description() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a project description exceeding 2000 characters is submitted
    THEN an error banner is displayed and no network request is executed.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    long_text = "x" * (MAX_DESCRIPTION_LENGTH + 1)
    at.text_area[0].set_value(long_text)
    at.button[0].click().run()

    assert len(at.error) > 0
    assert f"cannot exceed {MAX_DESCRIPTION_LENGTH} characters" in at.error[0].value
    assert at.session_state.latest_estimation is None


def test_apptest_valid_form_submission_and_telemetry() -> None:
    """
    GIVEN an initialized Streamlit app and a mock estimation response
    WHEN a valid project description is submitted through the primary form
    THEN the estimation markdown is rendered, prompt version is shown, and telemetry expander is populated.
    """
    fake_result = {
        "text": "## Comprehensive Project Estimation\n- Phase 1: 40h\n- Phase 2: 60h",
        "prompt_version": "v1",
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

    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_result
    mock_resp.raise_for_status.return_value = None
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.text_area[0].set_value("Client requires a full stack web portal with authentication and automated reporting.")
        at.button[0].click().run()

    assert not at.exception
    assert at.session_state.latest_estimation == fake_result
    assert any("Prompt version:" in m.value for m in at.markdown)
    assert any("Comprehensive Project Estimation" in m.value for m in at.markdown)


def test_apptest_session_state_persistence_across_reruns() -> None:
    """
    GIVEN a Streamlit app with prior estimation stored in session state
    WHEN rendered on a fresh rerun without form submission
    THEN the estimation markdown and telemetry remain displayed in the viewport.
    """
    persisted_result = {
        "text": "## Persisted Estimation Output",
        "prompt_version": "v1",
        "model": "gpt-6-luna",
        "provider": "openai",
        "cost_usd": 0.015,
        "usage": {"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
    }

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.session_state.latest_estimation = persisted_result
    at.run()

    assert not at.exception
    assert any("Persisted Estimation Output" in m.value for m in at.markdown)
    assert any("Prompt version:" in m.value for m in at.markdown)


@pytest.mark.parametrize(
    ("status_code", "error_message"),
    [
        (422, "Unprocessable Entity"),
        (500, "Internal Server Error"),
    ],
    ids=["validation_error_422", "server_error_500"],
)
def test_apptest_submission_http_status_errors(status_code: int, error_message: str) -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN the backend responds with an HTTP status error code
    THEN an error alert displaying the status code and text is presented.

    :param status_code: HTTP status error code.
    :param error_message: HTTP error response message text.
    """
    mock_client_instance = AsyncMock()
    mock_resp = MagicMock(status_code=status_code, text=error_message)
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        error_message,
        request=MagicMock(),
        response=mock_resp,
    )

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.text_area[0].set_value("Valid length description string to trigger backend status error handling.")
        at.button[0].click().run()

    assert len(at.error) > 0
    assert f"Service returned {status_code}" in at.error[0].value


def test_apptest_submission_network_request_error() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a network connectivity failure occurs during estimation request dispatch
    THEN a connection error banner is rendered.
    """
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.RequestError("Connection refused", request=MagicMock())
    mock_client_instance.__aenter__.return_value = mock_client_instance

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.text_area[0].set_value("Valid length description string to trigger connection error handling.")
        at.button[0].click().run()

    assert len(at.error) > 0
    assert any("Could not reach the estimator backend" in err.value for err in at.error)


# endregion
