# tests/test_streamlit_app.py

# --- Python Standard Library ---
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
import httpx
import pytest
from streamlit.testing.v1 import AppTest

# --- App modules ---
from streamlit_app import send_estimation_request, simulate_stream

_APP_PATH: Final[str] = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
_DEFAULT_TIMEOUT: Final[float] = 10.0


def test_simulate_stream_chunks_and_reconstructs_text() -> None:
    """
    GIVEN an estimation output string
    WHEN simulate_stream is iterated with zero delay
    THEN the sequence of yielded tokens reconstructs the original text exactly.
    """
    input_text = "Task 1: Requirements gathering.\nTask 2: API development."
    tokens = list(simulate_stream(input_text, delay_seconds=0.0))

    assert len(tokens) > 1
    reconstructed = "".join(tokens)
    assert reconstructed == input_text


@pytest.mark.parametrize(
    ("raw_text", "expected_token_count"),
    [
        ("Hello world", 2),
        ("SingleWord", 1),
        ("Three separate words", 3),
    ],
    ids=["two_words", "single_word", "three_words"],
)
def test_simulate_stream_word_token_counts(raw_text: str, expected_token_count: int) -> None:
    """
    GIVEN an input string with known word boundaries
    WHEN simulate_stream is invoked
    THEN the number of yielded chunks matches the expected word count.

    :param raw_text: Input text string.
    :param expected_token_count: Expected number of generated tokens.
    """
    tokens = list(simulate_stream(raw_text, delay_seconds=0.0))
    assert len(tokens) == expected_token_count


@pytest.mark.anyio
async def test_send_estimation_request_dispatches_expected_payload(
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN a meeting transcription
    WHEN send_estimation_request is executed
    THEN an HTTP POST request is sent with the JSON transcription payload and returns the response.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    """
    fake_response = make_estimation_result(
        estimation="## Estimation Breakdown",
        latency_ms=120,
    )

    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await send_estimation_request("Transcribed meeting discussion text about infrastructure.")

    assert result == fake_response
    mock_client_instance.post.assert_called_once()
    call_kwargs = mock_client_instance.post.call_args.kwargs
    assert call_kwargs["json"] == {
        "transcription": "Transcribed meeting discussion text about infrastructure.",
        "engine": "litellm",
        "preprocessing": "none",
        "example_format": "markdown",
        "use_examples": True,
        "num_examples": 3,
        "max_tokens": 4000,
        "evaluate": True,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("custom_kwargs", "expected_extra_keys"),
    [
        (
            {
                "model": "gpt-4o",
                "engine": "native",
                "preprocessing": "two_phase",
                "reasoning_effort": "high",
                "example_format": "json",
                "use_examples": False,
                "num_examples": 0,
                "max_tokens": 8000,
                "evaluate": False,
            },
            {
                "model": "gpt-4o",
                "engine": "native",
                "preprocessing": "two_phase",
                "reasoning_effort": "high",
                "example_format": "json",
                "use_examples": False,
                "num_examples": 0,
                "max_tokens": 8000,
                "evaluate": False,
            },
        ),
        (
            {
                "model": "claude-3-7-sonnet",
                "engine": "litellm",
                "preprocessing": "none",
                "reasoning_effort": "low",
                "example_format": "narrative",
                "use_examples": True,
                "num_examples": 5,
                "max_tokens": 12000,
                "evaluate": True,
            },
            {
                "model": "claude-3-7-sonnet",
                "engine": "litellm",
                "preprocessing": "none",
                "reasoning_effort": "low",
                "example_format": "narrative",
                "use_examples": True,
                "num_examples": 5,
                "max_tokens": 12000,
                "evaluate": True,
            },
        ),
    ],
    ids=["native_two_phase_json", "litellm_none_narrative"],
)
async def test_send_estimation_request_custom_parameters(
    make_estimation_result: Callable[..., dict[str, Any]],
    custom_kwargs: dict[str, Any],
    expected_extra_keys: dict[str, Any],
) -> None:
    """
    GIVEN custom estimation parameters
    WHEN send_estimation_request is executed
    THEN the HTTP POST request carries the serialized custom attributes in its JSON payload.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    :param custom_kwargs: Keyword arguments supplied to send_estimation_request.
    :param expected_extra_keys: Expected key-value pairs in the posted JSON payload.
    """
    fake_response = make_estimation_result(estimation="## Custom Estimation", latency_ms=250)

    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_response
    mock_resp.raise_for_status.return_value = None
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        result = await send_estimation_request(
            "Detailed meeting transcript with sufficient characters to pass validation.",
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
    WHEN send_estimation_request is called
    THEN httpx.HTTPStatusError is propagated.
    """
    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error",
        request=MagicMock(),
        response=MagicMock(status_code=500),
    )
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance), pytest.raises(httpx.HTTPStatusError, match=r"Server Error"):
        await send_estimation_request("Sample transcription text with minimum length required.")


# region Headless Streamlit AppTest Suite
# ---------------------------------------------------------------------------


def test_apptest_initial_render_and_sidebar_widgets() -> None:
    """
    GIVEN the Streamlit application file
    WHEN initialized within AppTest
    THEN default headers, title, chat input, and empty session states are populated.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    assert not at.exception
    assert at.title[0].value == "Software Effort Estimator"
    assert len(at.chat_input) == 1
    assert at.session_state.messages == []
    assert at.session_state.last_metrics is None


def test_apptest_short_transcript_warning() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a meeting transcript with fewer than 50 characters is submitted
    THEN a validation warning alert is rendered and no network request is sent.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    at.chat_input[0].set_value("Too short transcript").run()

    assert len(at.warning) > 0
    assert "at least 50 characters" in at.warning[0].value
    assert len(at.session_state.messages) == 0


def test_apptest_valid_transcript_submission_and_metrics(
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN an initialized Streamlit app and a mock estimation response
    WHEN a valid transcript is submitted through chat input
    THEN messages are appended to session state, estimation markdown is rendered, and metrics are stored.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    """
    fake_result = make_estimation_result(
        estimation="## Estimation Breakdown\n- Task 1: 10h",
        model="reasoner-fast",
        provider="openrouter",
        latency_ms=150,
        cost_usd=0.0042,
    )
    fake_result["extracted_requirements"] = "- Requirement 1: User login"
    fake_result["validation"] = {
        "score": 1.0,
        "issues": [],
        "declared_total_hours": 10,
        "sum_row_hours": 10,
        "hours_match": True,
        "declared_total_cost": 1500.0,
        "sum_row_cost": 1500.0,
        "cost_match": True,
        "finish_reason_ok": True,
        "has_title": True,
        "has_breakdown_table": True,
        "has_totals_section": True,
        "has_team_section": True,
        "has_duration_section": True,
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
        at.chat_input[0].set_value("Client requires a full stack web portal with authentication and automated reporting.").run()

    assert not at.exception
    assert len(at.session_state.messages) == 2
    assert at.session_state.messages[0]["role"] == "user"
    assert at.session_state.messages[1]["role"] == "assistant"
    assert at.session_state.messages[1]["metadata"] == fake_result
    assert at.session_state.last_metrics == fake_result


def test_apptest_submission_http_status_error() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN the backend responds with an HTTP status error
    THEN an error banner is displayed and the error message is recorded in session history.
    """
    mock_client_instance = AsyncMock()
    mock_resp = MagicMock(status_code=502, text="Bad Gateway")
    mock_client_instance.post.return_value = mock_resp
    mock_client_instance.__aenter__.return_value = mock_client_instance
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("Bad Gateway", request=MagicMock(), response=mock_resp)

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.chat_input[0].set_value("Valid length meeting transcription to trigger status error handling.").run()

    assert len(at.error) > 0
    assert "HTTP error 502" in at.error[0].value
    assert len(at.session_state.messages) == 2


def test_apptest_submission_request_error() -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN a network connectivity failure occurs during estimation request dispatch
    THEN a connection error banner is displayed.
    """
    mock_client_instance = AsyncMock()
    mock_client_instance.post.side_effect = httpx.RequestError("Connection refused", request=MagicMock())
    mock_client_instance.__aenter__.return_value = mock_client_instance

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.chat_input[0].set_value("Valid length meeting transcription to trigger request connection error.").run()

    assert any("Could not reach estimation backend" in err.value for err in at.error)
    assert len(at.session_state.messages) == 2


def test_apptest_validation_failures_and_clear_history() -> None:
    """
    GIVEN a Streamlit app with prior messages and structural validation failures in session state
    WHEN rendered and the clear history button is clicked
    THEN validation failure warnings are rendered and clicking clear resets session state.
    """
    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    failed_metadata = {
        "model": "gpt-6-luna",
        "provider": "openai",
        "latency_ms": 220,
        "cache_hit": False,
        "cost_usd": 0.0035,
        "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
        "extracted_requirements": "- Requirement A\n- Requirement B",
        "validation": {
            "score": 0.625,
            "issues": ["Totals section missing", "Invalid team sizing"],
            "declared_total_hours": None,
            "sum_row_hours": 40,
            "hours_match": False,
            "declared_total_cost": None,
            "sum_row_cost": 2400.0,
            "cost_match": False,
            "finish_reason_ok": True,
            "has_title": True,
            "has_breakdown_table": True,
            "has_totals_section": False,
            "has_team_section": False,
            "has_duration_section": True,
        },
    }
    at.session_state.messages = [
        {"role": "user", "content": "Previous prompt"},
        {"role": "assistant", "content": "Previous response", "metadata": failed_metadata},
    ]
    at.session_state.last_metrics = failed_metadata
    at.run()

    assert not at.exception
    assert len(at.warning) > 0
    assert any("Structural evaluation incomplete" in w.value for w in at.warning)

    # Click clear chat history button
    at.button[0].click().run()
    assert at.session_state.messages == []
    assert at.session_state.last_metrics is None


def test_apptest_successive_turns_retain_metadata(
    make_estimation_result: Callable[..., dict[str, Any]],
) -> None:
    """
    GIVEN an initialized Streamlit app
    WHEN multiple successive valid transcripts are submitted
    THEN each assistant message in session history retains its respective metadata dictionary.

    :param make_estimation_result: Shared factory fixture generating valid estimation dictionaries.
    """
    result_turn_1 = make_estimation_result(
        estimation="## Breakdown 1\n- Task A: 5h",
        model="turn-1-model",
        latency_ms=100,
        cost_usd=0.001,
    )
    result_turn_2 = make_estimation_result(
        estimation="## Breakdown 2\n- Task B: 15h",
        model="turn-2-model",
        latency_ms=250,
        cost_usd=0.003,
    )

    mock_client_instance = AsyncMock()
    mock_resp_1 = MagicMock()
    mock_resp_1.json.return_value = result_turn_1
    mock_resp_1.raise_for_status.return_value = None

    mock_resp_2 = MagicMock()
    mock_resp_2.json.return_value = result_turn_2
    mock_resp_2.raise_for_status.return_value = None

    mock_client_instance.post.side_effect = [mock_resp_1, mock_resp_2]
    mock_client_instance.__aenter__.return_value = mock_client_instance

    at = AppTest.from_file(_APP_PATH, default_timeout=_DEFAULT_TIMEOUT)
    at.run()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        at.chat_input[0].set_value("Initial meeting transcript passing minimum character requirements.").run()
        at.chat_input[0].set_value("Subsequent meeting transcript with different requirements passing check.").run()

    assert not at.exception
    assert len(at.session_state.messages) == 4
    assert at.session_state.messages[0]["role"] == "user"
    assert at.session_state.messages[1]["role"] == "assistant"
    assert at.session_state.messages[1]["metadata"] == result_turn_1
    assert at.session_state.messages[2]["role"] == "user"
    assert at.session_state.messages[3]["role"] == "assistant"
    assert at.session_state.messages[3]["metadata"] == result_turn_2
    assert at.session_state.last_metrics == result_turn_2

