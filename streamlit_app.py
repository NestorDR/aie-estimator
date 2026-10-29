# streamlit_app.py

# --- Python Standard Library ---
from collections.abc import Iterator
import os
import time
from typing import Any, Final

# --- Third Party Libraries ---
from dotenv import load_dotenv
import httpx
import streamlit as st

# --- App modules ---
from aie_estimator.services.llm_service import build_system_prompt

load_dotenv()

DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
STREAM_WORD_DELAY: Final[float] = 0.01
PAGE_TITLE: Final[str] = "Software Estimator"
PAGE_ICON: Final[str] = "📊"
MIN_TRANSCRIPTION_LENGTH: Final[int] = 50

# Application page setup
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
)

api_base_url: Final[str] = os.getenv("ESTIMATOR_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
estimate_endpoint: Final[str] = f"{api_base_url}/api/v1/estimate"


def simulate_stream(text: str, delay_seconds: float = STREAM_WORD_DELAY) -> Iterator[str]:
    """
    Yield word-level chunks from complete text with minor delay to simulate progressive streaming.

    :param text: Complete estimation Markdown text to stream.
    :param delay_seconds: Pause between successive word chunks in seconds.

    :return: Iterator yielding string tokens.
    """
    words = text.split(" ")
    for idx, word in enumerate(words):
        yield word if idx == len(words) - 1 else f"{word} "
        time.sleep(delay_seconds)


def send_estimation_request(transcription: str) -> dict[str, Any]:
    """
    Dispatch an HTTP POST request to the backend estimation endpoint.

    :param transcription: Meeting transcript text to analyze.

    :return: Deserialized JSON estimation response dictionary.
    :raises httpx.HTTPError: If the backend service is unreachable or responds with an HTTP error.
    """
    payload = {"transcription": transcription}
    with httpx.Client(timeout=120.0) as client:
        response = client.post(estimate_endpoint, json=payload)
        response.raise_for_status()
        return response.json()


# Initialize conversation history and metrics in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

# Render main chat header
st.title("Software Effort Estimator")
st.caption(
    "Paste a client meeting transcription below. The Context-Augmented Generation (CAG) "
    "service analyzes requirements and streams a structured project effort breakdown."
)

# Render historical messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Paste meeting transcription text (min 50 characters)..."):
    if len(prompt.strip()) < MIN_TRANSCRIPTION_LENGTH:
        st.warning(f"Meeting transcription must contain at least {MIN_TRANSCRIPTION_LENGTH} characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing transcription and assembling estimation..."):
                try:
                    result = send_estimation_request(prompt)
                    st.session_state.last_metrics = result
                    estimation_markdown = result.get("estimation", "")
                    rendered_response = st.write_stream(simulate_stream(estimation_markdown))
                except httpx.HTTPStatusError as exc:
                    rendered_response = f"Backend returned HTTP error {exc.response.status_code}: {exc.response.text}"
                    st.error(rendered_response)
                except httpx.RequestError as exc:
                    rendered_response = f"Could not reach estimation backend at `{estimate_endpoint}`: {exc}"
                    st.error(rendered_response)

        st.session_state.messages.append({"role": "assistant", "content": rendered_response})

# Render inspection sidebar (Level 3)
with st.sidebar:
    st.header("Service Status")
    st.code(estimate_endpoint, language="text")

    st.header("Last Call Metrics")
    metrics = st.session_state.last_metrics
    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model", metrics.get("model", "N/A"))
            st.metric("Latency", f"{metrics.get('latency_ms', 0)} ms")
        with col2:
            st.metric("Provider", metrics.get("provider", "N/A"))
            cache_hit_label = "Hit" if metrics.get("cache_hit") else "Miss"
            st.metric("Cache", cache_hit_label)

        usage = metrics.get("usage", {})
        st.markdown(
            f"**Token Usage:** {usage.get('input_tokens', 0)} in / "
            f"{usage.get('output_tokens', 0)} out ({usage.get('total_tokens', 0)} total)"
        )
    else:
        st.caption("No estimation request has been dispatched in this session yet.")

    st.header("Active CAG Context")
    with st.expander("Active System Prompt", expanded=False):
        active_system_prompt = build_system_prompt()
        st.text_area(
            "System Prompt Instructions",
            value=active_system_prompt,
            height=300,
            disabled=True,
            label_visibility="collapsed",
        )

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()
