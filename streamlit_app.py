# streamlit_app.py

# --- Python Standard Library ---
import asyncio
from collections.abc import Iterator
import os
import time
from typing import Any, Final

# --- Third Party Libraries ---
from dotenv import load_dotenv
import httpx
import streamlit as st

# --- App modules ---
from aie_estimator.services.providers.model_catalog import get_supported_models
from aie_estimator.settings import get_settings

load_dotenv()

DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
STREAM_WORD_DELAY: Final[float] = 0.01  # Delay in seconds between streamed word chunks to simulate progressive output
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


async def send_estimation_request(
    transcription: str,
    *,
    model: str | None = None,
    engine: str = "litellm",
    preprocessing: str = "none",
    reasoning_effort: str | None = None,
    example_format: str = "markdown",
    use_examples: bool = True,
    num_examples: int = 3,
    max_tokens: int = 4000,
    evaluate: bool = True,
) -> dict[str, Any]:
    """
    Dispatch an asynchronous HTTP POST request to the backend estimation endpoint with configured parameters.

    :param transcription: Meeting transcript text to analyze.
    :param model: Target model identifier override from supported catalog.
    :param engine: Execution engine ('litellm' or 'native').
    :param preprocessing: Input preprocessing mode ('none', 'inline_cleaning', 'two_phase').
    :param reasoning_effort: Reasoning effort tier ('low', 'medium', 'high', or None).
    :param example_format: Format used to render reference CAG examples ('markdown', 'json', 'narrative').
    :param use_examples: Toggle reference CAG examples injection in system prompt.
    :param num_examples: Number of reference canonical examples to inject.
    :param max_tokens: Maximum output tokens for the estimation call.
    :param evaluate: Run structural evaluation on the generated estimation.

    :return: Deserialized JSON estimation response dictionary.
    :raises httpx.HTTPError: If the backend service is unreachable or responds with an HTTP error.
    """
    payload: dict[str, Any] = {
        "transcription": transcription,
        "engine": engine,
        "preprocessing": preprocessing,
        "example_format": example_format,
        "use_examples": use_examples,
        "num_examples": num_examples,
        "max_tokens": max_tokens,
        "evaluate": evaluate,
    }
    if model is not None:
        payload["model"] = model
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(estimate_endpoint, json=payload)
        response.raise_for_status()
        return response.json()


def render_estimation_metadata(metadata: dict[str, Any]) -> None:
    """
    Render turn-anchored telemetry and structural evaluation results in an expandable container.

    :param metadata: Deserialized JSON estimation response dictionary.
    """
    with st.expander(":material/insights: Execution details and evaluation", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", str(metadata.get("model", "N/A")))
            st.metric("Latency", f"{metadata.get('latency_ms', 0)} ms")
        with col2:
            st.metric("Provider", str(metadata.get("provider", "N/A")))
            cache_hit_label = "Hit" if metadata.get("cache_hit") else "Miss"
            st.metric("Cache", cache_hit_label)
        with col3:
            cost_usd = metadata.get("cost_usd")
            cost_label = f"${cost_usd:.5f}" if cost_usd is not None else "N/A"
            st.metric("Cost", cost_label)
            st.metric("Finish reason", str(metadata.get("finish_reason", "N/A")))

        usage = metadata.get("usage", {})
        prep_in = usage.get("preprocessing_input_tokens", 0)
        prep_out = usage.get("preprocessing_output_tokens", 0)
        prep_info = f" [Prep tokens: {prep_in} in / {prep_out} out]" if (prep_in or prep_out) else ""
        st.markdown(
            f"**Token usage:** {usage.get('input_tokens', 0)} in / "
            f"{usage.get('output_tokens', 0)} out ({usage.get('total_tokens', 0)} total){prep_info} | "
            f"**Preprocessing:** `{metadata.get('preprocessing', 'none')}`"
        )

        extracted_reqs = metadata.get("extracted_requirements")
        if extracted_reqs:
            with st.expander(":material/list_alt: Extracted requirements (Phase 1)", expanded=False):
                st.markdown(extracted_reqs)

        validation = metadata.get("validation")
        if validation:
            st.divider()
            score = float(validation.get("score", 0.0))
            score_pct = int(round(score * 100))
            if score >= 1.0:
                st.success(f"Structural evaluation passed ({score_pct}%)", icon=":material/check_circle:")
            else:
                st.warning(f"Structural evaluation incomplete ({score_pct}%)", icon=":material/warning:")

            decl_hours = validation.get("declared_total_hours")
            sum_hours = validation.get("sum_row_hours")
            hours_match = validation.get("hours_match")
            hours_match_str = "Match: OK" if hours_match else ("Mismatch" if hours_match is False else "N/A")

            decl_cost = validation.get("declared_total_cost")
            sum_cost = validation.get("sum_row_cost")
            cost_match = validation.get("cost_match")
            cost_match_str = "Match: OK" if cost_match else ("Mismatch" if cost_match is False else "N/A")

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                decl_h_str = f"{decl_hours}h" if decl_hours is not None else "N/A"
                sum_h_str = f"{sum_hours}h" if sum_hours is not None else "N/A"
                st.markdown(f"**Hours:** {decl_h_str} declared / {sum_h_str} table sum ({hours_match_str})")
            with v_col2:
                decl_c_str = f"{decl_cost:,.2f} EUR" if decl_cost is not None else "N/A"
                sum_c_str = f"{sum_cost:,.2f} EUR" if sum_cost is not None else "N/A"
                st.markdown(f"**Cost:** {decl_c_str} declared / {sum_c_str} table sum ({cost_match_str})")

            issues = validation.get("issues", [])
            if issues:
                st.markdown("**Issues detected:**")
                for issue in issues:
                    st.caption(f"• {issue}")


# Initialize conversation history and metrics in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

# Render sidebar controls and live configuration
with st.sidebar:
    st.header("Service status")
    st.code(estimate_endpoint, language="text")

    st.header("Generation parameters")

    with st.container(border=True):
        st.subheader(":material/tune: Model and engine")
        supported_models = get_supported_models()
        default_model = get_settings().llm_model
        default_idx = supported_models.index(default_model) if default_model in supported_models else 0
        selected_model: str = st.selectbox(
            "Target model",
            options=supported_models,
            index=default_idx,
            help="Select an available LLM or dynamic routing profile from the catalog.",
        )
        selected_engine: str = (
            st.segmented_control(
                "Execution engine",
                options=["litellm", "native"],
                default="litellm",
                help="LiteLLM provides dynamic provider routing; native dispatches directly via SDK.",
            )
            or "litellm"
        )
        selected_reasoning_effort_opt: str = (
            st.segmented_control(
                "Reasoning effort",
                options=["None", "low", "medium", "high"],
                default="None",
                help="Qualitative reasoning tier for supported reasoning models.",
            )
            or "None"
        )
        selected_reasoning_effort: str | None = None if selected_reasoning_effort_opt == "None" else selected_reasoning_effort_opt

    with st.container(border=True):
        st.subheader(":material/psychology: Context and preprocessing")
        selected_preprocessing: str = (
            st.segmented_control(
                "Preprocessing strategy",
                options=["inline_cleaning", "two_phase", "none"],
                default="none",
                help="Transcription preparation strategy prior to effort estimation.",
            )
            or "none"
        )
        selected_use_examples: bool = st.toggle(
            "Include canonical examples",
            value=True,
            help="Inject historical estimation examples from the canonical context repository.",
        )
        selected_example_format: str = (
            st.segmented_control(
                "Example format",
                options=["markdown", "json", "narrative"],
                default="markdown",
                disabled=not selected_use_examples,
                help="Serialization format for CAG reference examples.",
            )
            or "markdown"
        )
        selected_num_examples: int = st.slider(
            "Example count",
            min_value=0,
            max_value=5,
            value=3,
            step=1,
            disabled=not selected_use_examples,
            help="Number of reference canonical estimations injected into prompt context.",
        )

    with st.container(border=True):
        st.subheader(":material/rule: Output and validation")
        selected_max_tokens: int = st.slider(
            "Max output tokens",
            min_value=500,
            max_value=16000,
            value=4000,
            step=500,
            help="Token generation upper limit for the estimation response.",
        )
        selected_evaluate: bool = st.toggle(
            "Run structural evaluation",
            value=True,
            help="Perform Level-1 validation on generated estimation tables and totals.",
        )

    if st.button("Clear chat history", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.rerun()

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
        if message["role"] == "assistant" and message.get("metadata"):
            render_estimation_metadata(message["metadata"])

# Process user input
if prompt := st.chat_input(f"Paste meeting transcription text (min {MIN_TRANSCRIPTION_LENGTH} characters)..."):
    if len(prompt.strip()) < MIN_TRANSCRIPTION_LENGTH:
        st.warning(f"Meeting transcription must contain at least {MIN_TRANSCRIPTION_LENGTH} characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing transcription and assembling estimation..."):
                message_metadata: dict[str, Any] | None = None
                try:
                    result = asyncio.run(
                        send_estimation_request(
                            prompt,
                            model=selected_model,
                            engine=selected_engine,
                            preprocessing=selected_preprocessing,
                            reasoning_effort=selected_reasoning_effort,
                            example_format=selected_example_format,
                            use_examples=selected_use_examples,
                            num_examples=selected_num_examples if selected_use_examples else 0,
                            max_tokens=selected_max_tokens,
                            evaluate=selected_evaluate,
                        )
                    )
                    st.session_state.last_metrics = result
                    estimation_markdown = result.get("estimation", "")
                    rendered_response = st.write_stream(simulate_stream(estimation_markdown))
                    render_estimation_metadata(result)
                    message_metadata = result
                except httpx.HTTPStatusError as exc:
                    rendered_response = f"Backend returned HTTP error {exc.response.status_code}: {exc.response.text}"
                    st.error(rendered_response)
                except httpx.RequestError as exc:
                    rendered_response = f"Could not reach estimation backend at `{estimate_endpoint}`: {exc}"
                    st.error(rendered_response)

        assistant_message: dict[str, Any] = {"role": "assistant", "content": rendered_response}
        if message_metadata is not None:
            assistant_message["metadata"] = message_metadata
        st.session_state.messages.append(assistant_message)
