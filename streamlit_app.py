# streamlit_app.py

# --- Python Standard Library ---
import asyncio
import os
from typing import Any, Final

# --- Third Party Libraries ---
from dotenv import load_dotenv
import httpx
import streamlit as st
from streamlit import runtime

# --- App modules ---
from aie_estimator.schemas.estimations import DetailLevel, OutputFormat, ProjectType, ProviderMode
from aie_estimator.model_catalog import get_supported_models
from aie_estimator.settings import get_settings

load_dotenv()

DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
PAGE_TITLE: Final[str] = "Software Estimator"
PAGE_ICON: Final[str] = "🧮"  # :abacus: - Visit: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
MIN_DESCRIPTION_LENGTH: Final[int] = 20
MAX_DESCRIPTION_LENGTH: Final[int] = 2000

api_base_url: Final[str] = os.getenv("ESTIMATOR_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
estimate_endpoint: Final[str] = f"{api_base_url}/api/v1/estimate"


async def send_estimation_request(
        description: str,
        project_type: str,
        detail_level: str,
        output_format: str,
        *,
        model: str | None = None,
        engine: str = "litellm",
        endpoint: str = estimate_endpoint,
) -> dict[str, Any]:
    """
    Dispatch an asynchronous HTTP POST request to the backend estimation endpoint with typed request parameters.

    :param description: Free-text project description or transcription.
    :param project_type: Project domain category ('web_saas', 'mobile_app', 'internal_tool', 'data_pipeline').
    :param detail_level: Estimation depth tier ('summary', 'medium', 'detailed').
    :param output_format: Target output format ('phases_table', 'line_items', 'narrative').
    :param model: Optional model identifier override from supported catalog.
    :param engine: Provider dispatch engine ('litellm' or 'native').
    :param endpoint: Target backend estimation endpoint URL.

    :return: Deserialized JSON estimation response dictionary conforming to EstimationResponse.
    :raises httpx.HTTPError: If the backend service is unreachable or returns an HTTP error status code.
    """
    payload: dict[str, Any] = {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "engine": engine,
    }
    if model:
        payload["model"] = model

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()


def option_to_text(option: str, splitter: str = "_") -> str:
    """
    Remove splitter and capitalize words in a string for display purposes.

    :param option: The option string to convert (e.g., 'web_saas').
    :param splitter: The character used to split the option string (default is underscore).
    :return:
    """
    return " ".join(word.capitalize() for word in option.split(splitter))


def text_to_option(text: str, joiner: str = "_") -> str:
    """
    Convert a display string back to an option string with underscores.

    :param text: The display string to convert.
    :param joiner: The character used to join the words (default is underscore).
    :return:
    """
    return text.lower().replace(" ", joiner)


def render_telemetry_container(metadata: dict[str, Any]) -> None:
    """
    Render turn-anchored execution telemetry in an expandable container.

    :param metadata: Deserialized JSON estimation response dictionary containing model, provider, cost, and usage.
    """
    with st.expander(":material/insights: Execution telemetry", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", str(metadata.get("model", "N/A")))
        with col2:
            st.metric("Provider", str(metadata.get("provider", "N/A")))
        with col3:
            cost_usd = metadata.get("cost_usd")
            cost_label = f"${cost_usd:.5f}" if cost_usd is not None else "N/A"
            st.metric("Cost", cost_label)

        usage = metadata.get("usage", {})
        prep_in = usage.get("preprocessing_input_tokens", 0)
        prep_out = usage.get("preprocessing_output_tokens", 0)
        prep_info = f" [Prep tokens: {prep_in} in / {prep_out} out]" if (prep_in or prep_out) else ""
        st.markdown(
            f"**Token usage:** {usage.get('input_tokens', 0)} in / "
            f"{usage.get('output_tokens', 0)} out ({usage.get('total_tokens', 0)} total){prep_info}"
        )


if runtime.exists():
    # Application page setup
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
    )

    st.markdown(
        """
        <style>
            .stApp { font-size: 1.05rem; }
            h1 { font-size: 2.4rem !important; }
            h2 { font-size: 1.7rem !important; }
            h3 { font-size: 1.35rem !important; }
            
            /* Widgets, labels, buttons, markdown, text area */
            .stApp p,
            .stApp label,
            .stApp button,
            .stApp div[data-testid="stMarkdownContainer"] { font-size: 1.05rem; }
            .stTextArea textarea { font-size: 1.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for persistent estimation display
    if "latest_estimation" not in st.session_state:
        st.session_state.latest_estimation = None

    # Render sidebar controls and live configuration
    with st.sidebar:
        st.header("Service status")
        st.code(estimate_endpoint, language="text")

        settings = get_settings()
        configured_model = settings.llm_model
        cache_ttl_seconds = settings.cache_ttl

        st.markdown(f"**Configured model:** `{configured_model}`")
        st.markdown("**Resilience tier:** `Tier 1 (cost-based failover to Tier 2)`")
        st.markdown(f"**Cache TTL:** `{cache_ttl_seconds}s`")

        st.header("Generation parameters")
        with st.container(border=True):
            st.subheader(":material/tune: Model and engine")
            supported_models = get_supported_models()
            default_index = supported_models.index(configured_model) if configured_model in supported_models else 0
            selected_model: str = st.selectbox(
                "Target model",
                options=[option_to_text(m, "-") for m in supported_models],
                index=default_index,
                help="Choose an available LLM or dynamic routing profile from the catalog.",
            )
            supported_engines = [option_to_text(e) for e in ProviderMode.__args__]
            selected_engine: str | None = st.segmented_control(
                "Execution engine",
                options=supported_engines,
                default=supported_engines[0],
                help="LiteLLM provides dynamic provider routing; Native dispatches directly via SDK.",
            )

    # Render main product header
    st.title("Software Estimator")
    st.caption(
        "Fill in the project parameters below to generate a software effort estimation. "
        "The service processes your requirements using prompt templates and returns a structured effort breakdown."
    )

    # Render primary structured parameter form
    with st.form("estimation_form", clear_on_submit=False):
        project_description = st.text_area(
            "Project description",
            height=200,
            placeholder="Describe the project: goals, key features, constraints…",
            help=f"Between {MIN_DESCRIPTION_LENGTH} and {MAX_DESCRIPTION_LENGTH} characters.",
        )

        col_type, col_detail, col_format, col_submit = st.columns([1, 1, 1, 0.7], gap="small")
        with col_type:
            project_type = st.selectbox(
                "Project type",
                options=[option_to_text(t.value) for t in ProjectType],
                index=0,
            )
        with col_detail:
            detail_level = st.selectbox(
                "Detail level",
                options=[option_to_text(d.value) for d in DetailLevel],
                index=1,
            )
        with col_format:
            output_format = st.selectbox(
                "Output format",
                options=[option_to_text(f.value) for f in OutputFormat],
                index=0,
            )
        with col_submit:
            with st.container(horizontal=False):
                st.markdown('<div style="height: 1.9rem;"></div>', unsafe_allow_html=True)

                submitted = st.form_submit_button(
                    "Estimate",
                    type="primary",
                    width="stretch",
                )

    # Handle form submission and validation
    if submitted:
        cleaned_description = project_description.strip()
        if len(cleaned_description) < MIN_DESCRIPTION_LENGTH:
            st.error(
                f"The project description must be at least {MIN_DESCRIPTION_LENGTH} characters long.",
                icon=":material/error:",
            )
        elif len(cleaned_description) > MAX_DESCRIPTION_LENGTH:
            st.error(
                f"The project description cannot exceed {MAX_DESCRIPTION_LENGTH} characters.",
                icon=":material/error:",
            )
        else:
            with st.spinner("Calling the estimator service…"):
                try:
                    result = asyncio.run(
                        send_estimation_request(
                            cleaned_description,
                            project_type=text_to_option(project_type),
                            detail_level=text_to_option(detail_level),
                            output_format=text_to_option(output_format),
                            model=text_to_option(selected_model, "-"),
                            engine=text_to_option(selected_engine or "litellm"),
                        )
                    )
                    st.session_state.latest_estimation = result
                except httpx.HTTPStatusError as exc:
                    st.error(f"Service returned {exc.response.status_code}: {exc.response.text}",
                             icon=":material/error:")
                except httpx.RequestError as exc:
                    st.error(f"Could not reach the estimator backend at `{estimate_endpoint}`: {exc}",
                             icon=":material/error:")

    # Render latest estimation output and telemetry from session state
    if st.session_state.latest_estimation:
        estimation_data = st.session_state.latest_estimation
        prompt_version = estimation_data.get("prompt_version", "?")
        is_cached = bool(estimation_data.get("cached", False))

        cache_status = " :material/bolt: `Cached`" if is_cached else ""
        st.markdown(f"**Prompt version:** `{prompt_version}`{cache_status}")

        res = estimation_data.get("result")
        if isinstance(res, dict):
            summary = res.get("summary", "")
            confidence_pct = int(res.get("confidence_pct", 0))

            if confidence_pct < 30 or summary.startswith("Out of scope:"):
                st.warning(summary, icon=":material/warning:")
            else:
                st.info(summary, icon=":material/info:")

            col_cost, col_dur, col_conf = st.columns(3)
            with col_cost:
                st.metric("Total cost", f"€{res.get('total_cost_eur', 0):,}")
            with col_dur:
                st.metric("Total duration", f"{res.get('total_duration_weeks', 0)} weeks")
            with col_conf:
                st.metric("Confidence", f"{confidence_pct}%")

            phases = res.get("phases", [])
            if phases:
                st.subheader("Phase breakdown")
                phases_table = [
                    {
                        "Phase": phase.get("name", ""),
                        "Duration (weeks)": phase.get("duration_weeks", 0),
                        "Cost (€)": f"€{phase.get('cost_eur', 0):,}",
                        "Summary": phase.get("summary", ""),
                    }
                    for phase in phases
                ]
                st.dataframe(phases_table, width="stretch", hide_index=True)
        elif "text" in estimation_data:
            st.markdown(estimation_data.get("text", ""))

        render_telemetry_container(estimation_data)
