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
from aie_estimator.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from aie_estimator.services.providers.model_catalog import get_supported_models
from aie_estimator.settings import get_settings

load_dotenv()

DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
PAGE_TITLE: Final[str] = "Software Estimator"
PAGE_ICON: Final[str] = "📊"
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

    # Initialize session state for persistent estimation display
    if "latest_estimation" not in st.session_state:
        st.session_state.latest_estimation = None

    # Render sidebar controls and live configuration
    with st.sidebar:
        st.header("Service status")
        st.code(estimate_endpoint, language="text")

        settings = get_settings()
        primary_model = settings.llm_model
        fallback_model = os.getenv("FALLBACK_MODEL", "claude-haiku-5-5")
        cache_ttl_seconds = settings.cache_ttl

        st.markdown(f"**Primary model:** `{primary_model}`")
        st.markdown(f"**Fallback model:** `{fallback_model}`")
        st.markdown(f"**Cache TTL:** `{cache_ttl_seconds}s`")

        st.header("Generation parameters")
        with st.container(border=True):
            st.subheader(":material/tune: Model and engine")
            supported_models = get_supported_models()
            default_index = supported_models.index(primary_model) if primary_model in supported_models else 0
            selected_model: str = st.selectbox(
                "Target model",
                options=supported_models,
                index=default_index,
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
        project_type = st.selectbox(
            "Project type",
            options=[t.value for t in ProjectType],
            index=1,
        )
        detail_level = (
            st.segmented_control(
                "Detail level",
                options=[d.value for d in DetailLevel],
                default=DetailLevel.MEDIUM.value,
            )
            or DetailLevel.MEDIUM.value
        )
        output_format = st.selectbox(
            "Output format",
            options=[f.value for f in OutputFormat],
            index=0,
        )
        submitted = st.form_submit_button("Generate estimation", type="primary", width="stretch")

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
                            project_type=project_type,
                            detail_level=detail_level,
                            output_format=output_format,
                            model=selected_model,
                            engine=selected_engine,
                        )
                    )
                    st.session_state.latest_estimation = result
                except httpx.HTTPStatusError as exc:
                    st.error(f"Service returned {exc.response.status_code}: {exc.response.text}", icon=":material/error:")
                except httpx.RequestError as exc:
                    st.error(f"Could not reach the estimator backend at `{estimate_endpoint}`: {exc}", icon=":material/error:")

    # Render latest estimation output and telemetry from session state
    if st.session_state.latest_estimation:
        estimation_data = st.session_state.latest_estimation
        st.markdown(f"**Prompt version:** `{estimation_data.get('prompt_version', '?')}`")
        st.markdown(estimation_data.get("text", ""))
        render_telemetry_container(estimation_data)
