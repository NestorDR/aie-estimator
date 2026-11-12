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
from aie_estimator.model_catalog import get_supported_models
from aie_estimator.schemas.estimations import DetailLevel, OutputFormat, ProjectType, ProviderMode
from aie_estimator.settings import get_settings

load_dotenv()

DEFAULT_API_BASE_URL: Final[str] = "http://localhost:8000"
PAGE_TITLE: Final[str] = "Software Estimator"
PAGE_ICON: Final[str] = "🧮"  # :abacus: - Visit: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
MIN_DESCRIPTION_LENGTH: Final[int] = 20
MAX_DESCRIPTION_LENGTH: Final[int] = 2000
_REQUEST_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(120.0, connect=10.0)

api_base_url: Final[str] = os.getenv("ESTIMATOR_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
sessions_endpoint: Final[str] = f"{api_base_url}/api/v1/sessions"


async def create_session(endpoint: str = sessions_endpoint) -> str:
    """
    Create a new conversational estimation session on the backend.

    :param endpoint: Base sessions endpoint URL.

    :return: Newly provisioned UUID v4 ``session_id`` string.
    :raises httpx.HTTPError: If the backend service is unreachable or returns an HTTP error status code.
    """
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        response = await client.post(endpoint)
        response.raise_for_status()
        return str(response.json()["session_id"])


async def get_session_info(session_id: str, endpoint: str = sessions_endpoint) -> dict[str, Any]:
    """
    Fetch current window counts and distilled ``ProjectMetadata`` for an active session.

    :param session_id: Target session UUID string.
    :param endpoint: Base sessions endpoint URL.

    :return: Deserialized ``SessionInfoResponse`` dictionary.
    :raises httpx.HTTPError: If the session is missing (404) or the backend is unreachable.
    """
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        response = await client.get(f"{endpoint}/{session_id}")
        response.raise_for_status()
        return response.json()


async def send_session_turn(
    session_id: str,
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    *,
    model: str | None = None,
    engine: str = "litellm",
    attachments: list[tuple[str, bytes, str]] | None = None,
    endpoint: str = sessions_endpoint,
) -> dict[str, Any]:
    """
    Dispatch a multi-turn estimation request with optional file attachments to a session endpoint.

    :param session_id: Target session UUID string.
    :param transcript: Current turn user transcript or requirements text.
    :param project_type: Project domain category (``web_saas``, ``mobile_app``, ``internal_tool``, ``data_pipeline``).
    :param detail_level: Estimation depth tier (``summary``, ``medium``, ``detailed``).
    :param output_format: Target output format (``phases_table``, ``line_items``, ``narrative``).
    :param model: Optional model identifier override from supported catalog.
    :param engine: Provider dispatch engine (``litellm`` or ``native``).
    :param attachments: Optional list of ``(filename, content_bytes, content_type)`` tuples.
    :param endpoint: Base sessions endpoint URL.

    :return: Deserialized JSON estimation response dictionary conforming to ``EstimationResponse``.
    :raises httpx.HTTPError: If the backend service is unreachable or returns an HTTP error status code.
    """
    form_data: dict[str, str] = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
        "engine": engine,
    }
    if model:
        form_data["model"] = model

    files = [
        ("attachments", (filename, content, content_type or "application/octet-stream"))
        for filename, content, content_type in (attachments or [])
    ]

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{endpoint}/{session_id}/estimate",
            data=form_data,
            files=files or None,
        )
        response.raise_for_status()
        return response.json()


def format_http_error(response: httpx.Response) -> tuple[str, str]:
    """
    Map backend HTTP error responses into user-facing alert messages and Material icons.

    :param response: Failed ``httpx.Response`` instance.

    :return: Tuple of ``(formatted_message, material_icon)``.
    """
    status_code = response.status_code
    try:
        payload = response.json()
    except Exception:
        payload = None

    detail = payload.get("detail") if isinstance(payload, dict) else None

    if status_code == 400 and isinstance(detail, dict) and "reason" in detail and "message" in detail:
        return (
            f"Request blocked by input guardrail ({detail['reason']}): {detail['message']}",
            ":material/security:",
        )

    if status_code == 415 and isinstance(detail, dict) and "filename" in detail and "message" in detail:
        return (
            f"Unsupported attachment '{detail['filename']}': {detail['message']}",
            ":material/error:",
        )

    if status_code == 422:
        if isinstance(detail, dict) and "filename" in detail and "message" in detail:
            return (
                f"Attachment error ({detail['filename']}): {detail['message']}",
                ":material/error:",
            )
        if isinstance(payload, dict) and isinstance(payload.get("errors"), list) and payload["errors"]:
            formatted_errors = []
            for err in payload["errors"]:
                loc = ".".join(str(part) for part in err.get("loc", []))
                msg = err.get("msg", "Invalid value")
                formatted_errors.append(f"{loc}: {msg}" if loc else str(msg))
            return f"Validation error: {'; '.join(formatted_errors)}", ":material/error:"

    return f"Service returned {status_code}: {response.text}", ":material/error:"


def option_to_text(option: str, splitter: str = "_") -> str:
    """
    Remove splitter and capitalize words in a string for display purposes.

    :param option: The option string to convert (e.g., ``web_saas``).
    :param splitter: The character used to split the option string (default is underscore).

    :return: Human-readable capitalized string.
    """
    return " ".join(word.capitalize() for word in option.split(splitter))


def text_to_option(text: str, joiner: str = "_") -> str:
    """
    Convert a display string back to an option string with underscores or hyphens.

    :param text: The display string to convert.
    :param joiner: The character used to join the words (default is underscore).

    :return: Normalized lowercase option identifier.
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


def render_estimation(estimation_data: dict[str, Any]) -> None:
    """
    Render a structured or text estimation response alongside execution telemetry.

    :param estimation_data: Deserialized ``EstimationResponse`` dictionary.
    """
    prompt_version = estimation_data.get("prompt_version", "?")
    st.markdown(f"**Prompt version:** `{prompt_version}`")

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
            st.table(phases_table, hide_index=True, alt="Phase breakdown table")
    elif "text" in estimation_data:
        st.markdown(estimation_data.get("text", ""))

    render_telemetry_container(estimation_data)


def render_user_turn(user_text: str, attachment_names: list[str]) -> None:
    """
    Render a user chat message with optional attachment filename badges.

    :param user_text: Submitted transcript text.
    :param attachment_names: Filenames attached to the turn.
    """
    with st.chat_message("user"):
        st.markdown(user_text)
        if attachment_names:
            st.caption(f":material/attach_file: Attachments: {', '.join(attachment_names)}")


def render_session_sidebar(session_info: dict[str, Any] | None, session_id: str | None) -> None:
    """
    Render live session window metrics and distilled ``ProjectMetadata`` in the sidebar.

    :param session_info: Deserialized ``SessionInfoResponse`` payload, if fetched.
    :param session_id: Active session UUID string, or ``None`` before the first turn.
    """
    st.subheader(":material/memory: Session state")
    if not session_id:
        st.caption("No active session yet. Send a message to start.")
        return

    st.markdown(f"**Session ID:** `{session_id}`")
    if not session_info:
        st.caption("Session metadata unavailable.")
        return

    msg_count = int(session_info.get("message_count", 0))
    max_turns = int(session_info.get("max_turns", 6))
    st.markdown(f"**Window:** `{msg_count} / {max_turns * 2}` messages")

    metadata = session_info.get("metadata", {})
    has_metadata = False
    if metadata.get("project_name"):
        st.markdown(f"**Project:** `{metadata['project_name']}`")
        has_metadata = True
    if metadata.get("assumed_team_size") is not None:
        st.markdown(f"**Team size:** `{metadata['assumed_team_size']}`")
        has_metadata = True
    if metadata.get("agreed_scope"):
        st.markdown(f"**Agreed scope:** {metadata['agreed_scope']}")
        has_metadata = True
    for label, key in (
        ("Technologies", "mentioned_technologies"),
        ("Constraints", "explicit_constraints"),
        ("Rejected", "rejected_options"),
    ):
        items = metadata.get(key) or []
        if items:
            st.markdown(f"**{label}:** {', '.join(f'`{item}`' for item in items)}")
            has_metadata = True

    if not has_metadata:
        st.caption("No project metadata extracted yet.")

    with st.expander("Raw session JSON", expanded=False):
        st.json(session_info)


def init_session_state() -> None:
    """Initialize conversational UI keys in ``st.session_state`` once."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "thread" not in st.session_state:
        st.session_state.thread = []
    if "session_info" not in st.session_state:
        st.session_state.session_info = None
    if "upload_nonce" not in st.session_state:
        st.session_state.upload_nonce = 0
    if "recovery_warning" not in st.session_state:
        st.session_state.recovery_warning = None


def reset_conversation() -> None:
    """Clear active session ID, thread history, and cached session inspection state."""
    st.session_state.session_id = None
    st.session_state.thread = []
    st.session_state.session_info = None
    st.session_state.upload_nonce += 1
    st.session_state.recovery_warning = None


def _dispatch_turn_with_recovery(
    transcript: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    model: str | None,
    engine: str,
    attachments: list[tuple[str, bytes, str]],
) -> dict[str, Any] | None:
    """
    Execute a session estimation turn with lazy creation and single-retry 404 recovery.

    :return: Deserialized ``EstimationResponse`` dictionary, or ``None`` if an error alert was shown.
    """
    try:
        if st.session_state.session_id is None:
            st.session_state.session_id = asyncio.run(create_session())

        try:
            return asyncio.run(
                send_session_turn(
                    session_id=st.session_state.session_id,
                    transcript=transcript,
                    project_type=project_type,
                    detail_level=detail_level,
                    output_format=output_format,
                    model=model,
                    engine=engine,
                    attachments=attachments,
                )
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                msg, icon = format_http_error(exc.response)
                st.error(msg, icon=icon)
                return None

            st.session_state.session_id = asyncio.run(create_session())
            st.session_state.thread = []
            st.session_state.session_info = None
            st.session_state.recovery_warning = (
                "Previous session expired or was reset; started a new session without earlier context."
            )
            return asyncio.run(
                send_session_turn(
                    session_id=st.session_state.session_id,
                    transcript=transcript,
                    project_type=project_type,
                    detail_level=detail_level,
                    output_format=output_format,
                    model=model,
                    engine=engine,
                    attachments=attachments,
                )
            )
    except httpx.HTTPStatusError as exc:
        msg, icon = format_http_error(exc.response)
        st.error(msg, icon=icon)
    except httpx.RequestError as exc:
        st.error(f"Could not reach the estimator backend at `{sessions_endpoint}`: {exc}", icon=":material/error:")
    return None


if runtime.exists():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
    )

    init_session_state()
    settings = get_settings()

    # Render sidebar controls
    with st.sidebar:
        st.header("Service status")
        st.code(sessions_endpoint, language="text")

        configured_model = settings.llm_model
        st.markdown(f"**Configured model:** `{configured_model}`")
        st.markdown("**Resilience tier:** `Tier 1 (cost-based failover to Tier 2)`")

        if st.button(":material/refresh: New conversation", type="primary", width="stretch", key="new_conversation_btn"):
            reset_conversation()

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

        with st.container(border=True):
            st.subheader(":material/sliders: Estimation dimensions")
            project_type = st.selectbox(
                "Project type",
                options=[option_to_text(t.value) for t in ProjectType],
                index=0,
            )
            detail_level = st.selectbox(
                "Detail level",
                options=[option_to_text(d.value) for d in DetailLevel],
                index=1,
            )
            output_format = st.selectbox(
                "Output format",
                options=[option_to_text(f.value) for f in OutputFormat],
                index=0,
            )

        supported_file_types = [ext.lstrip(".") for ext in settings.supported_attachment_extensions]
        with st.container(border=True):
            st.subheader(":material/attach_file: Attachments")
            st.html(
                """
                <style>
                div[class*="st-key-attachments_uploader_"] section button {
                    background-color: #ff4b4b;
                    color: #ffffff;
                    border: 1px solid #ff4b4b;
                    font-weight: 600;
                }
                div[class*="st-key-attachments_uploader_"] section button:hover {
                    background-color: #ff2b2b;
                    border-color: #ff2b2b;
                    color: #ffffff;
                }
                </style>
                """
            )
            uploaded_files = st.file_uploader(
                "Supporting files",
                type=supported_file_types,
                accept_multiple_files=True,
                key=f"attachments_uploader_{st.session_state.upload_nonce}",
                help="Optional .pdf, .docx, .txt, or .md specifications (max 5 files per turn).",
            )

    # Render main product header
    st.title("Software Estimator")
    st.caption(
        "Iteratively refine your software effort estimation across turns. "
        "Attach supporting specifications (.pdf, .docx, .txt, .md) from the sidebar before sending a message."
    )

    if st.session_state.recovery_warning:
        st.warning(st.session_state.recovery_warning, icon=":material/warning:")
        st.session_state.recovery_warning = None

    # Render existing conversation thread
    for turn in st.session_state.thread:
        render_user_turn(turn["user_text"], turn.get("attachment_names", []))
        with st.chat_message("assistant"):
            render_estimation(turn["estimation"])

    prompt_value = st.chat_input(
        "Describe your project or follow-up requirements…",
        key="chat_input",
    )

    if prompt_value is not None:
        cleaned_transcript = str(prompt_value).strip()

        if len(cleaned_transcript) < MIN_DESCRIPTION_LENGTH:
            st.error(
                f"The project description must be at least {MIN_DESCRIPTION_LENGTH} characters long.",
                icon=":material/error:",
            )
        elif len(cleaned_transcript) > MAX_DESCRIPTION_LENGTH:
            st.error(
                f"The project description cannot exceed {MAX_DESCRIPTION_LENGTH} characters.",
                icon=":material/error:",
            )
        else:
            files_list = list(uploaded_files or [])
            attachment_payloads = [
                (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")
                for uploaded in files_list
            ]
            attachment_names = [uploaded.name for uploaded in files_list]

            with st.spinner("Calling the estimator service…"):
                estimation_result = _dispatch_turn_with_recovery(
                    transcript=cleaned_transcript,
                    project_type=text_to_option(project_type),
                    detail_level=text_to_option(detail_level),
                    output_format=text_to_option(output_format),
                    model=text_to_option(selected_model, "-"),
                    engine=text_to_option(selected_engine or "litellm"),
                    attachments=attachment_payloads,
                )

            if estimation_result is not None:
                st.session_state.thread.append(
                    {
                        "user_text": cleaned_transcript,
                        "attachment_names": attachment_names,
                        "estimation": estimation_result,
                    }
                )
                try:
                    st.session_state.session_info = asyncio.run(
                        get_session_info(st.session_state.session_id)
                    )
                except httpx.HTTPError:
                    pass

                if files_list:
                    st.session_state.upload_nonce += 1
                    st.rerun()

                if st.session_state.recovery_warning:
                    st.warning(st.session_state.recovery_warning, icon=":material/warning:")
                    st.session_state.recovery_warning = None

                render_user_turn(cleaned_transcript, attachment_names)
                with st.chat_message("assistant"):
                    render_estimation(estimation_result)

    # Refresh sidebar session inspection if active session exists without cached info
    if st.session_state.session_id and st.session_state.session_info is None:
        try:
            st.session_state.session_info = asyncio.run(get_session_info(st.session_state.session_id))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                reset_conversation()
        except httpx.RequestError:
            pass

    if st.session_state.thread:
        window_turns = (
            st.session_state.session_info.get("max_turns", settings.session_max_turns)
            if st.session_state.session_info
            else settings.session_max_turns
        )
        st.caption(f"Model sees last {window_turns} turns.")

    with st.sidebar:
        render_session_sidebar(st.session_state.session_info, st.session_state.session_id)
