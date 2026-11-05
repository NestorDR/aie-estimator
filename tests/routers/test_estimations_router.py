# tests/routers/test_estimations_router.py

# --- Python Standard Library ---
from collections.abc import Callable
from typing import Any, Final
from unittest.mock import AsyncMock, patch

# --- Third Party Libraries ---
from fastapi.testclient import TestClient
import pytest

# --- App modules ---
from aie_estimator.services.providers.model_catalog import DEFAULT_FAST_PROFILE, DEFAULT_GPT_LLM

VALID_PAYLOAD: Final[dict[str, Any]] = {
    "description": "This is a comprehensive description of an inventory management SaaS system.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


@pytest.mark.parametrize(
    ("payload_override", "expected_loc", "expected_err_substring"),
    [
        ({"description": "Too short"}, "description", "at least 20 characters"),
        ({"description": ""}, "description", "at least 20 characters"),
        ({"description": "a" * 2001}, "description", "at most 2000 characters"),
        ({"project_type": "invalid_type"}, "project_type", "Input should be"),
        ({"detail_level": "invalid_level"}, "detail_level", "Input should be"),
        ({"output_format": "invalid_format"}, "output_format", "Input should be"),
        ({"model": "invalid-model-name"}, "model", "Input should be one of:"),
        ({"engine": "invalid_engine"}, "engine", "Input should be 'litellm' or 'native'"),
    ],
    ids=[
        "description_too_short",
        "description_empty",
        "description_too_long",
        "invalid_project_type",
        "invalid_detail_level",
        "invalid_output_format",
        "invalid_model",
        "invalid_engine",
    ],
)
def test_create_estimation_invalid_fields_returns_422(
    client: TestClient,
    payload_override: dict[str, Any],
    expected_loc: str,
    expected_err_substring: str,
) -> None:
    """
    GIVEN an estimation payload with invalid field values
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 422 Unprocessable Content is returned with field-level validation details.

    :param client: Shared TestClient fixture.
    :param payload_override: Dictionary of field overrides to test.
    :param expected_loc: Expected field name in error location.
    :param expected_err_substring: Substring expected in error message.
    """
    payload = {**VALID_PAYLOAD, **payload_override}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422

    body = response.json()
    errors = body.get("errors", [])
    assert any(err.get("loc") == ["body", expected_loc] for err in errors)
    assert any(expected_err_substring in err.get("msg", "") for err in errors)


@pytest.mark.parametrize(
    "missing_field",
    ["description", "project_type", "detail_level", "output_format"],
    ids=["missing_description", "missing_project_type", "missing_detail_level", "missing_output_format"],
)
def test_create_estimation_missing_required_fields_returns_422(
    client: TestClient,
    missing_field: str,
) -> None:
    """
    GIVEN an estimation payload omitting a mandatory schema field
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 422 Unprocessable Content is returned specifying the missing field.

    :param client: Shared TestClient fixture.
    :param missing_field: Field to remove from valid payload.
    """
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing_field}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 422

    body = response.json()
    errors = body.get("errors", [])
    assert any(err.get("loc") == ["body", missing_field] for err in errors)
    assert any("Field required" in err.get("msg", "") for err in errors)


def test_create_estimation_valid_request_returns_200(
    client: TestClient,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a valid estimation request
    WHEN a POST request is sent to /api/v1/estimate
    THEN HTTP 200 is returned with text, prompt_version, model, provider, and usage metadata.

    :param client: Shared TestClient fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            estimation="Detailed project effort breakdown narrative.",
            model=DEFAULT_GPT_LLM,
            provider="openai",
            cost_usd=0.0025,
            usage={"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    with patch("aie_estimator.routers.estimations_router.render_estimation_prompt") as mock_render:
        mock_render.return_value = ("System instructions", "User project description")

        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Detailed project effort breakdown narrative."
        assert data["prompt_version"] == "v1"
        assert data["model"] == DEFAULT_GPT_LLM
        assert data["provider"] == "openai"
        assert data["cost_usd"] == 0.0025
        assert data["usage"] == {
            "input_tokens": 120,
            "output_tokens": 60,
            "total_tokens": 180,
            "preprocessing_input_tokens": 0,
            "preprocessing_output_tokens": 0,
        }

        # Verify render_estimation_prompt was called with resolved active version "v1"
        mock_render.assert_called_once()
        _, call_kwargs = mock_render.call_args
        assert call_kwargs.get("version") == "v1"

        # Verify llm_service was called with rendered prompts
        mock_generate.assert_called_once()
        gen_args, gen_kwargs = mock_generate.call_args
        called_system = gen_kwargs.get("system_prompt") or (gen_args[0] if gen_args else None)
        called_user = gen_kwargs.get("user_prompt") or (gen_args[1] if len(gen_args) > 1 else None)
        assert called_system == "System instructions"
        assert called_user == "User project description"


@pytest.mark.parametrize(
    ("model_override", "expected_model", "expected_provider"),
    [
        (DEFAULT_GPT_LLM, DEFAULT_GPT_LLM, "openai"),
        (DEFAULT_FAST_PROFILE, DEFAULT_FAST_PROFILE, "litellm"),
    ],
    ids=["gpt_model_override", "fast_profile_override"],
)
def test_create_estimation_valid_model_override_accepted(
    client: TestClient,
    model_override: str,
    expected_model: str,
    expected_provider: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a valid model or logical profile identifier override
    WHEN a POST request is sent to /api/v1/estimate
    THEN the request passes validation and reflects the requested model and provider.

    :param client: Shared TestClient fixture.
    :param model_override: Model identifier to test.
    :param expected_model: Expected model in response.
    :param expected_provider: Expected provider in response.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            model=expected_model,
            provider=expected_provider,
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    payload = {**VALID_PAYLOAD, "model": model_override}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == expected_model
    assert data["provider"] == expected_provider


@pytest.mark.parametrize(
    ("cost_input", "expected_cost"),
    [
        (0.00125, 0.00125),
        (None, None),
    ],
    ids=["calculated_cost_present", "cost_null_default"],
)
def test_create_estimation_cost_usd_serialization(
    client: TestClient,
    cost_input: float | None,
    expected_cost: float | None,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an LLM response containing calculated cost_usd or None
    WHEN a POST request is processed by /api/v1/estimate
    THEN the response serializes cost_usd accurately in JSON.

    :param client: Shared TestClient fixture.
    :param cost_input: Cost value from LLM service response.
    :param expected_cost: Expected serialized cost value.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            model=DEFAULT_FAST_PROFILE,
            provider="litellm",
            cost_usd=cost_input,
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    payload = {**VALID_PAYLOAD, "model": DEFAULT_FAST_PROFILE}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["cost_usd"] == expected_cost
    assert data["model"] == DEFAULT_FAST_PROFILE


@pytest.mark.parametrize("engine_choice", ["litellm", "native"])
def test_create_estimation_explicit_engine_dispatch(
    client: TestClient,
    engine_choice: str,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an explicit execution engine selection ('litellm' or 'native')
    WHEN a POST request is processed by /api/v1/estimate
    THEN the request dispatches with the specified engine option.

    :param client: Shared TestClient fixture.
    :param engine_choice: Engine name to test.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            model=DEFAULT_GPT_LLM,
            provider="openai",
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    payload = {**VALID_PAYLOAD, "engine": engine_choice}
    response = client.post("/api/v1/estimate", json=payload)
    assert response.status_code == 200

    mock_generate.assert_called_once()
    call_args, call_kwargs = mock_generate.call_args
    passed_opts = call_kwargs.get("opts") or (call_args[2] if len(call_args) > 2 else None)
    assert passed_opts is not None
    assert passed_opts.engine == engine_choice


def test_create_estimation_prompt_version_decoupled_from_router_prefix(
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN an arbitrary router URL prefix mounted in a test application
    WHEN a POST request is sent to that endpoint
    THEN prompt resolution relies on configured active version rather than the router prefix.

    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    from fastapi import FastAPI
    from aie_estimator.routers.estimations_router import router

    custom_app = FastAPI()
    custom_app.include_router(router, prefix="/api/v99_custom")
    custom_client = TestClient(custom_app)

    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            estimation="Decoupled prompt version response.",
            model=DEFAULT_GPT_LLM,
            provider="openai",
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)

    with patch("aie_estimator.routers.estimations_router.render_estimation_prompt") as mock_render:
        mock_render.return_value = ("System prompt", "User prompt")

        response = custom_client.post("/api/v99_custom/api/v1/estimate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["prompt_version"] == "v1"

        mock_render.assert_called_once()
        _, call_kwargs = mock_render.call_args
        assert call_kwargs.get("version") == "v1"


def test_create_estimation_reflects_configured_active_prompt_version(
    client: TestClient,
    make_estimation_result: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GIVEN a custom active prompt version resolved by resolve_prompt_version
    WHEN a POST request is processed by /api/v1/estimate
    THEN the rendered prompt and response metadata reflect that configured active version.

    :param client: Shared TestClient fixture.
    :param make_estimation_result: Estimation result factory fixture.
    :param monkeypatch: Pytest monkeypatch fixture.
    """
    mock_generate = AsyncMock(
        return_value=make_estimation_result(
            estimation="Result with active version v2 candidate.",
            model=DEFAULT_GPT_LLM,
            provider="openai",
        )
    )
    monkeypatch.setattr("aie_estimator.routers.estimations_router.estimations_service.generate_estimation", mock_generate)
    monkeypatch.setattr("aie_estimator.routers.estimations_router.resolve_prompt_version", lambda: "v2-candidate")

    with patch("aie_estimator.routers.estimations_router.render_estimation_prompt") as mock_render:
        mock_render.return_value = ("System prompt v2", "User prompt v2")

        response = client.post("/api/v1/estimate", json=VALID_PAYLOAD)

        assert response.status_code == 200
        data = response.json()
        assert data["prompt_version"] == "v2-candidate"

        mock_render.assert_called_once()
        _, call_kwargs = mock_render.call_args
        assert call_kwargs.get("version") == "v2-candidate"
