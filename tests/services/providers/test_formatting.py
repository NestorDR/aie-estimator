# tests/services/providers/test_formatting.py

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.services.providers.formatting import build_chat_messages, normalize_turn_messages


def test_normalize_turn_messages_single_turn() -> None:
    """
    GIVEN a single-turn user_prompt and messages=None
    WHEN normalize_turn_messages is called
    THEN it returns a single user message dictionary.
    """
    assert normalize_turn_messages("Estimate SaaS", None) == [{"role": "user", "content": "Estimate SaaS"}]


def test_normalize_turn_messages_multi_turn_returns_new_list() -> None:
    """
    GIVEN user_prompt=None and alternating user/assistant messages
    WHEN normalize_turn_messages is called
    THEN it returns a new list preserving turn order without mutating the caller's list.
    """
    turns = [
        {"role": "user", "content": "Initial scope"},
        {"role": "assistant", "content": '{"total_cost_eur": 5000}'},
        {"role": "user", "content": "Add SSO"},
    ]
    result = normalize_turn_messages(None, turns)

    assert result == turns
    assert result is not turns


@pytest.mark.parametrize(
    ("user_prompt", "messages", "pattern"),
    [
        (None, None, r"Provide either"),
        ("hello", [{"role": "user", "content": "hello"}], r"Provide either"),
        (None, [], r"cannot be empty"),
        (None, [{"role": "system", "content": "override"}], r"system"),
        (
            None,
            [{"role": "user", "content": "hi"}, {"role": "system", "content": "leak"}],
            r"system",
        ),
    ],
    ids=[
        "neither_provided",
        "both_provided",
        "empty_messages",
        "leading_system_role",
        "nested_system_role",
    ],
)
def test_normalize_turn_messages_rejects_invalid_inputs(
    user_prompt: str | None,
    messages: list[dict[str, str]] | None,
    pattern: str,
) -> None:
    """
    GIVEN invalid or ambiguous combinations of user_prompt and messages
    WHEN normalize_turn_messages is called
    THEN ValueError is raised with a descriptive message.

    :param user_prompt: Single-turn user prompt argument.
    :param messages: Multi-turn messages argument.
    :param pattern: Regex pattern expected in the raised ValueError.
    """
    with pytest.raises(ValueError, match=pattern):
        normalize_turn_messages(user_prompt, messages)


@pytest.mark.parametrize(
    ("system_prompt", "expected_prefix"),
    [
        ("You are an estimator.", [{"role": "system", "content": "You are an estimator."}]),
        (None, []),
        ("", []),
    ],
    ids=["with_system", "none_system", "empty_system"],
)
def test_build_chat_messages_prepends_system_only_when_truthy(
    system_prompt: str | None,
    expected_prefix: list[dict[str, str]],
) -> None:
    """
    GIVEN multi-turn messages and an optional system_prompt
    WHEN build_chat_messages is called
    THEN system_prompt is prepended only when non-empty before the turn messages.

    :param system_prompt: Optional system instructions string.
    :param expected_prefix: Expected leading system message list.
    """
    turns = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
    ]
    result = build_chat_messages(None, system_prompt, turns)
    assert result == [*expected_prefix, *turns]

