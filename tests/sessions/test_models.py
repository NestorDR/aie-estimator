# tests/sessions/test_models.py

# --- Third Party Libraries ---
from pydantic import ValidationError
import pytest

# --- App modules ---
from aie_estimator.sessions.models import ConversationHistory, Message, MetadataExtractionResult, ProjectMetadata, Session


def test_message_rejects_system_role() -> None:
    """
    GIVEN a Message constructor call with role="system"
    WHEN Message validates the payload
    THEN a ValidationError is raised restricting roles to user or assistant.
    """
    with pytest.raises(ValidationError, match=r"Input should be 'user' or 'assistant'"):
        Message(role="system", content="System prompt")  # type: ignore[arg-type]


def test_conversation_history_append_within_capacity() -> None:
    """
    GIVEN a ConversationHistory with max_turns=2
    WHEN two user/assistant pairs are appended
    THEN no messages are evicted and to_messages preserves alternating roles.
    """
    history = ConversationHistory(max_turns=2)

    evicted_1 = history.append(user="Turn 1 user", assistant="Turn 1 assistant")
    evicted_2 = history.append(user="Turn 2 user", assistant="Turn 2 assistant")

    assert evicted_1 == []
    assert evicted_2 == []
    assert history.to_messages() == [
        {"role": "user", "content": "Turn 1 user"},
        {"role": "assistant", "content": "Turn 1 assistant"},
        {"role": "user", "content": "Turn 2 user"},
        {"role": "assistant", "content": "Turn 2 assistant"},
    ]


def test_conversation_history_append_evicts_oldest_pairs_in_order() -> None:
    """
    GIVEN a ConversationHistory with max_turns=2 already holding two pairs
    WHEN a third pair is appended
    THEN the oldest pair is evicted and returned in chronological order while preserving user-first alternation.
    """
    history = ConversationHistory(max_turns=2)
    history.append(user="U1", assistant="A1")
    history.append(user="U2", assistant="A2")

    evicted = history.append(user="U3", assistant="A3")

    assert [(m.role, m.content) for m in evicted] == [("user", "U1"), ("assistant", "A1")]
    assert history.to_messages() == [
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "U3"},
        {"role": "assistant", "content": "A3"},
    ]


def test_project_metadata_is_empty_transitions() -> None:
    """
    GIVEN an empty ProjectMetadata instance and one populated with any field
    WHEN is_empty is evaluated
    THEN it returns True only when all six fields are unset or empty.
    """
    assert ProjectMetadata().is_empty() is True
    assert ProjectMetadata(explicit_constraints=["SOC 2 Type II"]).is_empty() is False
    assert ProjectMetadata(rejected_options=["Microservices"]).is_empty() is False


def test_project_metadata_merge_with_overwrites_scalars_and_deduplicates_lists() -> None:
    """
    GIVEN a base ProjectMetadata and an incremental MetadataExtractionResult update
    WHEN merge_with combines them
    THEN non-empty scalars overwrite prior values and list fields merge case-insensitively.
    """
    base = ProjectMetadata(
        project_name="MyProject",
        assumed_team_size=3,
        mentioned_technologies=["Rails", "PostgreSQL"],
        agreed_scope="MVP phase 1",
        explicit_constraints=["GDPR"],
        rejected_options=["GraphQL"],
    )
    update = MetadataExtractionResult(
        project_name="MyProject Pro",
        assumed_team_size=5,
        mentioned_technologies=["rails", "React"],
        agreed_scope=None,
        explicit_constraints=["gdpr", "SOC 2"],
        rejected_options=["graphql", "Microservices"],
    )

    merged = base.merge_with(update)

    assert merged == ProjectMetadata(
        project_name="MyProject Pro",
        assumed_team_size=5,
        mentioned_technologies=["Rails", "PostgreSQL", "React"],
        agreed_scope="MVP phase 1",
        explicit_constraints=["GDPR", "SOC 2"],
        rejected_options=["GraphQL", "Microservices"],
    )


@pytest.mark.parametrize(
    ("base", "update", "expected"),
    [
        (
            ProjectMetadata(
                mentioned_technologies=["Rails", "PostgreSQL"],
                explicit_constraints=["GDPR", "SOC 2"],
                rejected_options=["Microservices"],
            ),
            MetadataExtractionResult(
                mentioned_technologies=["Node"],
                retracted_constraints=["gdpr"],
                retracted_options=["microservices"],
                rejected_options=["Rails"],
            ),
            ProjectMetadata(
                mentioned_technologies=["PostgreSQL", "Node"],
                explicit_constraints=["SOC 2"],
                rejected_options=["Rails"],
            ),
        ),
        (
            ProjectMetadata(
                mentioned_technologies=["Rails", "PostgreSQL"],
            ),
            MetadataExtractionResult(
                mentioned_technologies=["Node", "rails"],
                rejected_options=["rails"],
            ),
            ProjectMetadata(
                mentioned_technologies=["PostgreSQL", "Node"],
                rejected_options=["rails"],
            ),
        ),
        (
            ProjectMetadata(
                mentioned_technologies=["PostgreSQL"],
                rejected_options=["GraphQL"],
            ),
            MetadataExtractionResult(
                mentioned_technologies=["graphql"],
            ),
            ProjectMetadata(
                mentioned_technologies=["PostgreSQL", "graphql"],
                rejected_options=[],
            ),
        ),
    ],
    ids=[
        "explicit_retracted_lists",
        "rejected_option_prunes_active_and_incoming_technology",
        "readopted_technology_prunes_rejected_option",
    ],
)
def test_project_metadata_merge_with_retractions_and_cross_pruning(
    base: ProjectMetadata,
    update: MetadataExtractionResult,
    expected: ProjectMetadata,
) -> None:
    """
    GIVEN base ProjectMetadata and a MetadataExtractionResult containing retractions or conflicting choices
    WHEN merge_with applies the update
    THEN retracted entries are removed and conflicting technologies/rejections are pruned consistently.

    :param base: Existing session ProjectMetadata state.
    :param update: Incremental MetadataExtractionResult delta.
    :param expected: Expected resulting ProjectMetadata state.
    """
    assert base.merge_with(update) == expected


def test_session_initializes_unique_ids_and_empty_state() -> None:
    """
    GIVEN two newly instantiated Session objects
    WHEN inspected after creation
    THEN each has a distinct UUID v4 identifier, empty history, and empty metadata.
    """
    first = Session()
    second = Session()

    assert first.session_id != second.session_id
    assert first.history.to_messages() == []
    assert first.metadata.is_empty() is True
