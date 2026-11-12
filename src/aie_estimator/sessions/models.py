# src/aie_estimator/sessions/models.py

# --- Python Standard Library ---
from datetime import UTC, datetime
from typing import Final, Literal
from uuid import uuid4

# --- Third Party Libraries ---
from pydantic import BaseModel, Field

DEFAULT_MAX_TURNS: Final[int] = 6
_PROJECT_NAME_MAX_LENGTH: Final[int] = 120
_AGREED_SCOPE_MAX_LENGTH: Final[int] = 2000
_MAX_TEAM_SIZE: Final[int] = 50

Role = Literal["user", "assistant"]


def _merge_unique(existing: list[str], updates: list[str]) -> list[str]:
    """
    Merge two string lists preserving first-seen casing while deduplicating case-insensitively.

    :param existing: Current list of values.
    :param updates: Incoming values to append if not already present.

    :return: Deduplicated list combining existing and new values.
    """
    merged = list(existing)
    seen = {item.lower() for item in merged}
    for item in updates:
        key = item.lower()
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _remove_items(existing: list[str], removals: list[str]) -> list[str]:
    """
    Remove items from ``existing`` matching ``removals`` case-insensitively while preserving order.

    :param existing: Current list of values.
    :param removals: Items to remove regardless of casing.

    :return: Filtered list retaining original casing of remaining entries.
    """
    if not removals:
        return list(existing)
    removal_keys = {item.lower() for item in removals}
    return [item for item in existing if item.lower() not in removal_keys]


class Message(BaseModel):
    """Single user or assistant turn entry stored inside conversation history."""

    role: Role
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationHistory(BaseModel):
    """Bounded sliding window of alternating user and assistant messages."""

    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1)
    messages: list[Message] = Field(default_factory=list)

    def append(self, *, user: str, assistant: str) -> list[Message]:
        """
        Append one user/assistant pair and evict oldest overflow pairs from the front.

        :param user: User turn text.
        :param assistant: Assistant turn text.

        :return: Evicted messages in chronological order, or an empty list if within capacity.
        """
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        overflow = len(self.messages) - (self.max_turns * 2)
        if overflow <= 0:
            return []
        evicted = self.messages[:overflow]
        del self.messages[:overflow]
        return evicted

    def to_messages(self) -> list[dict[str, str]]:
        """
        Serialize stored history into LLM-ready role/content dictionaries without a system entry.

        :return: Ordered list of ``{"role": ..., "content": ...}`` dictionaries.
        """
        return [{"role": message.role, "content": message.content} for message in self.messages]


class ProjectMetadata(BaseModel):
    """Distilled project facts maintained independently of sliding-window history."""

    project_name: str | None = Field(default=None, max_length=_PROJECT_NAME_MAX_LENGTH)
    assumed_team_size: int | None = Field(default=None, ge=1, le=_MAX_TEAM_SIZE)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=_AGREED_SCOPE_MAX_LENGTH)
    explicit_constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """
        Check whether all metadata scalar and collection fields are unset or empty.

        :return: ``True`` if no project facts have been recorded yet, ``False`` otherwise.
        """
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
            and not self.explicit_constraints
            and not self.rejected_options
        )

    def merge_with(self, update: "MetadataExtractionResult") -> "ProjectMetadata":
        """
        Return a new ``ProjectMetadata`` applying scalar overwrites, additions, retractions, and pruning.

        Non-empty scalar fields overwrite existing values, ``rejected_options`` prune active
        technologies, ``retracted_options`` and newly adopted technologies un-reject prior
        rejections, ``retracted_constraints`` remove lifted constraints, and remaining
        additions merge via case-insensitive union.

        :param update: Incremental metadata extraction delta from the latest turn.

        :return: Merged ``ProjectMetadata`` instance.
        """
        rejected_keys = {item.lower() for item in update.rejected_options}
        # Explicit rejections take precedence over active technologies within the same turn.
        accepted_technologies = [item for item in update.mentioned_technologies if item.lower() not in rejected_keys]
        technologies = _merge_unique(
            _remove_items(self.mentioned_technologies, update.rejected_options),
            accepted_technologies,
        )
        rejected = _merge_unique(
            _remove_items(self.rejected_options, [*update.retracted_options, *accepted_technologies]),
            update.rejected_options,
        )
        constraints = _merge_unique(
            _remove_items(self.explicit_constraints, update.retracted_constraints),
            update.explicit_constraints,
        )

        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size if update.assumed_team_size is not None else self.assumed_team_size,
            mentioned_technologies=technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
            explicit_constraints=constraints,
            rejected_options=rejected,
        )


class MetadataExtractionResult(BaseModel):
    """Structured extraction delta capturing metadata updates and explicit retractions."""

    project_name: str | None = Field(default=None, max_length=_PROJECT_NAME_MAX_LENGTH)
    assumed_team_size: int | None = Field(default=None, ge=1, le=_MAX_TEAM_SIZE)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = Field(default=None, max_length=_AGREED_SCOPE_MAX_LENGTH)
    explicit_constraints: list[str] = Field(default_factory=list)
    rejected_options: list[str] = Field(default_factory=list)
    retracted_constraints: list[str] = Field(default_factory=list)
    retracted_options: list[str] = Field(default_factory=list)


class Session(BaseModel):
    """In-memory conversational estimation session container."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory = Field(default_factory=ConversationHistory)
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
