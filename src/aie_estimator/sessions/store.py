# src/aie_estimator/sessions/store.py

# --- Python Standard Library ---
from datetime import UTC, datetime
from typing import Final

# --- App modules ---
from aie_estimator.sessions.models import DEFAULT_MAX_TURNS, ConversationHistory, Session

DEFAULT_SESSION_TTL_SECONDS: Final[int] = 86_400


class SessionNotFoundError(LookupError):
    """Raised when a requested conversational session does not exist or has expired."""

    def __init__(self, session_id: str) -> None:
        """
        Initialize the error with the missing or expired session identifier.

        :param session_id: Identifier that failed resolution.
        """
        super().__init__(f"Session '{session_id}' not found.")
        self.session_id: Final[str] = session_id


class SessionStore:
    """
    In-memory store indexing active ``Session`` instances by ``session_id``.

    Sessions live in a process-local dictionary and are lost on restart. Multi-worker deployments
    (``uvicorn --workers > 1``) are unsupported until a shared Redis or PostgreSQL store replaces
    this class behind the same interface.
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        """
        Initialize the in-memory session store with sliding-window and TTL policies.

        :param max_turns: Maximum user/assistant pairs retained in each new session's history.
        :param ttl_seconds: Inactivity expiration threshold in seconds.
        """
        self._sessions: dict[str, Session] = {}
        self.max_turns: Final[int] = max_turns
        self.ttl_seconds: Final[int] = ttl_seconds

    def _is_expired(self, session: Session, now: datetime) -> bool:
        """
        Check whether a session has exceeded the inactivity TTL relative to ``now``.

        :param session: Session instance to inspect.
        :param now: Current UTC timestamp.

        :return: ``True`` if elapsed inactivity exceeds ``ttl_seconds``, otherwise ``False``.
        """
        return (now - session.last_accessed_at).total_seconds() > self.ttl_seconds

    def create(self) -> Session:
        """
        Sweep expired sessions and provision a new empty session with a UUID v4 identifier.

        :return: Newly created and registered ``Session`` instance.
        """
        now = datetime.now(UTC)
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_expired(session, now)
        ]
        for session_id in expired_ids:
            del self._sessions[session_id]

        session = Session(
            history=ConversationHistory(max_turns=self.max_turns),
            created_at=now,
            last_accessed_at=now,
        )
        self._sessions[session.session_id] = session
        return session

    def get_or_404(self, session_id: str) -> Session:
        """
        Retrieve an active session, evicting it lazily if expired, or raise ``SessionNotFoundError``.

        :param session_id: Target session UUID string.

        :return: Active ``Session`` instance with refreshed ``last_accessed_at``.
        :raises SessionNotFoundError: If ``session_id`` does not exist or has expired.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)

        now = datetime.now(UTC)
        if self._is_expired(session, now):
            del self._sessions[session_id]
            raise SessionNotFoundError(session_id)

        session.last_accessed_at = now
        return session
