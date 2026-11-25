# tests/sessions/test_store.py

# --- Python Standard Library ---
from datetime import timedelta

# --- Third Party Libraries ---
import pytest

# --- App modules ---
from aie_estimator.sessions.store import SessionNotFoundError, SessionStore


def test_session_store_create_and_get_or_404_refreshes_last_accessed_at() -> None:
    """
    GIVEN a SessionStore configured with max_turns=4 and ttl_seconds=60
    WHEN an active session is retrieved within TTL
    THEN the same session is returned and last_accessed_at advances.
    """
    store = SessionStore(max_turns=4, ttl_seconds=60)
    created = store.create()
    prior_access = created.last_accessed_at - timedelta(seconds=30)
    created.last_accessed_at = prior_access

    fetched = store.get_or_404(created.session_id)

    assert fetched is created
    assert fetched.history.max_turns == 4
    assert fetched.last_accessed_at > prior_access


def test_session_store_lazy_eviction_on_expired_lookup() -> None:
    """
    GIVEN a SessionStore entry whose last_accessed_at is older than ttl_seconds
    WHEN get_or_404 is called for the expired or missing identifier
    THEN the expired session is evicted and SessionNotFoundError is raised.
    """
    store = SessionStore(ttl_seconds=60)
    session = store.create()
    session.last_accessed_at -= timedelta(seconds=61)

    with pytest.raises(SessionNotFoundError, match=r"not found"):
        store.get_or_404(session.session_id)
    assert session.session_id not in store._sessions

    with pytest.raises(SessionNotFoundError, match=r"not found"):
        store.get_or_404("missing-id")


def test_session_store_create_sweeps_expired_sessions_without_disturbing_active() -> None:
    """
    GIVEN one expired session and one active session in SessionStore
    WHEN create() provisions a new session
    THEN the expired session is purged while the active and new sessions remain.
    """
    store = SessionStore(ttl_seconds=60)
    expired = store.create()
    expired.last_accessed_at -= timedelta(seconds=61)
    active = store.create()

    newest = store.create()

    assert expired.session_id not in store._sessions
    assert store.get_or_404(active.session_id) is active
    assert store.get_or_404(newest.session_id) is newest
