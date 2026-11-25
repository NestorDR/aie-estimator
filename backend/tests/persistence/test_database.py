# tests/persistence/test_database.py

# --- Python Standard Library ---
from unittest.mock import AsyncMock, MagicMock, patch

# --- Third Party Libraries ---
from pydantic import SecretStr
import pytest

# --- App modules ---
from aie_estimator.persistence.database import close_pool, open_pool
from aie_estimator.settings import Environment, Settings


@pytest.mark.anyio
async def test_open_pool_raises_when_persistence_unconfigured() -> None:
    """
    GIVEN Settings with postgres_password set to None
    WHEN open_pool is awaited
    THEN a ConnectionError is raised without instantiating AsyncConnectionPool.
    """
    settings = Settings(app_env=Environment.TEST, postgres_password=None)
    with (
        patch("aie_estimator.persistence.database.AsyncConnectionPool") as mock_pool_cls,
        pytest.raises(ConnectionError, match=r"not configured"),
    ):
        await open_pool(settings)

    mock_pool_cls.assert_not_called()


@pytest.mark.anyio
async def test_open_pool_constructs_opens_and_returns_pool() -> None:
    """
    GIVEN configured PostgreSQL settings and a healthy AsyncConnectionPool
    WHEN open_pool is awaited
    THEN AsyncConnectionPool receives kwargs with open=False, open(wait=True) is awaited, and the pool is returned.
    """
    settings = Settings(
        app_env=Environment.TEST,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_user="estimator",
        postgres_password=SecretStr("secret"),
        postgres_db="estimator",
    )
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock(return_value=None)

    with patch("aie_estimator.persistence.database.AsyncConnectionPool", return_value=mock_pool) as mock_pool_cls:
        result = await open_pool(settings)

    call_kwargs = mock_pool_cls.call_args.kwargs
    assert call_kwargs["kwargs"] == {
        "host": "localhost",
        "port": 5432,
        "user": "estimator",
        "password": "secret",
        "dbname": "estimator",
    }
    assert call_kwargs["open"] is False
    mock_pool.open.assert_awaited_once_with(wait=True)
    assert result is mock_pool


@pytest.mark.anyio
async def test_open_pool_closes_on_failure_and_raises_connection_error() -> None:
    """
    GIVEN an AsyncConnectionPool whose open() call fails
    WHEN open_pool is awaited
    THEN the pool is closed and ConnectionError is raised.
    """
    settings = Settings(app_env=Environment.TEST, postgres_password=SecretStr("pass"))
    mock_pool = MagicMock()
    mock_pool.open = AsyncMock(side_effect=OSError("Connection refused"))
    mock_pool.close = AsyncMock(return_value=None)

    with (
        patch("aie_estimator.persistence.database.AsyncConnectionPool", return_value=mock_pool),
        pytest.raises(ConnectionError, match=r"Failed to open"),
    ):
        await open_pool(settings)

    mock_pool.close.assert_awaited_once()


@pytest.mark.anyio
async def test_close_pool_handles_none_and_active_pool() -> None:
    """
    GIVEN either None or an active AsyncConnectionPool
    WHEN close_pool is awaited
    THEN None is ignored cleanly and an active pool has close() awaited once.
    """
    await close_pool(None)

    mock_pool = MagicMock()
    mock_pool.close = AsyncMock(return_value=None)
    await close_pool(mock_pool)
    mock_pool.close.assert_awaited_once()
