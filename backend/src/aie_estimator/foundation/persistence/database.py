# src/aie_estimator/foundation/persistence/database.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
from psycopg_pool import AsyncConnectionPool
import structlog

# --- App modules ---
from aie_estimator.settings import Settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger(__name__)

DEFAULT_MIN_POOL_SIZE: Final[int] = 1
DEFAULT_MAX_POOL_SIZE: Final[int] = 10
DEFAULT_POOL_TIMEOUT_SECONDS: Final[float] = 10.0


async def open_pool(settings: Settings) -> AsyncConnectionPool:
    """
    Create and open the asynchronous PostgreSQL connection pool, verifying connectivity.

    :param settings: Application settings containing single-source PostgreSQL configuration.

    :return: Opened AsyncConnectionPool instance.
    :raises ConnectionError: If PostgreSQL is unconfigured or the pool fails to open.
    """
    conn_kwargs = settings.get_database_kwargs()
    if conn_kwargs is None:
        raise ConnectionError("PostgreSQL is not configured.")

    pool = AsyncConnectionPool(
        kwargs=conn_kwargs,
        min_size=DEFAULT_MIN_POOL_SIZE,
        max_size=DEFAULT_MAX_POOL_SIZE,
        timeout=DEFAULT_POOL_TIMEOUT_SECONDS,
        open=False,
    )
    try:
        await pool.open(wait=True)
    except Exception as e:
        await pool.close()
        log.critical("database_pool_failed", error=repr(e), exc_info=e)
        raise ConnectionError(f"Failed to open PostgreSQL connection pool: {e!r}") from e

    log.info("database_pool_initialized", min_size=DEFAULT_MIN_POOL_SIZE, max_size=DEFAULT_MAX_POOL_SIZE)
    return pool


async def close_pool(pool: AsyncConnectionPool | None) -> None:
    """
    Close the asynchronous PostgreSQL connection pool cleanly if initialized.

    :param pool: Active AsyncConnectionPool instance or None when unconfigured.
    """
    if pool is None:
        return
    await pool.close()
    log.info("database_pool_closed")
