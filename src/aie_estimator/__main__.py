# src/aie_estimator/__main__.py

# --- Python Standard Library ---
from typing import Final

# --- Third Party Libraries ---
import structlog
import uvicorn

# --- App modules ---
from aie_estimator.settings import Environment, get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()


def main() -> None:
    """
    Main entry point for running the application with Uvicorn.
    """
    settings = get_settings()
    log.info(
        "starting_uvicorn_server",
        server_ip=settings.server_ip,
        server_port=settings.server_port,
        app_env=settings.app_env.value,
    )

    uvicorn.run(
        app="aie_estimator.server:server.app",
        host=settings.server_ip,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        reload=settings.app_env != Environment.PROD,
        reload_dirs=["src/aie_estimator"],
        log_config=None,
    )


if __name__ == "__main__":
    main()
