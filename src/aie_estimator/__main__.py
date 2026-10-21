# src/aie_estimator/__main__.py

# --- Third Party Libraries ---
import uvicorn

# --- App modules ---
from aie_estimator.server import server
from aie_estimator.settings import Environment, settings


def main():
    """Main entry point for running the application with Uvicorn."""
    server.logger.info(f"Starting Uvicorn server on {settings.server_ip}:{settings.server_port}. Environment: {settings.app_env}")

    uvicorn.run(
        app="aie_estimator.server:server.app",
        host=settings.server_ip,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
        reload=settings.app_env != Environment.PROD.value,
        reload_dirs=["src/aie_estimator"],
        log_config=settings.log_config,
    )


if __name__ == "__main__":
    main()
