# src/aie_estimator/__main__.py

# --- Third Party Libraries ---
import uvicorn

# --- App modules ---
from aie_estimator.config import Environment, config
from aie_estimator.server import server


def main():
    """Main entry point for running the application with Uvicorn."""
    server.logger.info(f"Starting Uvicorn server on {config.server_ip}:{config.server_port}. Environment: {config.app_env}")

    uvicorn.run(
        app="aie_estimator.server:server.app",
        host=config.server_ip,
        port=config.server_port,
        log_level=config.log_level.lower(),
        reload=config.app_env != Environment.PROD.value,
        reload_dirs=["src/aie_estimator"],
        log_config=config.log_config,
    )


if __name__ == "__main__":
    main()
