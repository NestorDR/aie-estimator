:: auto\dev.cmd
:: Purpose: Launch FastAPI backend in the current console (blocking)
:: Usage: auto\dev.cmd (run from the project root directory)

uv run aie-estimator

:: Alternatively: uv run uvicorn aie_estimator.server:server.app --reload
