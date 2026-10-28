:: auto\dev.cmd
:: Purpose: Start dev server for fastapi
:: Usage: auto\dev.cmd (run from the project root directory)

uv run aie-estimator

:: Alternatively: uv run uvicorn aie_estimator.server:server.app --reload