:: auto\dev.cmd
:: Purpose: Launch Streamlit in a separate window and run FastAPI in the current console
:: Usage: auto\dev.cmd (run from the project root directory)

:: Launch Streamlit UI asynchronously in a separate console window
start "Streamlit UI" uv run streamlit run streamlit_app.py

:: Run FastAPI backend in the current console (blocking)
uv run aie-estimator
:: Alternatively: uv run uvicorn aie_estimator.server:server.app --reload
