:: auto/test.cmd
:: Purpose: Run all tests in the project and clean up temporary test artifacts
:: Usage: auto\test.cmd [target_test_script.py]

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Require invocation from this repository's root directory.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the project root: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

:: Set the virtual environment for this test run.
:: e.g. to separate environment: set VIRTUAL_ENV=C:\Development\VirtualEnvs\aie-estimator-3.14.8-uv-env
set VIRTUAL_ENV=.venv

:: Run pytest with cache clearing, temporary basetemp, and branch coverage reporting:
:: %*: Forwards all optional script arguments (target test file, -k filter, etc.).
:: --locked: Require uv.lock to match pyproject.toml; do not update the lockfile.
:: --active: Use the active virtual environment, set above as .venv.
:: --cache-clear: Clear pytest internal cache prior to running tests.
:: --basetemp: Route temporary fixture directories to the system TEMP folder.
:: --cov=streamlit_app: Measure line and branch coverage across the Streamlit UI module.
:: --cov-fail-under=85: Require overall test coverage of at least 85%.
uv run --locked --active pytest %* --cache-clear --basetemp="%TEMP%\pytest_runner" --cov=streamlit_app --cov-fail-under=85
set _EXIT_CODE=%ERRORLEVEL%

:: Post-test cleanup of project root artifacts
echo Cleaning up temporary test artifacts...

:: Remove .pytest_cache if created locally
if exist "%PROJECT_ROOT%\.pytest_cache" rmdir /s /q "%PROJECT_ROOT%\.pytest_cache"

:: Remove coverage reports (if using pytest-cov)
if exist "%PROJECT_ROOT%\.coverage" del /f /q "%PROJECT_ROOT%\.coverage"
if exist "%PROJECT_ROOT%\htmlcov" rmdir /s /q "%PROJECT_ROOT%\htmlcov"
if exist "%PROJECT_ROOT%\.coverage.*" del /f /q "%PROJECT_ROOT%\.coverage.*"

:end
:: Restore the original VIRTUAL_ENV and exit with the original pytest exit code
endlocal & exit /b %_EXIT_CODE%