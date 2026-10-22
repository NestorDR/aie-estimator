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
:: e.g. to separate environment: set VIRTUAL_ENV=C:\Development\VirtualEnvs\radar-3.14.8-uv-env
set VIRTUAL_ENV=.venv

:: Run pytest with --cache-clear and route temporary fixture files to system TEMP.
:: --locked: Require uv.lock to match pyproject.toml; do not update the lockfile.
:: --active: Use the active virtual environment, set above as .venv.
uv run --locked --active pytest %1 --cache-clear --basetemp="%TEMP%\pytest_runner"
set _EXIT_CODE=%ERRORLEVEL%

:: Post-test cleanup of project root artifacts
echo Cleaning up temporary test artifacts...

:: Remove .pytest_cache if created locally
if exist "%PROJECT_ROOT%\.pytest_cache" rmdir /s /q "%PROJECT_ROOT%\.pytest_cache"

:: Remove coverage reports (if using pytest-cov)
if exist "%PROJECT_ROOT%\.coverage" del /f /q "%PROJECT_ROOT%\.coverage"
if exist "%PROJECT_ROOT%\htmlcov" rmdir /s /q "%PROJECT_ROOT%\htmlcov"
if exist "%PROJECT_ROOT%\.coverage.*" del /f /q "%PROJECT_ROOT%\.coverage.*"

:: Remove test price cache directories if left behind
if exist "%PROJECT_ROOT%\tests\cache" rmdir /s /q "%PROJECT_ROOT%\tests\cache"

:end
:: Restore the original VIRTUAL_ENV and exit with the original pytest exit code
endlocal & exit /b %_EXIT_CODE%