:: backend/auto/lint.cmd
:: Purpose: Preview and apply Ruff lint fixes using the version pinned in uv.lock.
:: Usage: auto\lint.cmd (run from the backend directory)

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Require invocation from the backend directory
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the backend directory: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

:: Preview suggested fixes without modifying files
uv run --locked --group dev ruff check . --diff

:: Apply autofixable lint fixes in place
uv run --locked --group dev ruff check . --fix
set "_EXIT_CODE=%ERRORLEVEL%"

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
