:: auto/migrate.cmd
:: Purpose: Apply pending versioned PostgreSQL SQL delta migrations
:: Usage: auto\migrate (run from the project root directory)
@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
:: Enable `delayed expansion` to handle environment variables correctly within blocks
setlocal ENABLEDELAYEDEXPANSION
set "_EXIT_CODE=0"
echo !DATE! !TIME!

:: Require invocation from this repository's root directory.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "!CURRENT_DIR!"=="!PROJECT_ROOT!" (
    echo Error: Run this script from the project root: "!PROJECT_ROOT!"
    set "_EXIT_CODE=1"
    goto end
)

:run_migrate
:: DATABASE MIGRATION RUNNER
:: Entrypoint: uv run python -m aie_estimator.persistence.migrate
::
:: Reads POSTGRES_* settings from .env.<APP_ENV> (default: dev) and applies
:: pending .sql scripts in src/aie_estimator/persistence/migrations/.
uv run python -m aie_estimator.persistence.migrate || goto command_failed

echo [SUCCESS] Database migrations completed.
echo !DATE! !TIME!
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Database migrations failed with exit code %_EXIT_CODE%.

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
