:: auto/inspect-budgets.cmd
:: Purpose: Inspect cleaned, GDPR-pseudonymized Session 6 budgets as JSON on stdout (read-only, in memory)
:: Usage: auto\inspect-budgets [flags] (run from the project root directory)
::   default: auto\inspect-budgets
::   pipe:    auto\inspect-budgets > cleaned_budgets.json
@echo off

:: setlocal preserves the original environment variables; endlocal restores them before exit
:: Enable `delayed expansion` to handle environment variables correctly within blocks
setlocal ENABLEDELAYEDEXPANSION
set "_EXIT_CODE=0"
echo !DATE! !TIME!>&2

:: Require invocation from this repository's root directory.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "!CURRENT_DIR!"=="!PROJECT_ROOT!" (
    echo Error: Run this script from the project root: "!PROJECT_ROOT!">&2
    set "_EXIT_CODE=1"
    goto end
)

:run_inspect
:: CLEANED BUDGET JSON INSPECTOR
:: Entrypoint: uv run python -m aie_estimator.ingestion.inspect_budgets
::
:: FLAGS EXPLANATION (passed through via %*):
:: --no-pii ..............: skip PII pseudonymization and print raw cleaned budgets
:: --help ................: display CLI usage and exit
::
:: EXAMPLES:
::   auto\inspect-budgets
::   auto\inspect-budgets --no-pii
::   auto\inspect-budgets > cleaned_budgets.json
uv run python -m aie_estimator.ingestion.inspect_budgets %* || goto command_failed

echo [SUCCESS] Budget inspection completed.>&2
echo !DATE! !TIME!>&2
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Budget inspection failed with exit code %_EXIT_CODE%.>&2

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
