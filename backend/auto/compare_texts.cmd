:: auto/compare_texts.cmd
:: Purpose: Compare cosine similarity between two texts using the configured RAG embedder
:: Usage: auto\compare_texts --1st "..." --2nd "..." (run from the backend directory)
@echo off

:: setlocal preserves the original environment variables; endlocal restores them before exit
:: Enable `delayed expansion` to handle environment variables correctly within blocks
setlocal ENABLEDELAYEDEXPANSION
set "_EXIT_CODE=0"
echo !DATE! !TIME!>&2

:: Require invocation from this repository's backend directory.
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "!CURRENT_DIR!"=="!PROJECT_ROOT!" (
    echo Error: Run this script from the backend directory: "!PROJECT_ROOT!">&2
    set "_EXIT_CODE=1"
    goto end
)

:run_compare
:: COSINE SIMILARITY CLI WRAPPER
:: Entrypoint: uv run python scripts/compare.py
::
:: FLAGS EXPLANATION (passed through via %*):
:: --1st .................: first text to embed and compare
:: --2nd .................: second text to embed and compare
:: --help ................: display CLI usage and exit
::
:: EXAMPLE:
::   auto\compare_texts --1st "Backend services" --2nd "API development"
uv run python scripts/compare.py %* || goto command_failed

echo [SUCCESS] Text comparison completed.>&2
echo !DATE! !TIME!>&2
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Text comparison failed with exit code %_EXIT_CODE%.>&2

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
