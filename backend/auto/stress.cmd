:: auto/stress.cmd
:: Purpose: Run the Session 6 CAG stress-test harness and optional Markdown table generator
:: Usage: auto/stress [flags] (run from the project root directory)
::   dry-run: auto/stress --dry-run --output evals/stress/dry_run.csv --report-tables evals/stress/dry_tables.md
::      live: auto/stress --http http://127.0.0.1:8000 --repeats 1 --output evals/stress/results.csv --report-tables evals/stress/tables.md
@echo off
cls

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

:run_stress
:: STRESS EVALUATION HARNESS
:: Entrypoint: uv run python -m evals.stress.run
::
:: FLAGS EXPLANATION (passed through via %*):
:: --http BASE_URL ............: target a running HTTP server (defaults to in-process ASGI)
:: --scenarios NAMES ..........: comma-separated scenarios (default: growing,pivot,contradiction)
:: --attachment-sizes SIZES ...: comma-separated PDF sizes in KB (default: 0,5,20,50,100)
:: --repeats N ................: repetitions per sweep unit (default: 1)
:: --full-matrix ..............: run full Cartesian scenarios x attachment_sizes matrix
:: --model MODEL ..............: pinned model or profile alias (default: reasoner-fast)
:: --latency-budget-ms MS .....: per-turn latency budget in ms (default: 8000)
:: --cost-budget-usd USD ......: per-turn cost budget in USD (default: 0.05)
:: --timeout SECONDS ..........: HTTP request timeout in seconds (default: 180.0)
:: --output PATH ..............: output CSV path (default: evals/stress/results.csv)
:: --report-tables PATH .......: optional Markdown file path for generated summary and curve tables
:: --dry-run ..................: run in-process with deterministic mocked EstimationService
:: --help .....................: display CLI usage and exit
uv run python -m evals.stress.run %* || goto command_failed

echo [SUCCESS] Stress evaluation run completed.
echo !DATE! !TIME!
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Stress evaluation run failed with exit code %_EXIT_CODE%.

:end
endlocal & exit /b %_EXIT_CODE%
