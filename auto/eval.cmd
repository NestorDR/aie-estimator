:: auto/eval.cmd
:: Purpose: Run the Family 1-3 LLM evaluation harness against the Golden Dataset
:: Usage: auto/eval [flags] (run from the project root directory)
::  fast: auto/eval --limit 3 --llm-judge --output evals/baseline.json 2>&1 | Tee-Object -FilePath logs/eval_console.log
::  slow: auto/eval --runs 3  --llm-judge --output evals/baseline.json 2>&1 | Tee-Object -FilePath logs/eval_console.log
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

:run_eval
:: EVALUATION HARNESS
:: Entrypoint: uv run python -m evals.run
::
:: FLAGS EXPLANATION (passed through via %*):
:: --dataset PATH ........: path to a custom Golden Dataset JSON file (defaults to evals/golden_dataset.json)
:: --limit N .............: run only the first N cases from the Golden Dataset
:: --runs N ..............: repeat each case N times to compute Family 2 Coefficient of Variation (CV)
:: --llm-judge ...........: enable Family 3 structured LLM-as-Judge scoring (coherence, actionability, tier)
:: --judge-model MODEL ...: override the model or logical profile used by the Family 3 judge
:: --http BASE_URL .......: target a running HTTP server instead of the default in-process ASGI app
:: --output PATH .........: write the full JSON evaluation report to PATH (e.g. evals/baseline.json)
:: --help ................: display CLI usage and exit
::
:: EXAMPLES:
::   auto\eval --limit 3
::   auto\eval --runs 3 --llm-judge --output evals\baseline.json
::   auto\eval --http http://127.0.0.1:8000 --llm-judge
uv run python -m evals.run %* || goto command_failed

echo [SUCCESS] Evaluation run completed.
echo !DATE! !TIME!
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Evaluation run failed with exit code %_EXIT_CODE%.

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
