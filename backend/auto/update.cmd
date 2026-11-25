:: auto/update.cmd
:: Purpose: Update project dependencies to the latest allowed versions according to the restrictions in pyproject.toml,
::          and update the virtual environment to match the project's lock file (uv.lock).
:: Usage: auto\update.cmd (run from the project root directory)

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Require invocation from this repository's root directory
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the project root: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

:: Self-actualization and stop if it fails
uv self update || goto command_failed

:: Activate the project environment
:: e.g. to separate environment: set VIRTUAL_ENV=C:\Development\VirtualEnvs\radar-3.14.8-uv-env
set "VIRTUAL_ENV=.venv"
call "%VIRTUAL_ENV%\Scripts\activate.bat" || goto command_failed

:: Upgrade the project lock file to the latest allowed versions according to the restrictions in pyproject.toml
:: --upgrade  : tries to update all direct and transitive dependencies to the newest compatible version.
uv lock --upgrade || goto command_failed

:: Sync the virtual environment to the project's lock file (uv.lock)
:: --group dev: include the dev dependencies
:: --active   : target the active environment
uv sync --active --group dev || goto command_failed

:: Run python script freeze: to creates a list of all the installed packages in the virtual environment, along with their versions.
:: Previously %VIRTUAL_ENV%\Scripts\pip3.exe freeze > %PROJECT_FOLDER%\requirements.txt
@echo Freezing requirements for review purpose
del /q requirements.lock.tmp
uv pip freeze > requirements.lock.tmp
if errorlevel 1 (
    del /q requirements.lock.tmp 2>nul
    echo Error: Failed to freeze requirements; requirements.lock was not changed.
    set "_EXIT_CODE=1"
    goto end
)

move /y requirements.lock.tmp requirements.lock
if errorlevel 1 (
    del /q requirements.lock.tmp 2>nul
    echo Error: Could not replace requirements.lock.
    set "_EXIT_CODE=1"
)

goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%