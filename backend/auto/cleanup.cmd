:: backend/auto/cleanup.cmd
:: Purpose: Remove known Python/tool cache directories and optionally .venv
:: Usage: auto\cleanup.cmd [--include-venv] (run from the backend directory)

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Accept no argument, or the explicit option to remove `.venv` (preserve `.venv` by default)
set "INCLUDE_VENV=0"
if "%~1"=="" goto arguments_valid
if /i "%~1"=="--include-venv" (
    if not "%~2"=="" goto invalid_arguments
    set "INCLUDE_VENV=1"
) else goto invalid_arguments

:arguments_valid
:: Require invocation from the backend directory
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the backend directory: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

:: Remove only explicitly allowlisted cache directories beneath backend/; skip .venv unless opted in
python -c "import os,pathlib,shutil,stat,sys; root=pathlib.Path.cwd(); include_venv=sys.argv[1]=='1'; targets={'__pycache__','.pytest_cache','.ruff_cache','.mypy_cache','.pytest_tmp'}; onexc=lambda func,path,_exc: (os.chmod(path,stat.S_IWRITE),func(path)); venv=root/'.venv'; shutil.rmtree(venv,onexc=onexc) if include_venv and venv.is_dir() else None; exec('for current, directories, _ in os.walk(root):\n for name in directories[:]:\n  if name.casefold()==\'.venv\':\n   directories.remove(name)\n  elif name.casefold() in targets:\n   shutil.rmtree(pathlib.Path(current)/name,onexc=onexc)\n   directories.remove(name)')" "%INCLUDE_VENV%"
if errorlevel 1 goto command_failed

echo Backend cache cleanup completed.

if "%INCLUDE_VENV%"=="1" (
    echo.
    echo .venv was removed. From the backend directory, recreate it with:
    echo   uv sync --locked --group dev
    echo.
)

:: Restore the original environment variables and exit successfully.
goto end

:invalid_arguments
echo Usage: auto\cleanup.cmd [--include-venv]
set "_EXIT_CODE=2"
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
