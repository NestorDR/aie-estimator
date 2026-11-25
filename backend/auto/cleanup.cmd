:: auto/cleanup.cmd
:: Purpose: Remove repository cache directories and optionally .venv
:: Usage: auto/cleanup.cmd [--include-venv] (run from the project root directory)

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Accept no argument, or the explicit option to remove the project `.venv` (preserve `.venv` by default)
set "INCLUDE_VENV=0"
if "%~1"=="" goto arguments_valid
if /i "%~1"=="--include-venv" (
    if not "%~2"=="" goto invalid_arguments
    set "INCLUDE_VENV=1"
) else goto invalid_arguments

:arguments_valid
:: Require invocation from this repository's root directory
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the project root: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

:: Remove cache-named directories beneath the project root; skip .venv unless opted in
python -c "import os,pathlib,shutil,stat,sys; root=pathlib.Path.cwd(); include_venv=sys.argv[1]=='1'; onerror=lambda function,path,_: (os.chmod(path,stat.S_IWRITE),function(path)); venv=root/'.venv'; shutil.rmtree(venv,onerror=onerror) if include_venv and venv.is_dir() else None; exec('for current, directories, _ in os.walk(root):\n for name in directories[:]:\n  if name.casefold()==\'.venv\':\n   directories.remove(name)\n  elif \'cache\' in name.casefold():\n   shutil.rmtree(pathlib.Path(current)/name,onerror=onerror)\n   directories.remove(name)')" "%INCLUDE_VENV%"
if errorlevel 1 goto command_failed

:: Do not clear Ruff's user-level cache, which may be shared with other checkouts
:: uvx ruff clean || exit /b

echo Repository cache cleanup completed.

if "%INCLUDE_VENV%"=="1" (
    echo.
    echo .venv was removed. From the project root, recreate it with:
    echo   uv venv --python 3.14.8 --seed .venv
    echo   uv sync --group dev
    echo   uv pip install --python .venv\Scripts\python.exe https://github.com/cgohlke/talib-build/releases/download/v0.6.4/ta_lib-0.6.4-cp314-cp314-win_amd64.whl --no-cache-dir
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