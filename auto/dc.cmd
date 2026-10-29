:: auto/dc.cmd
:: Purpose: Start  the Docker Compose environment for a specific deployment environment: dev, prod
:: Usage: auto/dc <dev|prod> (run from the project root directory)
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

:: Capture 1st argument
set TARGET_DEPLOY=%1
if "!TARGET_DEPLOY!"=="" (
    echo Parameter is required. Usage: %0 ^<target_deploy^>
    echo Valid values for target_deploy: dev, prod
    echo Example: %0 dev
    set "_EXIT_CODE=2"
    goto end
)

:: Since `docker compose` is case-sensitive, force `TARGET_DEPLOY` to lowercase to match the
:: `docker-compose.*.yml` and `.env.*` filenames, ensuring consistent resolution regardless of user input casing.
:: /i : Case-insensitive comparison
if /i "!TARGET_DEPLOY!"=="dev" (
    set TARGET_DEPLOY=
    set "TARGET_DEPLOY=dev"
) else if /i "!TARGET_DEPLOY!"=="prod" (
    set TARGET_DEPLOY=
    set "TARGET_DEPLOY=prod"
) else (
    echo Invalid target. Valid values are: dev, prod.
    set "_EXIT_CODE=2"
    goto end
)

:: This wrapper accepts one target only; reject extra arguments for now.
if not "%2"=="" (
    echo Unexpected extra argument. Usage: %0 ^<dev^|prod^>
    set "_EXIT_CODE=2"
    goto end
)

:compose
:: DOCKER-COMPOSE
:: Visit: https://docs.docker.com/reference/cli/docker/compose/up/

:: FLAGS EXPLANATION:
:: Only Build
:: build ..........: builds images before starting containers
:: --no-cache .....: to force rebuild
:: ^  : newline
:: && : starts next command if the previous one ended successfully
:: ```cmd
:: docker compose -f docker/docker-compose.!TARGET_DEPLOY!.yml build --no-cache ^
:: && docker compose ... up
:: ```

:: Build and launch all services as detached
:: --env-file .....: to inject the environment variables from a specific file
:: -f .............: to specify a particular compose file .yml
:: -p .............: to add an isolated specific project name
:: up .............: to raise the services defined in the .yml file
:: -d .............: to run in background (detached mode), visible in Docker Desktop
:: --build ........: reconstructs images if it detects changes in the build context or the Dockerfile
:: --force-recreate: ensures that containers are created again (stops and removes existing containers and creates new ones)
:: %* .............: for extra arguments
docker compose --env-file src/aie_estimator/.env.!TARGET_DEPLOY! -f docker/docker-compose.!TARGET_DEPLOY!.yml -p estimator-!TARGET_DEPLOY! up -d --build || goto command_failed

:: Check if the command was successful
echo [SUCCESS] !TARGET_DEPLOY! Environment is up.
echo !DATE! !TIME!

:: Restore the original environment variables and exit
goto end

:command_failed
set "_EXIT_CODE=%ERRORLEVEL%"
echo [ERROR] Docker Compose for !TARGET_DEPLOY! Environment failed to start.

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
