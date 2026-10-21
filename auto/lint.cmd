:: auto/lint.cmd
:: Purpose: Linter and autoformat.
:: :: Usage: auto\lint.cmd (run from the project root directory)

:: UVX: In most cases, executing a tool with `uvx` is more appropriate than installing the tool (https://docs.astral.sh/uv/concepts/tools/#execution-vs-installation)

:: Type checker and language server - ACTIVATE if you want the code to be stricter (good luck!)
:: DEACTIVATED uvx ty check .

:: Replace `uvx ruff` with `uv run --locked --group dev ruff` to use the Ruff version pinned in uv.lock.
:: Suggested fixes without changing files
uv run --locked --group dev ruff check . --diff
:: Apply the fixes shown before
uv run --locked --group dev ruff check . --fix

:: To request the latest version of Ruff and refresh the cache, use the @latest suffix:
:: uvx ruff@latest check . --fix

:: Formatter - ACTIVATE if you want to run a very aggressive formatter
:: DEACTIVATED uvx ruff format
