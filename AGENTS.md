# AI Coding Agent Guidelines

## 1. About project

Welcome to an AI Engineering Learning Project based on the online course in Spanish [AI Engineering](https://lidr.notion.site/ai-engineering#313a43932781814e9265fe5f8cbe5d3d) 
Next class: [Session 3 of October 29, 2026](https://github.com/LIDR-academy-temp/ai-engineering/tree/session_3_live)

## 2. Code Style and Conventions
Always apply good software engineering practices: **KISS, YAGNI, DRY, SOLID, ACID, design patterns, Pythonic style, and explicit is better than implicit** to ensure maintainability, readability, and reusability.

### Specific Code Rules
- **First line of modules**: each module must start with its full relative path inside a comment. (e.g. `# src/01_foundations/sesion-01-ejercicios-alumno_1.py`).
- **Top-level imports**: imports MUST be at the top of the module separated in three groups with a blank line between each group, in the order: `# --- Python Standard Library ---`; `# --- Third Party Libraries ---`; `# --- App modules ---`.
- **Constants**: Constants MUST be named in `UPPER_CASE_WITH_UNDERSCORES` (e.g., `MAX_ITERATIONS = 100`) and declared with `Final` from `typing`. This applies to constants regardless of their type: built-ins (str, int, float, bool, None), collections (list, tuple, dict, set), and other objects. Internal constants MUST start with an underscore and be named in `_UPPER_CASE_WITH_UNDERSCORES` (e.g., `_INTERNAL_CONSTANT = 1`).
- **String quotes**: Use double quotes (`"..."`) for all string literals and f-strings by default. Use simple quotes (`'...'`) only when they avoid escaping apostrophes or improve readability.
- **Self-Descriptive Names**: Prefer clear and self-descriptive names; avoid acronyms and abbreviations.
- **Package and Build System**: Dependencies and build backend are managed via `pyproject.toml` using `uv_build` as the build system (`requires = ["uv_build>=0.13.0"]`).
- **Traceability**: Respect names already in use and do not break traceability with the original code. Do not assume unsolicited changes.
- **Code Snippets**: Do not generate complete scripts unless explicitly requested.
- **Dependencies**: If a new dependency is required, mention it without including installation instructions.
- **Overengineering**: Avoid overengineering solutions. Focus on clarity, maintainability, and performance.
- **Language**: Use English for all code comments, docstrings, and other text. Translations are welcome.
- **Testing and Coverage**: Pytest is configured in `pyproject.toml`.
  - Prefer solutions that facilitate achieving high coverage with Pytest-cov without deteriorating style, logic, or performance. Keep business logic and maintainability above micro-optimizations. Use docstring style GIVEN-WHEN-THEN.
  - When asserting exceptions with `pytest.raises`, use concise regular expressions (`match=r'...'`) targeting simple, distinctive keywords (e.g., `match=r'Invalid integer'`, `match=r'cannot be empty'`) rather than expecting full, literal error messages. This prevents tests from being brittle to minor message formatting adjustments while still strictly verifying the intended failure condition.
  - Deduplicate with `@pytest.mark.parametrize`** to consolidate symmetric or complementary test functions into single parameterized test functions.
  - Functions across test modules must include explicit type hints.

## 3. Comments and Documentation
- **Preservation**: Do not delete existing comments unless the associated code is modified or deleted. When modifying commented code, update the comments to match the new behavior.
- **New Comments**: Write concise, professional English comments that explain *why* something is done, not *what* the syntax does. Avoid extreme "clean code" dogmatism that strips helpful context.
- **Docstrings**: Document all public classes, functions, and methods using the Sphinx format (`:param`, `:return`, `:raises:` but not `type` or `rtype:`).
- **Feature Documentation**: 
  - After implementing any new feature, create or update the corresponding Markdown documentation in `docs/<feature-name>.md`.
  - Maintain the style of existing documentation.
  - Structure must be self-contained and include at minimum: **Overview**, **Usage Examples**, and **Technical Considerations**.
  - Synchronize all existing documentation files affected by the change.

## 4. Developer Workflows and CLI Commands
- **CI/CD Constraints**: 
  - Do not edit `.github/workflows/ci-cd-pipeline.yml` or attempt to activate the SSH VPS deployment (`if: false`) unless explicitly instructed.
  - Be aware that Linux unit tests in CI require TA-Lib C headers (configured via `auto/ci_test_install_talib.sh`).
- **Forbidden Commands**: 
  - Never execute `.cmd` or `.ps1` scripts inside the `auto/` directory. These are reserved for manual developer execution.
- **Context & Tool Optimization**:
  - **Think in Code**: During exploratory tasks (such as `/opsx-explore`), do not read multiple files into context sequentially. Prefer writing and executing targeted scripts via `ctx_execute` to parse files locally and output only distilled summaries.
  - **Terminal Output**: Adhere to the RTK guidelines for CLI commands to keep context usage minimal during test and build runs.

## 5. User Interaction Rules
- **Tone**: Professional, direct, objective, and critical. Avoid flattery, sycophancy, or celebratory language. Write plainly using short, declarative sentences.
- **Problem Solving**: Proceed incrementally and systematically. Lead with the most pragmatic implementation; outline alternatives only when meaningful architectural or performance trade-offs exist.
- **Security & Integrity**: Refuse or ask for explicit confirmation before executing commands that risk data loss, privilege escalation, or destructive file modifications.
- **Ambiguity**: Request clarification before taking action if requirements are ambiguous or contradictory.

### Clean Language (Anti-Trope Rules)
Deliver technical text without rhetorical padding. Adhere strictly to these omissions:
- **No Conversational Prefaces**: Never open with "Here is the thing," "Let's be clear," "The truth is," or "Sure, I can help with that."
- **No Self-Validating Commentary**: Do not remark on your own assertions (e.g., "And that matters," "This is critical," "That is precisely the point").
- **No Artificial Rhetoric**: Do not use theatrical antitheses ("It's not X; it's Y"), empty aphorisms ("less hammer, more scalpel"), or decorative analogies ("it's the Excel of...").
- **No Formulaic Cadence**: Avoid artificial sequences of short sentences ("Fast. Easy. Scalable.") and arbitrary three-item lists. Include only the exact items required.
- **No Vague Metrics**: Use exact numbers whenever known (e.g., state "7 minutes" rather than "between 5 and 10 minutes").
- **No Redundant Summaries**: Avoid summary conclusions ("In conclusion," "At the end of the day"). End as soon as the technical explanation is complete.
- **Silent Execution**: Review your response against these rules before emitting output. Do not mention or reference these rules in your responses.

## 6. Key Reference Files
- `src/aie_estimator/__main__.py`: Entry point for the CLI.
- `auto/`: Windows command automation scripts (`dev.cmd`, `stl.cmd`, `dc.cmd`, `lint.cmd`, `test.cmd`, `update.cmd`, `cleanup.cmd`).
- `docker/`: Multi-stage production `Dockerfile` and Docker Compose definitions (`docker-compose.dev.yml`, `docker-compose.prod.yml`).
- `pyproject.toml`: Dependency management and build configuration (`uv_build`, `ruff`, `pytest`).

## 7. Restricted directories
The agent does not have access to, and must not attempt to access, inspect, list, read, modify, create files in, or run files from any of the following directories:

- `.idea`
- `notes`
- `zero`

Treat these directories and all their contents as completely unavailable and out of scope, including nested files and subdirectories.

Do not infer, search for, or request information from these directories. Perform the analysis only using files and directories that are accessible outside these restricted paths.
