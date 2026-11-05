# Prompt Template Engine

## Overview

The prompt template engine (`src/aie_estimator/prompts/`) externalizes LLM prompt composition from Python service logic into versioned Jinja2 templates. It structures prompt assembly into isolated, testable software artifacts with strict variable validation, explicit whitespace trimming, and versioning.

Templates are partitioned by use case and version under `src/aie_estimator/prompts/<use_case>/<version>/`:
- `system.j2`: System prompt/instruction with role, scope, hourly billing rates, conditional blocks for `output_format` and `detail_level`, output JSON schema constraints, and dynamic inclusion of canonical examples.
- `user.j2`: User message block wrapping the project description in `<project_description>` XML tags and declaring the project type.
- `examples.j2`: Canonical few-shot reference project estimations used as calibration anchors.

### Template Layout and Multi-Version Coexistence

```
src/aie_estimator/prompts/
├── __init__.py
├── loader.py
└── estimation/
    ├── v1/                <- Stable production prompt
    │   ├── system.j2
    │   ├── user.j2
    │   └── examples.j2
    └── v2/                <- Experimental / next prompt candidate (complete suite)
        ├── system.j2
        ├── user.j2
        └── examples.j2
```

---

## Decoupled Prompt and API Versioning

Prompt iterations and HTTP REST contracts evolve at fundamentally different rates and lifecycles:
- **HTTP REST APIs (`/api/v1`)**: Transport-level contracts (URLs, HTTP status codes, request/response models). Bumping to `/api/v2` requires client migrations and occurs very slowly (months/years).
- **Prompt Versions (`v1/`, `v2/`)**: AI prompt engineering artifacts (role framing, delimiters, calibration examples). These iterate rapidly (days/weeks) behind stable API routes.

Prompt versions are completely decoupled from router path prefixes:
1. **Configuration**: The active production version is configured via `active_prompt_version: str = "v1"` in `Settings` (`ACTIVE_PROMPT_VERSION=v1` environment variable).
2. **Resolution**: `resolve_prompt_version(version=None)` defaults to `Settings.active_prompt_version` when no explicit version is passed.
3. **Internal Evals**: Evaluation harnesses and benchmarks can pass `version="v2"` directly into `render_estimation_prompt` without touching HTTP routing or creating dummy REST endpoints.

---

## Usage Examples

### 1. Rendering a Prompt for Estimation (Default Version)

```python
from aie_estimator.prompts.loader import render_estimation_prompt
from aie_estimator.schemas.estimations import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)

request = EstimationRequest(
    description="A microservices platform for real-time fleet telematics.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.DETAILED,
    output_format=OutputFormat.PHASES_TABLE,
)

# Renders using the active configured version (e.g. v1)
system_prompt, user_prompt = render_estimation_prompt(request)

print(system_prompt)  # Formatted instructions with table layout & examples
print(user_prompt)  # Isolated project description within XML tags
```

### 2. Programmatic Version Override (for Evals and Benchmarking)

```python
# Evaluates an experimental prompt version candidate
system_v2, user_v2 = render_estimation_prompt(request, version="v2")
```

### 3. Listing and Validating Prompt Versions

```python
from aie_estimator.prompts.loader import list_prompt_versions, validate_prompt_version

# Discovers only complete version suites (containing system.j2, user.j2, examples.j2)
available_versions = list_prompt_versions("estimation")
print(available_versions)  # ['v1']

# Validates target version exists and is complete; raises ValueError if invalid
validated = validate_prompt_version("v1")
```

### 4. Invoking the Estimation Endpoint via HTTP

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A microservices platform for real-time fleet telematics.",
    "project_type": "web_saas",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }'
```

The response returns free-form estimation text along with the prompt template version and telemetry:

```json
{
  "text": "## Project Estimation\n...",
  "prompt_version": "v1",
  "model": "gpt-6-luna",
  "provider": "openai",
  "cost_usd": 0.0024,
  "usage": {
    "input_tokens": 420,
    "output_tokens": 280,
    "total_tokens": 700,
    "preprocessing_input_tokens": 0,
    "preprocessing_output_tokens": 0
  }
}
```

### 5. Handling Missing Context Variables

Because the Jinja2 environment enforces `StrictUndefined`, omitted parameters raise immediately:

```python
from jinja2.exceptions import UndefinedError
from aie_estimator.prompts.loader import _ENV

template = _ENV.get_template("estimation/v1/system.j2")
try:
    template.render(description="Valid description", project_type="web_saas")
except UndefinedError as exc:
    print(f"Template rendering halted: {exc}")
```

---

## Technical Considerations

### 1. Jinja2 Environment Configuration
- `undefined=StrictUndefined`: Halts execution immediately on missing context variables, preventing incomplete prompts from reaching LLM providers.
- `trim_blocks=True` and `lstrip_blocks=True`: Eliminates template-tag indentation and trailing newlines, ensuring predictable whitespace across all model providers.
- `keep_trailing_newline=True`: Preserves natural prompt boundaries.

### 2. Dynamic Relative Template Inclusion
The Jinja2 loader root is anchored to `src/aie_estimator/prompts/`. Modular includes inside `system.j2` dynamically reference the rendering version via context parameterization:
```jinja
{% include "estimation/" ~ version ~ "/examples.j2" %}
```
This guarantees that copying or creating new prompt versions (e.g. `v2/`) automatically references the corresponding version's examples without hardcoded path cross-talk or manual template editing.

### 3. Completeness Validation
A version directory is only recognized as valid if all three required files (`system.j2`, `user.j2`, and `examples.j2`) are present. Incomplete prompt drafts under development are ignored by the loader until complete, preventing runtime `TemplateNotFound` 500 errors.
