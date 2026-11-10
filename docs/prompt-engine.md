# Prompt Template Engine

## Overview

The prompt template engine in [`src/aie_estimator/prompts/loader.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/prompts/loader.py) externalizes LLM prompt composition from Python service logic into versioned Jinja2 templates under [`src/aie_estimator/prompts/<use_case>/<version>/`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/prompts/estimation/v1/):
- `system.j2`: System instructions defining consultant role, hourly rates, format conditionals, JSON constraints, and dynamic inclusion of reference examples.
- `user.j2`: User message block wrapping the project description in `<project_description>` XML tags.
- `examples.j2`: Canonical few-shot reference estimations used as calibration anchors.

Prompt versions (`v1/`, `v2/`) iterate independently behind stable REST API endpoints (`/api/v1`). The active production prompt version is configured via `active_prompt_version` in [`src/aie_estimator/settings.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/settings.py).

---

## Usage Examples

### 1. Rendering Estimation Prompts

```python
from aie_estimator.prompts.loader import render_estimation_prompt
from aie_estimator.schemas.estimations import EstimationRequest

request = EstimationRequest(
    description="A microservices platform for real-time fleet telematics.",
    project_type="web_saas",
    detail_level="detailed",
    output_format="phases_table",
)

# Render active default version (configured in settings)
system_prompt, user_prompt = render_estimation_prompt(request)

# Programmatic version override for evals and benchmarking
system_v2, user_v2 = render_estimation_prompt(request, version="v2")
```

### 2. Listing and Validating Prompt Versions

```python
from aie_estimator.prompts.loader import list_prompt_versions, validate_prompt_version

# Discovers complete suites containing system.j2, user.j2, and examples.j2
available = list_prompt_versions("estimation")  # ['v1']
validated_version = validate_prompt_version("v1")
```

---

## Technical Considerations

1. **Strict Undefined Variables**: The Jinja2 environment in [`src/aie_estimator/prompts/loader.py`](file:///c:/Development/Repos/ai-engineering/aie-estimator/src/aie_estimator/prompts/loader.py) uses `StrictUndefined`, halting immediately with `UndefinedError` if any template context variable is missing.
2. **Whitespace Trimming**: Configured with `trim_blocks=True` and `lstrip_blocks=True` to strip control block newlines, ensuring deterministic whitespace across model providers.
3. **Dynamic Version Ingestion**: `system.j2` includes canonical calibration examples via dynamic version evaluation (`{% include "estimation/" ~ version ~ "/examples.j2" %}`), preventing hardcoded path coupling when authoring new versions.
4. **Completeness Validation**: `list_prompt_versions()` verifies that all three required template files (`system.j2`, `user.j2`, `examples.j2`) exist, preventing runtime `TemplateNotFound` errors on partially drafted versions.
