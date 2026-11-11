# Prompt Template Engine

## Overview

The prompt template engine in [`src/aie_estimator/prompts/loader.py`](../src/aie_estimator/prompts/loader.py) externalizes LLM prompt composition from Python service logic into versioned Jinja2 templates under [`src/aie_estimator/prompts/<use_case>/<version>/`](../src/aie_estimator/prompts/estimation/v1/):
- `system.j2`: System instructions defining consultant role, hourly rates, format conditionals, JSON constraints, and dynamic inclusion of reference examples (`v2/` adds `<project_metadata>` injection and conversational authority rules).
- `user.j2`: User message block wrapping the project description in `<project_description>` XML tags.
- `examples.j2`: Canonical few-shot reference estimations used as calibration anchors.

Transactional estimations resolve `active_prompt_version` (default `"v1"`), while multi-turn sessions resolve `conversational_prompt_version` (default `"v2"`) from [`src/aie_estimator/settings.py`](../src/aie_estimator/settings.py).

---

## Usage Examples

### 1. Rendering Transactional and Conversational Prompts

```python
from aie_estimator.prompts.loader import render_conversational_prompt, render_estimation_prompt
from aie_estimator.schemas.estimations import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from aie_estimator.sessions.models import ProjectMetadata

request = EstimationRequest(
    description="A microservices platform for real-time fleet telematics.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.DETAILED,
    output_format=OutputFormat.PHASES_TABLE,
)

# Transactional estimation prompt (defaults to active_prompt_version="v1")
system_prompt, user_prompt = render_estimation_prompt(request)

# Conversational estimation prompt (defaults to conversational_prompt_version="v2")
conv_system, conv_user = render_conversational_prompt(
    description="Add OAuth2 SSO and audit logging.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.DETAILED,
    output_format=OutputFormat.PHASES_TABLE,
    metadata=ProjectMetadata(project_name="FleetOps"),
)
```

### 2. Listing and Validating Prompt Versions

```python
from aie_estimator.prompts.loader import list_prompt_versions, validate_prompt_version

available = list_prompt_versions("estimation")  # ["v1", "v2"]
validated_version = validate_prompt_version("v2")
```

---

## Technical Considerations

1. **Strict Undefined Variables**: The Jinja2 environment in [`src/aie_estimator/prompts/loader.py`](../src/aie_estimator/prompts/loader.py) uses `StrictUndefined`, halting immediately with `UndefinedError` if any template context variable is missing.
2. **Whitespace Trimming**: Configured with `trim_blocks=True` and `lstrip_blocks=True` to strip control block newlines, ensuring deterministic whitespace across model providers.
3. **Dynamic Version Inclusion**: `system.j2` includes canonical calibration examples via `{% include "estimation/" ~ version ~ "/examples.j2" %}`, while `estimation/v2/examples.j2` explicitly delegates to `estimation/v1/examples.j2` without duplicating examples.
4. **Per-Use-Case Completeness Validation**: `list_prompt_versions()` validates required files per use case (`system.j2`, `user.j2`, and `examples.j2` for `estimation`; `system.j2` and `user.j2` by default for other use cases), preventing runtime `TemplateNotFound` errors.
5. **Conditional `<project_metadata>` Injection**: `estimation/v2/system.j2` renders an empty-state note when `ProjectMetadata.is_empty()` is `True` and renders only populated fields among all six metadata attributes otherwise, avoiding placeholder noise and asserting authority over established facts.
