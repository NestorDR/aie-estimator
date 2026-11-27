# Prompt Template Engine

## Overview

The prompt template engine in [`loader.py`](../backend/src/aie_estimator/foundation/prompts/loader.py) externalizes LLM prompt composition into versioned Jinja2 templates under [`prompts/`](../backend/src/aie_estimator/foundation/prompts/):
- `estimation/v3/`: Tier-specific system templates (`developer.j2`, `pm.j2`, `executive.j2`) sharing `{% include %}` partials (`_project_metadata.j2`, `_conversation_summary.j2`, `_anchored_turns.j2`, `_scope_rates.j2`, `_rules.j2`, `_examples.j2`) plus `user.j2` (with optional `<critic_feedback>`).
- `critic/v1/`: Structured Critic audit templates (`system.j2`, `user.j2`).
- `metadata_extraction/v1/`: Post-turn `ProjectMetadata` delta and retraction extraction (`system.j2`, `user.j2`).
- `conversation_summary/v1/`: Cumulative rolling summarization of evicted turns (`system.j2`, `user.j2`).

---

## Usage Examples

```python
from aie_estimator.foundation.prompts.loader import (
    render_conversation_summary_prompt,
    render_tier_estimation_prompt,
)
from aie_estimator.domain.schemas.estimations import DetailLevel, OutputFormat, ProjectType
from aie_estimator.generation.conversation.models import Message, ProjectMetadata
from aie_estimator.generation.conversation.tier_resolver import Tier

system_prompt, user_prompt = render_tier_estimation_prompt(
    description="Add OAuth2 SSO and audit logging.",
    project_type=ProjectType.WEB_SAAS,
    detail_level=DetailLevel.DETAILED,
    output_format=OutputFormat.PHASES_TABLE,
    tier=Tier.PM,
    metadata=ProjectMetadata(project_name="FleetOps"),
    conversation_summary="Earlier turns established a 3-developer team.",
    anchored_turns=[Message(role="user", content="Scope is frozen under NDA.")],
)

sum_system, sum_user = render_conversation_summary_prompt(
    previous_summary="Initial discovery complete.",
    evicted=[Message(role="user", content="Use FastAPI and PostgreSQL.")],
)
```

---

## Technical Considerations

1. **Strict Undefined Variables**: The Jinja2 environment in [`loader.py`](../backend/src/aie_estimator/foundation/prompts/loader.py) uses `StrictUndefined`, halting immediately with `UndefinedError` if any template context variable is missing.
2. **Whitespace Trimming**: Configured with `trim_blocks=True` and `lstrip_blocks=True` to strip control block newlines, ensuring deterministic whitespace across model providers.
3. **Version Resolution**: [`resolve_prompt_version`](../backend/src/aie_estimator/foundation/prompts/loader.py) validates that `foundation/prompts/<use_case>/<version>` exists before rendering, raising `ValueError` otherwise.
4. **Conditional Session Context Injection**: `estimation/v3` partials inject `<project_metadata>`, `<conversation_summary>`, and `<anchored_turns>` only when present, keeping `ConversationHistory.to_messages()` strictly alternating `user`/`assistant` turns.
