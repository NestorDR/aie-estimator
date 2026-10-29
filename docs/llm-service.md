# LLM Service Architecture and Provider Integration

## Overview

`src/aie_estimator/services/llm_service.py` orchestrates software effort estimations across **OpenAI** and **Anthropic** using Context-Augmented Generation (CAG), configurable input preprocessing, and exact-match Redis caching.

Provider routing and reasoning capability resolution are driven by `MODEL_CATALOG` (`src/aie_estimator/schemas/models.py`):

| Provider | API Endpoint | Retry Mechanism | Reasoning / Thinking Parameter |
|:---------|:-------------|:----------------|:-------------------------------|
| **OpenAI** | `client.responses.create` | Custom (`call_with_retry`, 3 attempts) | `reasoning={"effort": effort}` (`is_reasoning=True`) |
| **Anthropic** | `client.messages.create` | SDK built-in (`max_retries=2`) | `thinking={"type": "enabled", "budget_tokens": ...}` |

---

## Usage Examples

### 1. Generate Estimation (Default & Custom Options)

```python
from aie_estimator.services.llm_service import GenerationOptions, generate_estimation

# Default options (gpt-4o-mini, 3 markdown examples, no preprocessing)
result = generate_estimation("Client needs an e-commerce backend with Stripe integration...")

# Custom options
opts = GenerationOptions(
    preprocessing="two_phase",
    example_format="markdown",
    num_examples=3,
    use_examples=True,
    model="claude-haiku-4-5-20251001",
    max_tokens=4000,
    thinking_budget=2048,
)
result = generate_estimation("Raw transcript...", opts=opts)
```

### 2. Prompt Construction & Requirement Extraction

```python
from aie_estimator.services.llm_service import (
    GenerationOptions,
    build_system_prompt,
    extract_requirements,
)

system_prompt = build_system_prompt(
    example_format="json",
    num_examples=2,
    use_examples=True,
    inline_cleaning=True,
)

clean_reqs, prep_usage = extract_requirements(
    transcription="Raw meeting transcript...",
    opts=GenerationOptions(model="gpt-4o-mini"),
)
```

### 3. Direct Provider & Client Lifecycle Calls

```python
from aie_estimator.services.llm_service import (
    _call_anthropic,
    _call_openai,
    _reset_clients,
    get_anthropic_client,
    get_openai_client,
)

openai_res = _call_openai("Estimate project...", "System prompt", "gpt-4o-mini", max_tokens=2048)
anthropic_res = _call_anthropic("Estimate project...", "System prompt", "claude-haiku-4-5-20251001", max_tokens=4000)

# Singleton retrieval and test reset
openai_client, anthropic_client = get_openai_client(), get_anthropic_client()
_reset_clients()
```

---

## Technical Considerations

### 1. Execution Pipeline (`generate_estimation`)

1. **Preprocessing (`opts.preprocessing`)**:
   - `"none"`: Uses raw transcription as `user_input`.
   - `"inline_cleaning"`: Appends `INLINE_CLEANING_BLOCK` to the system prompt (single LLM call) for a simple user input cleaning.
   - `"two_phase"`: Runs `extract_requirements` (`EXTRACTION_MAX_TOKENS = 1500`) first; replaces `user_input` with extracted Markdown requirements and records `preprocessing_input_tokens` / `preprocessing_output_tokens`.
2. **CAG System Prompt (`build_system_prompt`)**: Combines consultant role, billing rates (`62.50 EUR/h` dev, `50 EUR/h` design), `ACTIVE_OUTPUT_PROMPT` (`PROMPT_OUTPUT_BASIC` or `PROMPT_OUTPUT_STRUCTURED`), optional `INLINE_CLEANING_BLOCK`, and `0..N` few-shot examples (`markdown`, `json`, or `narrative`).
3. **Cache Lookup (`EstimationCache`)**: Computes a SHA-256 key from `(system_prompt, user_message, model, max_tokens, thinking_budget, preprocessing)`. Returns immediately on hit (`cache_hit=True`); persists result on miss (`cache_hit=False`).

### 2. OpenAI Responses API (`_call_openai` & `call_with_retry`)

- **Payload**: Sends `input=messages`, `instructions=system_prompt`, `max_output_tokens=max_tokens`, and `store=False`.
- **Reasoning vs. Temperature**: Sends `reasoning={"effort": effort}` when `spec.is_reasoning` is `True`; otherwise sends `temperature=0.3`. Automatically strips `"temperature"` and retries if a `BadRequestError` rejects it.
- **Backoff Formula**: Retries `BadRequestError`, `RateLimitError`, `InternalServerError`, and `APIConnectionError` up to 3 times with `wait = (2 ** attempt) + SystemRandom().uniform(0, 1)`.
- **Completion Check**: Requires `response.status == "completed"` (mapped to `finish_reason="stop"`). Raises `LLMServiceError` on `"incomplete"` or `"failed"`.

### 3. Anthropic Messages API (`_call_anthropic`)

- **Payload**: Wraps string input as `[{"role": "user", "content": messages}]` and passes `system=system_prompt` at top level.
- **Extended Thinking**: When `thinking_budget` is set, enables `thinking`, omits `temperature`, pads `max_tokens = max(max_tokens, thinking_budget + 1024)`, and extracts the first `type == "text"` block from `response.content`. Logs `"thinking_budget_ignored_for_provider"` if passed to an OpenAI model.
- **Truncation Warning**: Logs `"llm_response_truncated"` when `stop_reason == "max_tokens"`.

### 4. Error Shielding

All upstream SDK exceptions and non-completed states are logged via `structlog` and normalized into `LLMServiceError`, decoupled from FastAPI HTTP primitives.
