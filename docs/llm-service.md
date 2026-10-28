# LLM Service Architecture and Provider Integration

## Overview

This document specifies the LLM service architecture, multi-provider execution engine, CAG prompt construction, and provider communication protocols within `aie-estimator`.

The service supports two upstream LLM providers:
- **OpenAI**: Uses the modern Responses API (`client.responses.create`) with support for reasoning models (`o1`, `o3`), configurable token allocation (`max_output_tokens`), and automatic exponential backoff with full jitter.
- **Anthropic**: Uses the Messages API (`client.messages.create`) supporting Claude models with optional extended thinking (`thinking={"type": "enabled", "budget_tokens": ...}`).

Both providers consume a unified internal interface and return structured estimations while abstracting provider-specific payloads, token accounting, and retry mechanisms.

---

## Usage Examples

### 1. Generating an Estimation with Default Options

Invoke `generate_estimation` with a meeting transcription string:

```python
from aie_estimator.services import llm_service

transcription = "Client requires an e-commerce backend with Stripe integration and inventory management."
result = llm_service.generate_estimation(transcription)

print(result["estimation"])
print(result["model"])
print(result["provider"])
```

### 2. Passing Generation Options

Configure custom model overrides, two-phase preprocessing, or extended thinking budgets via `GenerationOptions`:

```python
from aie_estimator.services.llm_service import GenerationOptions, generate_estimation

opts = GenerationOptions(
    model="claude-3-7-sonnet-20250219",
    two_phase=True,
    thinking_budget=2048,
    temperature=None,
)

result = generate_estimation(transcription, opts=opts)
```

### 3. Direct Provider Invocations

When executing raw completions, call `_call_openai` or `_call_anthropic` with the normalized signature `(messages, system_prompt, model, max_tokens, ...)`:

```python
from aie_estimator.services.llm_service import _call_anthropic, _call_openai

# OpenAI Responses API invocation
openai_response = _call_openai(
    messages="Estimate this backend project...",
    system_prompt="You are a senior technical estimator.",
    model="gpt-4o-mini",
    max_tokens=2048,
)

# Anthropic Messages API invocation
anthropic_response = _call_anthropic(
    messages=[{"role": "user", "content": "Estimate this backend project..."}],
    system_prompt="You are a senior technical estimator.",
    model="claude-3-5-sonnet-20241022",
    max_tokens=4096,
)
```

### 4. Client Lifecycle Management

Cached client instances are retrieved via factory functions:

```python
from aie_estimator.services.llm_service import _reset_clients, get_anthropic_client, get_openai_client

openai_client = get_openai_client()
anthropic_client = get_anthropic_client()

# Reset cached client singletons during testing
_reset_clients()
```

---

## Technical Considerations

### 1. OpenAI Responses API Integration

- **Endpoint and Payload**: The service invokes `client.responses.create()` passing `input` (raw prompt string or normalized list of message dictionaries), `instructions` (system prompt instructions), and `max_output_tokens`.
- **Reasoning Models**: When targeting reasoning models (`o1`, `o3`, or model identifiers matching `-6-`), `reasoning={"effort": effort}` is passed and `temperature` is omitted. If an upstream `BadRequestError` indicates unsupported temperature, the request automatically falls back to omitting `temperature`.
- **Completion Mapping**: The Responses API returns status `completed`. The service maps this to `finish_reason="stop"` to maintain evaluation compatibility with downstream scoring functions.
- **Privacy and Storage**: Requests are submitted with `store=False` to prevent data retention on OpenAI servers.

### 2. Anthropic Messages API and Extended Thinking

- **System Prompt Separation**: Anthropic separates system guidance from the conversation history; `system=system_prompt` is passed as a top-level parameter, and conversation turns are formatted as `[{"role": "user" | "assistant", "content": ...}]`.
- **Thinking Budget**: When `thinking_budget` is configured:
  - `thinking={"type": "enabled", "budget_tokens": thinking_budget}` is passed.
  - `temperature` is omitted.
  - `max_tokens` must strictly exceed `thinking_budget`.
- **Truncation Detection**: If Anthropic returns `stop_reason == "max_tokens"`, the service emits a structured warning (`anthropic_response_truncated`).

### 3. Unified Error Handling and Exception Shielding

- **Decoupling**: Domain service functions do not raise HTTP exceptions or reference web framework primitives.
- **Logging**: When an upstream provider call fails, original API exception details, status codes, and headers are logged via `structlog.get_logger(__name__).error(...)` or `exception(...)`.
- **Exception Normalization**: All unrecoverable upstream failures are wrapped and re-raised as `LLMServiceError`.
- **Global Error Handling**: In `src/aie_estimator/server.py`, `@app.exception_handler(LLMServiceError)` captures these domain exceptions and returns a consistent, sanitized HTTP 500 response (`{"detail": "An error occurred while generating the estimation. Please try again later."}`).

### 4. Exponential Backoff and Jitter

The OpenAI call engine implements exponential backoff with full jitter:

```text
wait_time = random(0, min(MAX_RETRY_DELAY, BASE_RETRY_DELAY * (2 ** attempt)))
```

Default parameters:
- `BASE_RETRY_DELAY = 1.0` second
- `MAX_RETRY_DELAY = 16.0` seconds
- `MAX_RETRIES = 3` attempts

Transient errors (`RateLimitError`, `APIConnectionError`, `InternalServerError`) trigger backoff, while non-retryable errors (`AuthenticationError`, `BadRequestError`) abort immediately.
