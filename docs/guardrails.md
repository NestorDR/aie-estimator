# Guardrails and Semantic Validation

## Overview

[`src/aie_estimator/guardrails/`](../src/aie_estimator/guardrails/) implements defense-in-depth semantic validation across the estimation lifecycle. Guardrails enforce semantic integrity: verifying that input text is free from harmful content, prompt injection, and personally identifiable information (PII), while ensuring output remains safely bounded.

| Layer | Type | Target | Implementation | Failure Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Input Syntactic | Request structure, character limits, enums | Pydantic v2 ([`EstimationRequest`](../src/aie_estimator/schemas/estimations.py)) | **Exception**: HTTP 422 Unprocessable Content |
| **Layer 2** | Input Semantic | Content moderation, prompt injection, PII | `check_input` / `check_attachment_input` ([`input.py`](../src/aie_estimator/guardrails/input.py)) | **Exception**: HTTP 400 (`check_input`); **Redaction**: `[REDACTED_*]` on attachment PII (`check_attachment_input`) |
| **Layer 3** | Prompt Robustness | Role boundaries, refusal guidance | Jinja2 templates ([`prompts/estimation/v1/system.j2`](../src/aie_estimator/prompts/estimation/v1/system.j2)) | **Filter**: Guides model to out-of-scope format |
| **Layer 4** | Output Syntactic | Phase sum matches total, out-of-scope prefix | Instructor + `@model_validator` | **Fix-with-retry**: Re-prompts LLM up to `max_retries` |
| **Layer 5** | Output Semantic | Low-confidence normalization | `enforce_scope_response` ([`output.py`](../src/aie_estimator/guardrails/output.py)) | **Filter**: Normalizes unflagged low-confidence output to fallback phase |

---

## Usage Examples

### 1. Validate Input Text with `check_input`

```python
from aie_estimator.guardrails.input import InputGuardrailViolation, check_input, get_moderation_client

client = get_moderation_client()
try:
    await check_input("Build a CRM system with OAuth2.", openai_client=client)
except InputGuardrailViolation as exc:
    print(f"Rejected: reason={exc.reason}, message={exc.message}")
```

### 2. Sanitize Attachment Text with `check_attachment_input`

```python
from aie_estimator.guardrails.input import check_attachment_input, get_moderation_client

safe_text = await check_attachment_input(extracted_text, openai_client=get_moderation_client())
```

### 3. Normalize Output with `enforce_scope_response`

```python
from aie_estimator.guardrails.output import enforce_scope_response

# Normalizes unflagged low-confidence results (< 30% without "Out of scope:")
safe_result = enforce_scope_response(raw_model_result)
```

### 4. Handle Guardrail Rejections (HTTP 400)

```python
# FastAPI maps InputGuardrailViolation to HTTP 400:
# {"detail": {"reason": "prompt_injection", "message": "Suspicious instruction-like text detected: '...'"}}
```

---

## Technical Considerations

1. **Failure Policies**:
   - **Exception**: Terminates processing immediately on security/privacy violations (`InputGuardrailViolation` on toxic text, injection, or PII) returning HTTP 400.
   - **Fix-with-retry**: Instructor re-prompts the LLM on validation errors up to `max_retries`.
   - **Filter**: Gracefully normalizes output in [`output.py`](../src/aie_estimator/guardrails/output.py) (`confidence_pct < 30`) without throwing an exception, setting cost to 0 EUR and phase to `"Not estimated"`.
2. **Pre-Cache Execution**: `check_input` executes strictly before exact-match and semantic cache lookups. Malicious prompts or sensitive PII are rejected before querying or polluting cache stores.
3. **Decoupled Moderation (`omni-moderation-latest`)**: OpenAI moderation client is decoupled from the target estimation model. If the moderation endpoint is unreachable, `check_input` logs a warning and proceeds to evaluate deterministic regex heuristics (fail-open policy).
4. **Regex Heuristics & Attachment PII Invariant**: Precompiled regular expressions in [`src/aie_estimator/guardrails/input.py`](../src/aie_estimator/guardrails/input.py) check emails (`_EMAIL_RE`), IBANs (`_IBAN_RE`), credit cards (`_CREDIT_CARD_RE`), 10-to-13-digit numeric identifiers/phones (`_NUMERIC_IDENTIFIER_RE`), and prompt injection patterns (< 1 ms). While [`check_input`](../src/aie_estimator/guardrails/input.py) rejects PII with HTTP 400 on direct transcripts, [`check_attachment_input`](../src/aie_estimator/guardrails/input.py) enforces moderation and prompt-injection rejections while redacting matched PII via `[REDACTED_*]` placeholders (logging `attachment_pii_redacted` counts) so documents containing contact headers remain estimable.
