# Guardrails and Semantic Validation

## Overview

The `aie_estimator.guardrails` subpackage implements defense-in-depth semantic validation across the software estimation lifecycle. While Pydantic schemas enforce syntactic structure, types, and arithmetic invariants (`EstimationRequest` and `EstimationResult`), guardrails enforce semantic integrity: verifying that input text is free from harmful content, prompt injection attacks, and unintended personally identifiable information (PII), and ensuring that model output remains predictable and safely bounded.

Validation is organized across two axes (Input vs. Output, Syntactic vs. Semantic) and enforced through three failure policies:

| Layer | Type | Target | Tooling | Failure Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1** | Input Syntactic | Request shape, lengths (20–2000 chars), enums | Pydantic v2 (`EstimationRequest`) | **Exception**: HTTP 422 Unprocessable Content |
| **Layer 2** | Input Semantic | Content moderation, prompt injection, PII | `check_input` (`guardrails/input.py`) | **Exception**: HTTP 400 Bad Request (`InputGuardrailViolation`) |
| **Layer 3** | Prompt Robustness | Refusal permission, scope boundary | Jinja2 templates (`prompts/estimation/v1/system.j2`) | **Filter**: Model guided to out-of-scope response |
| **Layer 4** | Output Syntactic | Phase sum matches total, out-of-scope prefix | Instructor + `@model_validator` | **Fix-with-retry**: Instructor re-prompts LLM up to `max_retries` |
| **Layer 5** | Output Semantic | Low-confidence normalization | `enforce_scope_response` (`guardrails/output.py`) | **Filter**: Rewrites unflagged low-confidence output to fallback phase |

---

## Usage Examples

### 1. Validate Input Text with `check_input`

```python
from aie_estimator.guardrails.input import InputGuardrailViolation, check_input, get_moderation_client

# Retrieve the cached AsyncOpenAI client if OPENAI_API_KEY is configured
client = get_moderation_client()

try:
    await check_input(
        "Build a CRM application with OAuth2 integration.",
        openai_client=client,
    )
except InputGuardrailViolation as exc:
    print(f"Rejected: reason={exc.reason}, message={exc.message}")
```

### 2. Normalize Output with `enforce_scope_response`

```python
from aie_estimator.guardrails.output import enforce_scope_response
from aie_estimator.schemas.estimations import EstimationResult

# Unflagged low-confidence output is normalized to a safe out-of-scope representation
filtered_result = enforce_scope_response(raw_model_result)
print(filtered_result.summary)  # Begins with "Out of scope: not enough information..."
print(filtered_result.total_cost_eur)  # 0
print(filtered_result.phases[0].name)  # "Not estimated"
```

### 3. Handle Guardrail Rejections in FastAPI / HTTP Clients

```python
import httpx

try:
    response = await client.post("/api/v1/estimate", json=payload)
    response.raise_for_status()
except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 400:
        error_detail = exc.response.json().get("detail", {})
        reason = error_detail.get("reason")  # "moderation", "prompt_injection", or "pii"
        message = error_detail.get("message")
        print(f"Guardrail violation ({reason}): {message}")
```

---

## Technical Considerations

### 1. Three Canonical Failure Policies

Every guardrail explicitly declares which failure policy it enforces:
- **Exception**: Aborts processing immediately and records an audit log. Used when a request violates security or privacy boundaries (`InputGuardrailViolation` on toxic text, injection, or PII). The client receives HTTP 400 Bad Request with machine-readable error reasons.
- **Fix-with-retry**: Intercepts recoverable syntactic or arithmetic errors and prompts the LLM to self-correct up to `opts.max_retries` (default `3`). Managed by Instructor on `EstimationResult` model validators.
- **Filter**: Gracefully degrades or normalizes content without raising an operational error. Used in `enforce_scope_response` when `confidence_pct < 30` without an explicit out-of-scope summary, returning a single `"Not estimated"` phase with 0 EUR cost.

### 2. Execution Order Relative to Caching

Input guardrails execute **strictly before** exact-match cache resolution in `generate_estimation`:
```
Incoming Request
       │
       ▼
1. check_input() ───[Violation]──► Raise InputGuardrailViolation (HTTP 400)
       │
       ▼
2. Exact-Match Cache Lookup
       │
   [Miss] ──► LLM Generation ──► Output Filter ──► Store Cache
```
If input guardrails ran after the cache lookup, malicious prompt injections or requests containing sensitive employee PII could be retrieved and served from cache. Running guardrails first ensures dirty inputs never hit or populate the cache.

### 3. Moderation Provider Decoupling & `omni-moderation-latest`

Content moderation uses direct `AsyncOpenAI` targeting the `omni-moderation-latest` model:
- **Decoupled from Estimation Model**: The model chosen to produce the estimation (e.g., Claude 3.5 Sonnet or DeepSeek) does not dictate moderation. Anthropic lacks a free standalone moderation endpoint; decoupling guarantees consistent safety across all generation models.
- **Fail-Open Policy**: If the OpenAI Moderation API is unreachable or fails due to network/authentication errors, `check_input` logs a `moderation_call_failed` warning and proceeds to evaluate deterministic regex heuristics, preventing third-party outages from taking down the estimation service.

### 4. Custom Regex Heuristics vs. Microsoft Presidio

`aie-estimator` relies on precompiled regular expressions for PII detection rather than heavyweight NLP libraries like Microsoft Presidio (`presidio-analyzer`):

| Evaluation Dimension | Custom Regex Heuristics (`guardrails/input.py`) | Microsoft Presidio (`presidio-analyzer`) |
| :--- | :--- | :--- |
| **Execution Latency** | < 1 ms | 20–80 ms per inference pass |
| **Dependencies** | Python standard library (`re`) | `presidio-analyzer`, `spacy`, `thinc`, language models |
| **Memory Footprint** | Negligible (< 1 MB) | ~150–300 MB resident model memory |
| **Python 3.14+ Compatibility** | 100% native compatibility | Risk of C-extension compilation issues on bleeding-edge Python |
| **Failure Policy Fit** | Fits binary **Exception** policy (reject immediately) | Primarily designed for entity masking and redaction |
| **False Positive Profile** | Tuned to ignore dates (`2026-11-08`) and semver (`1.16.1`) | High risk of classifying tech jargon or libraries as names |

Custom regexes check emails (`_EMAIL_RE`), international bank account numbers (`_IBAN_RE`), credit card numbers (`_CREDIT_CARD_RE`), and phone numbers (`_PHONE_RE`), satisfying safety requirements without operational overhead. Prompt injection heuristics target instruction overrides, system tag boundary escapes, new instruction declarations, memory wipes, persona jailbreaks, disregard directives, system prompt exfiltration, and faux ChatML delimiters.

