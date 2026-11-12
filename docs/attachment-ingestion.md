# Attachment Ingestion

## Overview

The attachment subsystem ([`extractor.py`](../src/aie_estimator/attachments/extractor.py)) implements **Approach Local Text Extraction** for `.pdf`, `.docx`, `.txt`, and `.md` uploads. Plain text is extracted locally, bounded by [`Settings`](../src/aie_estimator/settings.py) budgets, checked and PII-redacted via [`check_attachment_input`](../src/aie_estimator/guardrails/input.py), and isolated inside XML `<attachments>` delimiters before reaching the prompt layer.

## Usage Examples

### Extract, sanitize, and enrich a turn

```python
from aie_estimator.attachments.extractor import enrich_transcript, extract_text
from aie_estimator.guardrails.input import check_attachment_input, get_moderation_client

raw_text = extract_text(filename="architecture.pdf", content=file_bytes)
safe_text = await check_attachment_input(raw_text, openai_client=get_moderation_client())
prompt_input = enrich_transcript(
    transcript="Estimate the backend API migration.",
    attachments=[("architecture.pdf", safe_text)],
)
```

## Technical Considerations

1. **Provider Independence Over Multimodal Files APIs**: Extracting text locally keeps requests compatible with `cost-based-routing` and cross-provider fallbacks across OpenAI, Gemini, Anthropic, Mistral, and DeepSeek in [`llm_profiles.yml`](../src/aie_estimator/llm_profiles.yml), avoids provider-locked `file_id` state, and reuses the same parsing stage for future RAG chunking.
2. **Two-Tier Budget Invariant**: [`extract_text`](../src/aie_estimator/attachments/extractor.py) rejects unsupported extensions (`UnsupportedAttachmentError`) and oversized payloads (`max_attachment_bytes`) before parsing, then caps each file at `max_attachment_chars`. [`enrich_transcript`](../src/aie_estimator/attachments/extractor.py) enforces `max_attachments_per_turn` and the cumulative `max_total_attachment_chars` turn budget so sliding-window history does not multiply token overhead across turns.
3. **Parser Scope & Binary Safeguards**: `.docx` parsing reads both paragraphs and `document.tables` rows (` | `-joined) so scope/rate matrices are preserved. `.txt` and `.md` payloads require strict UTF-8 decoding and reject `\x00` bytes (`AttachmentExtractionError`) to block disguised binaries. Empty or image-only documents return `""` and skip XML wrapping.
4. **Prompt-Injection Delimiting**: [`enrich_transcript`](../src/aie_estimator/attachments/extractor.py) strips path segments and unsafe attribute characters from filenames, escapes embedded `</attachment>` and `</attachments>` sequences, and wraps non-empty blocks in `<attachments><attachment filename="...">...</attachment></attachments>`, aligning with [`v1/user.j2`](../src/aie_estimator/prompts/estimation/v1/user.j2) without triggering `<system>`/`<prompt>` injection filters.
5. **Async Event-Loop Boundary**: `pypdf` and `python-docx` parsing is CPU-bound and synchronous; async FastAPI endpoints and services must invoke [`extract_text`](../src/aie_estimator/attachments/extractor.py) via `asyncio.to_thread`.
6. **UI Upload Entry Point**: [`streamlit_app.py`](../streamlit_app.py) exposes multi-file upload through a sidebar `st.file_uploader(accept_multiple_files=True)` constrained to [`Settings.supported_attachment_extensions`](../src/aie_estimator/settings.py) and surfaces HTTP `415`/`422` attachment errors directly; see [Conversational Estimation UI](streamlit-conversational-ui.md).

