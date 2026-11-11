# src/aie_estimator/attachments/extractor.py

# --- Python Standard Library ---
from collections.abc import Iterable
import io
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Final

# --- Third Party Libraries ---
from docx import Document
from pypdf import PdfReader
import structlog

# --- App modules ---
from aie_estimator.settings import get_settings

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

_DEFAULT_FILENAME: Final[str] = "attachment"
_TRUNCATION_MARKER: Final[str] = "\n[... truncated ...]"
_CLOSING_ATTACHMENT_TAG_RE: Final[re.Pattern[str]] = re.compile(
    r"</\s*(attachments?)\s*>",
    re.IGNORECASE,
)
_UNSAFE_FILENAME_CHARS_RE: Final[re.Pattern[str]] = re.compile(r'[\x00-\x1f\x7f"<>\\/:|?*]+')


class AttachmentExtractionError(Exception):
    """Raised when an attachment violates size/count limits or cannot be parsed."""

    def __init__(self, filename: str, message: str) -> None:
        """
        Initialize the extraction error with the offending filename and message.

        :param filename: Original or sanitized attachment filename.
        :param message: Human-readable explanation of the extraction failure.
        """
        super().__init__(message)
        self.filename: Final[str] = filename
        self.message: Final[str] = message


class UnsupportedAttachmentError(AttachmentExtractionError):
    """Raised when an uploaded attachment has an unsupported file extension."""


def _sanitize_filename(filename: str) -> str:
    """
    Strip directory traversal segments and unsafe XML attribute characters from a filename.

    :param filename: Raw filename supplied by the client.

    :return: Sanitized basename suitable for XML attributes and logging.
    """
    posix_name = PurePosixPath(filename.replace("\\", "/")).name
    win_name = PureWindowsPath(posix_name).name
    cleaned = _UNSAFE_FILENAME_CHARS_RE.sub("_", win_name).strip()
    return cleaned or _DEFAULT_FILENAME


def _extension(filename: str) -> str:
    """
    Extract the lowercase file extension from a filename or path.

    :param filename: Raw filename or path.

    :return: Lowercase suffix including leading dot, or empty string if absent.
    """
    return PurePosixPath(filename.replace("\\", "/")).suffix.lower()


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """
    Normalize configured attachment extensions to lowercase dotted suffixes.

    :param extensions: Collection of extension strings.

    :return: Normalized frozenset of lowercase dotted suffixes.
    """
    cleaned = (ext.strip().lower() for ext in extensions)
    return frozenset(ext if ext.startswith(".") else f".{ext}" for ext in cleaned if ext)


def _truncate_to_budget(text: str, max_chars: int) -> tuple[str, bool]:
    """
    Truncate text so its length never exceeds ``max_chars``, appending a marker when space allows.

    :param text: Input text to bound.
    :param max_chars: Maximum permitted character count (must be positive).

    :return: Tuple of ``(bounded_text, was_truncated)``.
    """
    if len(text) <= max_chars:
        return text, False
    room = max_chars - len(_TRUNCATION_MARKER)
    return (f"{text[:room].rstrip()}{_TRUNCATION_MARKER}" if room > 0 else text[:max_chars]), True


def _escape_xml_boundaries(text: str) -> str:
    """
    Neutralize closing ``</attachment>`` and ``</attachments>`` tags inside document text.

    :param text: Extracted attachment text.

    :return: Text with embedded closing attachment boundary tags escaped.
    """
    return _CLOSING_ATTACHMENT_TAG_RE.sub(lambda match: f"&lt;/{match.group(1)}&gt;", text)


def _extract_pdf(content: bytes) -> str:
    """
    Extract plain text across all readable pages of a PDF payload.

    :param content: Raw PDF file bytes.

    :return: Concatenated page text separated by blank lines.
    """
    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning("pdf_page_extract_failed", error=str(exc)[:200])
            text = ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(content: bytes) -> str:
    """
    Extract non-empty paragraphs and table rows from a DOCX payload.

    :param content: Raw DOCX file bytes.

    :return: Newline-delimited paragraph and table content.
    """
    document = Document(io.BytesIO(content))
    sections: list[str] = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                sections.append(" | ".join(cells))

    return "\n".join(sections)


def _extract_plain_text(filename: str, content: bytes) -> str:
    """
    Decode plain-text or Markdown bytes as strict UTF-8 after rejecting binary null bytes.

    :param filename: Attachment filename for error reporting.
    :param content: Raw text file bytes.

    :return: Decoded UTF-8 text.
    :raises AttachmentExtractionError: If null bytes are present or UTF-8 decoding fails.
    """
    if b"\x00" in content:
        raise AttachmentExtractionError(filename, f"Binary null bytes detected in text attachment {filename!r}.")
    try:
        return content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise AttachmentExtractionError(
            filename,
            f"Failed to decode {filename!r} as UTF-8: {exc}",
        ) from exc


def validate_attachment_count(
    count: int,
    *,
    max_attachments: int | None = None,
) -> None:
    """
    Verify that the number of attachments in a turn does not exceed the configured budget.

    :param count: Number of uploaded attachments for the turn.
    :param max_attachments: Optional per-turn count override; defaults to ``Settings.max_attachments_per_turn``.

    :raises AttachmentExtractionError: If ``count`` exceeds the configured limit.
    """
    limit = max_attachments if max_attachments is not None else get_settings().max_attachments_per_turn
    if count > limit:
        raise AttachmentExtractionError(
            "attachments",
            f"Too many attachments ({count}); maximum allowed per turn is {limit}.",
        )


def extract_text(
    *,
    filename: str,
    content: bytes,
    max_chars: int | None = None,
    max_bytes: int | None = None,
    supported_extensions: Iterable[str] | None = None,
) -> str:
    """
    Extract and budget-bound plain text from an uploaded attachment.

    :param filename: Client-supplied filename used for extension dispatch and diagnostics.
    :param content: Raw attachment bytes.
    :param max_chars: Optional per-file extracted character budget; defaults to ``Settings.max_attachment_chars``.
    :param max_bytes: Optional raw byte limit checked before parsing; defaults to ``Settings.max_attachment_bytes``.
    :param supported_extensions: Optional extension allowlist; defaults to ``Settings.supported_attachment_extensions``.

    :return: Stripped extracted text bounded to ``max_chars`` (empty string if document has no text).
    :raises UnsupportedAttachmentError: If the file extension is not in the allowlist.
    :raises AttachmentExtractionError: If the payload exceeds ``max_bytes`` or cannot be parsed.
    """
    settings = get_settings()
    allowed_exts = _normalize_extensions(
        supported_extensions if supported_extensions is not None else settings.supported_attachment_extensions
    )
    resolved_max_bytes = max_bytes if max_bytes is not None else settings.max_attachment_bytes
    resolved_max_chars = max_chars if max_chars is not None else settings.max_attachment_chars
    safe_name = _sanitize_filename(filename)

    ext = _extension(filename)
    if ext not in allowed_exts:
        raise UnsupportedAttachmentError(
            safe_name,
            f"Unsupported attachment extension {ext!r}; supported: {sorted(allowed_exts)}",
        )

    if len(content) > resolved_max_bytes:
        raise AttachmentExtractionError(
            safe_name,
            f"Attachment {safe_name!r} exceeds maximum allowed size of {resolved_max_bytes} bytes.",
        )

    try:
        if ext == ".pdf":
            text = _extract_pdf(content)
        elif ext == ".docx":
            text = _extract_docx(content)
        else:
            text = _extract_plain_text(safe_name, content)
    except AttachmentExtractionError:
        raise
    except Exception as exc:
        raise AttachmentExtractionError(
            safe_name,
            f"Failed to extract text from {safe_name}: {exc}",
        ) from exc

    text = text.strip()
    if not text:
        log.info("attachment_extracted_empty", filename=safe_name)
        return ""

    bounded_text, was_truncated = _truncate_to_budget(text, resolved_max_chars)
    if was_truncated:
        log.info(
            "attachment_truncated",
            filename=safe_name,
            original_chars=len(text),
            kept_chars=len(bounded_text),
        )

    return bounded_text


def enrich_transcript(
    *,
    transcript: str,
    attachments: list[tuple[str, str]],
    max_attachments: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    """
    Wrap non-empty extracted attachments in sanitized XML delimiters and append them to the transcript.

    Enforces the per-turn attachment count and cumulative character budget, strips unsafe filename
    characters, and escapes embedded ``</attachment>`` or ``</attachments>`` closing tags.

    :param transcript: Base user transcript for the estimation turn.
    :param attachments: Sequence of ``(filename, extracted_text)`` pairs.
    :param max_attachments: Optional per-turn attachment count override; defaults to ``Settings.max_attachments_per_turn``.
    :param max_total_chars: Optional cumulative attachment character budget; defaults to ``Settings.max_total_attachment_chars``.

    :return: Transcript unchanged if no non-empty attachments exist, otherwise transcript followed by ``<attachments>`` XML block.
    :raises AttachmentExtractionError: If the attachment count exceeds ``max_attachments``.
    """
    settings = get_settings()
    validate_attachment_count(len(attachments), max_attachments=max_attachments)
    remaining_budget = max_total_chars if max_total_chars is not None else settings.max_total_attachment_chars

    blocks: list[str] = []
    for raw_filename, raw_text in attachments:
        stripped = (raw_text or "").strip()
        if not stripped:
            continue

        safe_name = _sanitize_filename(raw_filename)
        if remaining_budget <= 0:
            log.info("turn_attachment_budget_exhausted", omitted_filename=safe_name)
            continue

        bounded, was_truncated = _truncate_to_budget(_escape_xml_boundaries(stripped), remaining_budget)
        # Re-escape after truncation in case truncation fell inside a previously escaped or partial tag
        bounded = _escape_xml_boundaries(bounded)
        if was_truncated:
            log.info(
                "turn_attachment_truncated",
                filename=safe_name,
                remaining_budget=remaining_budget,
                kept_chars=len(bounded),
            )

        remaining_budget -= len(bounded)
        blocks.append(f'<attachment filename="{safe_name}">\n{bounded}\n</attachment>')

    if not blocks:
        return transcript

    return f"{transcript.strip()}\n\n<attachments>\n" + "\n".join(blocks) + "\n</attachments>"
