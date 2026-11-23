# src/aie_estimator/ingestion/parsers/transcript_txt.py

"""
Parser for transcript .txt files.

The seed transcript corpus contains two coexisting formats. The parser detects the
format heuristically and produces a different ``Document`` granularity:

* **With speaker tags** ``[hh:mm:ss] Speaker: text``: one document per turn.
  ``speaker`` and ``timestamp`` go in ``metadata.extra`` so downstream filters
  can use them.

* **Without tags** (legacy pre-2024 format): one document per paragraph block
  (separated by blank lines). No ``speaker`` field — we cannot fabricate it.

The two paths produce the same ``Document`` shape; only the granularity and
``extra`` keys differ.
"""

# --- Python Standard Library ---
from collections.abc import Iterable
import re
from typing import ClassVar, Final

# --- App modules ---
from aie_estimator.ingestion.documents.models import Document, DocumentMetadata
from aie_estimator.ingestion.loaders.filesystem import LoadedBlob
from aie_estimator.ingestion.parsers.protocol import ParseContext

_MIN_TAGGED_LINES: Final[int] = 3
_TURN_RE: Final[re.Pattern[str]] = re.compile(
    r"^\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<speaker>[^:]+?):\s*(?P<text>.*)$",
)
_BLANK_LINE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")


class TranscriptTxtParser:
    """Parses tagged speaker-turn transcripts with a paragraph-block fallback for legacy TXT files."""

    supported_formats: ClassVar[set[str]] = {"txt"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        """
        Detect whether a transcript uses speaker tags and emit turn or paragraph-block Documents.

        :param blob: Raw UTF-8 text file blob.
        :param context: Run-scoped catalog and timestamp metadata.

        :return: Iterable of parsed Document instances.
        """
        text = blob.bytes_.decode("utf-8", errors="replace")
        if _has_speaker_tags(text):
            yield from self._parse_tagged(text, blob, context)
        else:
            yield from self._parse_legacy(text, blob, context)

    def _parse_tagged(self, text: str, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        for idx, line in enumerate(text.splitlines()):
            match = _TURN_RE.match(line.strip())
            if not match:
                continue
            content = match.group("text").strip()
            if not content:
                continue
            yield Document(
                id=f"{context.source.name}:{blob.relative_path}:turn-{idx:04d}",
                text=content,
                metadata=_build_metadata(
                    blob,
                    context,
                    extra={
                        "speaker": match.group("speaker").strip(),
                        "timestamp": match.group("ts").strip(),
                        "format_mode": "tagged",
                    },
                ),
            )

    def _parse_legacy(self, text: str, blob: LoadedBlob, context: ParseContext) -> Iterable[Document]:
        blocks = [block.strip() for block in _BLANK_LINE_SPLIT_RE.split(text) if block.strip()]
        for idx, block in enumerate(blocks):
            yield Document(
                id=f"{context.source.name}:{blob.relative_path}:block-{idx:04d}",
                text=block,
                metadata=_build_metadata(
                    blob,
                    context,
                    extra={
                        "format_mode": "legacy",
                        "block_index": idx,
                    },
                ),
            )


def _has_speaker_tags(text: str) -> bool:
    """
    Return True when enough lines match the ``[hh:mm:ss] Speaker:`` format.

    :param text: Decoded transcript text.

    :return: True if tagged speaker turns meet the detection threshold.
    """
    return sum(1 for line in text.splitlines() if _TURN_RE.match(line.strip())) >= _MIN_TAGGED_LINES


def _build_metadata(blob: LoadedBlob, context: ParseContext, extra: dict[str, object]) -> DocumentMetadata:
    return DocumentMetadata(
        source_name=context.source.name,
        source_version=context.source_version,
        source_location=blob.relative_path,
        ingested_at=context.ingested_at,
        extra=extra,
    )
