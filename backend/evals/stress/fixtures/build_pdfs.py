# evals/stress/fixtures/build_pdfs.py

# --- Python Standard Library ---
from pathlib import Path
from typing import Final

# --- Third Party Libraries ---
import structlog

log: Final[structlog.stdlib.BoundLogger] = structlog.get_logger()

STRESS_PDF_SIZES_KB: Final[tuple[int, ...]] = (5, 20, 50, 100)
_DEFAULT_OUTPUT_DIR: Final[Path] = Path(__file__).resolve().parent
_LINES_PER_PAGE: Final[int] = 28
_FILLER_TEXT: Final[str] = "Architecture spec covering API contracts, persistence, caching, and SLAs."


def canary_facts(size_kb: int) -> tuple[str, str, str]:
    """
    Return deterministic start, middle, and end canary facts for a given PDF size.

    :param size_kb: Calibrated target PDF size in kilobytes.

    :return: Tuple of ``(start_fact, mid_fact, end_fact)`` ASCII strings.
    """
    return (
        f"CANARY_START_{size_kb}KB: Project Nimbus requires OAuth2 SSO and SOC2 audit logging.",
        f"CANARY_MID_{size_kb}KB: Core backend uses Python 3.14 FastAPI with Redis semantic cache.",
        f"CANARY_END_{size_kb}KB: Final milestone budget cap is 45000 EUR for phase two.",
    )


def build_calibrated_pdf(size_kb: int) -> bytes:
    """
    Build a deterministic multi-page PDF 1.4 payload calibrated to ``size_kb`` kilobytes.

    Uses uncompressed ASCII content streams so file size in bytes closely matches
    extracted character count and embeds start, middle, and end canary facts.

    :param size_kb: Target size in kilobytes (must be positive).

    :return: Valid deterministic PDF 1.4 bytes.
    :raises ValueError: If ``size_kb`` is not positive.
    """
    if size_kb <= 0:
        raise ValueError("size_kb must be positive.")

    page_count = max(2, round(size_kb / 2.5))
    start_fact, mid_fact, end_fact = canary_facts(size_kb)
    mid_page = page_count // 2
    last_page = page_count - 1

    page_streams: list[bytes] = []
    for page_idx in range(page_count):
        lines = [f"P{page_idx + 1:02d}L{line_idx + 1:02d} {_FILLER_TEXT}"[:72] for line_idx in range(_LINES_PER_PAGE)]
        if page_idx == 0:
            lines[0] = start_fact
        if page_idx == mid_page:
            lines[_LINES_PER_PAGE // 2] = mid_fact
        if page_idx == last_page:
            lines[-1] = end_fact
        body = "\n".join(f"({line}) Tj T*" for line in lines)
        page_streams.append(f"BT /F1 10 Tf 50 740 Td 14 TL\n{body}\nET".encode("ascii"))

    # Object IDs: 1=Catalog, 2=Pages, 3=Font, then pairs (Page, Content) starting at 4.
    page_obj_ids = [4 + (idx * 2) for idx in range(page_count)]
    kids_ref = " ".join(f"{obj_id} 0 R" for obj_id in page_obj_ids)
    objects: list[bytes] = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        f"2 0 obj << /Type /Pages /Kids [{kids_ref}] /Count {page_count} >> endobj\n".encode("ascii"),
        b"3 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
    ]
    for page_id, stream in zip(page_obj_ids, page_streams, strict=True):
        content_id = page_id + 1
        objects.append(
            f"{page_id} 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 3 0 R >> >> >> endobj\n".encode("ascii")
        )
        objects.append(f"{content_id} 0 obj << /Length {len(stream)} >> stream\n".encode("ascii") + stream + b"\nendstream endobj\n")

    output = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj in objects:
        offsets.append(len(output))
        output.extend(obj)

    xref_pos = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii"))
    return bytes(output)


def generate_corpus(output_dir: Path = _DEFAULT_OUTPUT_DIR) -> dict[int, Path]:
    """
    Write calibrated synthetic stress PDFs into ``output_dir``.

    :param output_dir: Directory where ``attach_<size>kb.pdf`` files are written.

    :return: Mapping from size in KB to generated PDF file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("stress_corpus_generation_started", output_dir=str(output_dir), sizes_kb=list(STRESS_PDF_SIZES_KB))

    generated: dict[int, Path] = {}
    for size_kb in STRESS_PDF_SIZES_KB:
        target_path = output_dir / f"attach_{size_kb}kb.pdf"
        target_path.write_bytes(build_calibrated_pdf(size_kb))
        generated[size_kb] = target_path

    log.info(
        "stress_corpus_generation_completed",
        file_count=len(generated),
        byte_sizes={f"{k}kb": p.stat().st_size for k, p in generated.items()},
    )
    return generated


if __name__ == "__main__":
    generate_corpus()
