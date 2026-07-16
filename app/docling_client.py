import io
import logging

import httpx
from pypdf import PdfReader, PdfWriter

from .config import get_settings

logger = logging.getLogger(__name__)

# Conservative timeouts: connect fast, allow plenty of time for large PDFs to process.
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)

# PDFs with fewer than this many average extractable characters per page are
# classified as image-based (scanned) and sent to docling with ocr_mode=force.
_IMAGE_BASED_CHARS_PER_PAGE_THRESHOLD = 100


def slice_pdf_to_max_pages(pdf_bytes: bytes, max_pages: int) -> tuple[bytes, int, int]:
    """Keep only the first ``max_pages`` pages of a PDF.

    Returns ``(possibly_sliced_bytes, original_page_count, kept_page_count)``.
    When ``max_pages`` is <= 0 or the PDF already fits, the original bytes are
    returned unchanged. On read/write failure, returns the original bytes so
    the pipeline can still try docling.
    """
    if max_pages <= 0:
        return pdf_bytes, 0, 0
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        original_count = len(reader.pages)
        if original_count <= max_pages:
            return pdf_bytes, original_count, original_count

        writer = PdfWriter()
        for i in range(max_pages):
            writer.add_page(reader.pages[i])
        out = io.BytesIO()
        writer.write(out)
        sliced = out.getvalue()
        logger.info(
            "slice_pdf_to_max_pages: truncated PDF from %d to %d pages",
            original_count,
            max_pages,
        )
        return sliced, original_count, max_pages
    except Exception:
        logger.warning(
            "slice_pdf_to_max_pages: failed to truncate PDF; using original bytes",
            exc_info=True,
        )
        return pdf_bytes, 0, 0


def _looks_image_based(char_counts: list[int]) -> bool:
    """Classify from per-page extractable character counts.

    A PDF is image-based when the AVERAGE page has little text — OR when MOST
    pages have little text. The second test catches mixed documents (e.g. a
    scanned SBC with one text-based cover/disclaimer page) whose average is
    inflated by the few text pages while the benefit tables are pure images.
    """
    if not char_counts:
        return True
    avg = sum(char_counts) / len(char_counts)
    if avg < _IMAGE_BASED_CHARS_PER_PAGE_THRESHOLD:
        return True
    sparse_pages = sum(
        1 for c in char_counts if c < _IMAGE_BASED_CHARS_PER_PAGE_THRESHOLD
    )
    return sparse_pages > len(char_counts) / 2


def is_image_based_pdf(pdf_bytes: bytes) -> bool:
    """Return True if the PDF appears to be image-based (scanned, no text layer).

    Uses pypdf to attempt text extraction locally.  If the average extractable
    text across all pages is below the threshold the PDF is considered image-based
    and requires forced OCR for accurate content extraction.

    Failure to read the PDF is treated as text-based (safe default: don't
    force OCR unnecessarily).
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if not reader.pages:
            return True  # empty / unreadable — let docling decide

        char_counts = [
            len((page.extract_text() or "").strip())
            for page in reader.pages
        ]
        result = _looks_image_based(char_counts)
        logger.debug(
            "is_image_based_pdf: classification=%s avg_chars_per_page=%.1f",
            "image-based" if result else "text-based",
            sum(char_counts) / len(char_counts),
        )
        return result
    except Exception:
        logger.warning(
            "is_image_based_pdf: failed to inspect PDF text content; "
            "defaulting to text-based (will use configured ocr_mode)",
            exc_info=True,
        )
        return False


async def convert_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """POST the PDF to docling-service and return the parsed ConvertResponse dict.

    Automatically uses ocr_mode=force for image-based (scanned) PDFs, falling
    back to the configured DOCLING_OCR_MODE for text-based PDFs.

    Raises distinct exceptions so the caller can surface precise 502 error details:
      - httpx.ConnectError   → docling-service is not reachable
      - httpx.HTTPStatusError → docling-service returned a non-2xx response
      - ValueError           → response body is not valid JSON
    """
    settings = get_settings()
    endpoint = f"{settings.DOCLING_SERVICE_URL}{settings.DOCLING_ENDPOINT}"

    # Override ocr_mode to "force" for image-based PDFs so docling performs full OCR.
    if is_image_based_pdf(pdf_bytes):
        ocr_mode = "force"
        logger.info("docling_client: image-based PDF detected — overriding ocr_mode to 'force'")
    else:
        ocr_mode = settings.DOCLING_OCR_MODE

    logger.info(
        "docling_client: sending conversion request",
        extra={
            "endpoint": endpoint,
            "filename": filename,
            "ocr_mode": ocr_mode,
            "table_mode": settings.DOCLING_TABLE_MODE,
        },
    )

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                endpoint,
                files={"file": (filename, pdf_bytes, "application/pdf")},
                data={
                    "ocr_mode": ocr_mode,
                    "table_mode": settings.DOCLING_TABLE_MODE,
                },
            )
    except httpx.ConnectError as exc:
        logger.error("docling_client: docling-service unreachable", extra={"error": str(exc)})
        raise

    if response.status_code != 200:
        logger.error(
            "docling_client: docling-service returned error",
            extra={"status_code": response.status_code, "body_preview": response.text[:500]},
        )
        response.raise_for_status()

    try:
        data = response.json()
    except Exception as exc:
        raise ValueError(f"docling-service returned non-JSON body: {exc}") from exc

    logger.info(
        "docling_client: conversion complete",
        extra={"page_count": data.get("page_count"), "ocr_mode_used": ocr_mode},
    )
    # Inject the ocr_mode we actually used so the pipeline can surface it in the response.
    data["_ocr_mode_used"] = ocr_mode
    return data
