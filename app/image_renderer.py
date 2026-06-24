"""Render selected PDF pages to JPEG images for the VLM.

Uses PyMuPDF (fitz) rather than pdf2image: no poppler dependency, pure in-process
rendering from bytes, and faster startup -- important when rendering only a few
selected pages per request rather than a whole document.

JPEG is used instead of PNG: for text-heavy insurance documents at 150 DPI, JPEG
at quality 85 is 5-10x smaller than PNG with no meaningful loss for VLM extraction.
This keeps the VLM request payload well under nginx's client_max_body_size limit.

fitz is imported lazily inside render_pages (not at module level) to avoid a
Windows/Python-3.12 GC crash that surfaces during pytest teardown when the C
extension is loaded but no documents are opened (e.g. when render_pages is mocked).
"""
import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 150 DPI ~= 1240 px wide for letter-size; enough detail for the VLM without
# bloating payloads. Increase to 200 if accuracy on small text suffers.
_DPI = 150
# JPEG quality 85 gives a good accuracy/size tradeoff for text documents.
# Lower values (70-80) reduce size further but may hurt fine print legibility.
_JPEG_QUALITY = 85


def render_pages(pdf_bytes: bytes, page_numbers: list[int]) -> list[dict[str, Any]]:
    """Render the given 1-based page numbers from pdf_bytes to base64 JPEGs.

    Returns:
        List of {"page_number": int, "image_b64": str, "mime_type": str,
                 "width": int, "height": int}
        in the same order as page_numbers.
    """
    import fitz  # lazy import: keeps fitz out of the module namespace during tests

    scale = fitz.Matrix(_DPI / 72, _DPI / 72)
    results: list[dict[str, Any]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_num in page_numbers:
            idx = page_num - 1
            if idx < 0 or idx >= doc.page_count:
                logger.warning(
                    f"image_renderer: page {page_num} out of range "
                    f"(document has {doc.page_count} pages) -- skipping"
                )
                continue
            pix = doc[idx].get_pixmap(matrix=scale, alpha=False)
            jpeg_bytes = pix.tobytes("jpeg", jpg_quality=_JPEG_QUALITY)
            image_b64 = base64.b64encode(jpeg_bytes).decode()
            results.append({
                "page_number": page_num,
                "image_b64": image_b64,
                "mime_type": "image/jpeg",
                "width": pix.width,
                "height": pix.height,
            })
            logger.debug(
                f"image_renderer: rendered page {page_num} "
                f"({pix.width}x{pix.height}px, {len(jpeg_bytes)//1024}KB JPEG)"
            )
    finally:
        doc.close()

    logger.info(
        "image_renderer: rendering complete",
        extra={"pages_rendered": [r["page_number"] for r in results]},
    )
    return results
