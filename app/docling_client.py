import logging

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

# Conservative timeouts: connect fast, allow plenty of time for large PDFs to process.
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0)


async def convert_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """POST the PDF to docling-service and return the parsed ConvertResponse dict.

    Raises distinct exceptions so the caller can surface precise 502 error details:
      - httpx.ConnectError   → docling-service is not reachable
      - httpx.HTTPStatusError → docling-service returned a non-2xx response
      - ValueError           → response body is not valid JSON
    """
    settings = get_settings()
    endpoint = f"{settings.DOCLING_SERVICE_URL}{settings.DOCLING_ENDPOINT}"

    logger.info(
        "docling_client: sending conversion request",
        extra={
            "endpoint": endpoint,
            "filename": filename,
            "ocr_mode": settings.DOCLING_OCR_MODE,
            "table_mode": settings.DOCLING_TABLE_MODE,
        },
    )

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                endpoint,
                files={"file": (filename, pdf_bytes, "application/pdf")},
                data={
                    "ocr_mode": settings.DOCLING_OCR_MODE,
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
        extra={"page_count": data.get("page_count"), "ocr_mode_used": data.get("ocr_mode_used")},
    )
    return data
