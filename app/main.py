import asyncio
import logging
import time
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from .auth import verify_token
from .config import get_settings
from .docling_client import convert_pdf
from .image_renderer import render_pages
from .page_router import select_pages
from .schemas import (
    CATEGORY_FIELDS,
    VALID_CATEGORIES,
    VALID_CATEGORIES_SORTED,
    ExtractionResponse,
    LlmTestResponse,
    StageTimings,
)
from .vlm_client import complete_prompt, extract_fields

logger = logging.getLogger(__name__)

app = FastAPI(title="eb3-pdf-parser-v3", version="2.0.0")

_PDF_MAGIC = b"%PDF"


@app.post("/extract_json_v2", response_model=ExtractionResponse)
async def extract_json_v2(
    _token: Annotated[str, Depends(verify_token)],
    file: UploadFile = File(..., description="PDF document"),
    category: str = Form(
        ...,
        description=f"Insurance category. Valid values: {', '.join(VALID_CATEGORIES_SORTED)}",
    ),
) -> ExtractionResponse:
    settings = get_settings()

    if category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid category '{category}'. "
                f"Valid values: {', '.join(VALID_CATEGORIES_SORTED)}"
            ),
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB",
        )
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if not pdf_bytes[:1024].lstrip(b"\xef\xbb\xbf\x00")[:4] == _PDF_MAGIC:
        # lstrip handles BOM/null prefix seen in some PDF generators before the header
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF")

    filename = file.filename or "upload.pdf"
    field_names = CATEGORY_FIELDS[category]

    try:
        result = await asyncio.wait_for(
            _run_pipeline(pdf_bytes, filename, category, field_names, settings),
            timeout=float(settings.REQUEST_TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        logger.error(
            "extract_json_v2: pipeline exceeded timeout",
            extra={"timeout_seconds": settings.REQUEST_TIMEOUT_SECONDS, "category": category},
        )
        raise HTTPException(
            status_code=504,
            detail=f"Processing exceeded the {settings.REQUEST_TIMEOUT_SECONDS}s timeout",
        )

    return result


@app.post("/llm_test", response_model=LlmTestResponse)
async def llm_test(
    _token: Annotated[str, Depends(verify_token)],
    prompt: str = Form(..., description="Prompt to send to the VLM"),
) -> LlmTestResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    try:
        content = await complete_prompt(prompt)
    except httpx.ProxyError:
        raise HTTPException(status_code=502, detail="Proxy unreachable — cannot reach VLM service")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="VLM service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM service returned HTTP {exc.response.status_code}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"VLM returned invalid response: {exc}")
    except Exception:
        logger.exception("llm_test: unexpected VLM error")
        raise HTTPException(status_code=502, detail="VLM request failed unexpectedly")

    return LlmTestResponse(response=content)


async def _run_pipeline(
    pdf_bytes: bytes,
    filename: str,
    category: str,
    field_names: list[str],
    settings,
) -> ExtractionResponse:
    pipeline_start = time.monotonic()

    # ── Stage 1: docling PDF→markdown ────────────────────────────────────────
    logger.info("pipeline[1/4]: docling conversion — start")
    t0 = time.monotonic()
    try:
        docling_result = await convert_pdf(pdf_bytes, filename)
    except httpx.ProxyError:
        raise HTTPException(status_code=502, detail="Proxy unreachable — cannot reach docling-service")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="docling-service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"docling-service returned HTTP {exc.response.status_code}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"docling-service returned invalid response: {exc}")
    except Exception as exc:
        logger.exception("pipeline[1/4]: unexpected docling error")
        raise HTTPException(status_code=502, detail="docling-service failed unexpectedly")
    docling_seconds = time.monotonic() - t0
    logger.info(f"pipeline[1/4]: done in {docling_seconds:.2f}s")

    pages = docling_result.get("pages", [])
    if not pages:
        raise HTTPException(status_code=502, detail="docling-service returned no page content")

    # ── Stage 2: page routing ─────────────────────────────────────────────────
    logger.info("pipeline[2/4]: page routing — start")
    t0 = time.monotonic()
    selected_page_numbers = select_pages(pages, category, top_n=settings.PAGE_ROUTING_TOP_N)
    page_routing_seconds = time.monotonic() - t0
    logger.info(f"pipeline[2/4]: done in {page_routing_seconds:.4f}s — pages={selected_page_numbers}")

    # ── Stage 3: image rendering ──────────────────────────────────────────────
    logger.info("pipeline[3/4]: image rendering — start")
    t0 = time.monotonic()
    try:
        rendered_images = render_pages(pdf_bytes, selected_page_numbers)
    except Exception as exc:
        logger.exception("pipeline[3/4]: image rendering failed")
        raise HTTPException(status_code=400, detail=f"Failed to render PDF pages: {exc}")
    image_rendering_seconds = time.monotonic() - t0
    logger.info(f"pipeline[3/4]: done in {image_rendering_seconds:.2f}s")

    page_markdown_by_num = {p["page_number"]: p.get("markdown", "") for p in pages}
    selected_markdowns = [
        {"page_number": n, "markdown": page_markdown_by_num.get(n, "")}
        for n in selected_page_numbers
    ]

    # ── Stage 4: VLM extraction ───────────────────────────────────────────────
    logger.info("pipeline[4/4]: VLM extraction — start")
    t0 = time.monotonic()
    try:
        vlm_result = await extract_fields(
            category=category,
            field_names=field_names,
            page_markdowns=selected_markdowns,
            page_images=rendered_images,
        )
    except httpx.ProxyError:
        raise HTTPException(status_code=502, detail="Proxy unreachable — cannot reach VLM service")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="VLM service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM service returned HTTP {exc.response.status_code}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"VLM returned invalid JSON: {exc}")
    except Exception as exc:
        logger.exception("pipeline[4/4]: unexpected VLM error")
        raise HTTPException(status_code=502, detail="VLM extraction failed unexpectedly")
    vlm_extraction_seconds = time.monotonic() - t0
    logger.info(f"pipeline[4/4]: done in {vlm_extraction_seconds:.2f}s")

    total_seconds = time.monotonic() - pipeline_start
    stage_timings = StageTimings(
        docling_seconds=round(docling_seconds, 3),
        page_routing_seconds=round(page_routing_seconds, 4),
        image_rendering_seconds=round(image_rendering_seconds, 3),
        vlm_extraction_seconds=round(vlm_extraction_seconds, 3),
    )
    logger.info(
        "pipeline: complete",
        extra={
            "total_seconds": round(total_seconds, 2),
            "stage_timings": stage_timings.model_dump(),
            "pages_used": selected_page_numbers,
            "low_confidence_count": len(vlm_result.get("low_confidence_fields", [])),
        },
    )

    return ExtractionResponse(
        fields=vlm_result.get("fields", {}),
        low_confidence_fields=vlm_result.get("low_confidence_fields", []),
        pages_used=selected_page_numbers,
        processing_time_seconds=round(total_seconds, 3),
        stage_timings=stage_timings,
    )
