import asyncio
import logging
import time
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .auth import verify_token
from .config import get_settings
from .docling_client import convert_pdf
from .image_renderer import render_pages
from .page_router import select_pages, select_rx_pages
from .post_process import apply_post_processing, vlm_field_names
from .rx_extractor import (
    RX_EXTRACTOR_CATEGORIES,
    extract_rx_fields,
    rx_owned_fields,
    suppress_medical_deductible_echo,
)
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


def _mirror_ct_into_major_diagnostics(
    fields: dict,
    field_names: list[str],
) -> None:
    """Overwrite Major Diagnostics with the CT/PET/MRI (advanced imaging) value.

    Downstream consumers of this API read 'Major Diagnostics' as CT/PT/MRI, so
    the final response mirrors the dedicated imaging field into it — for every
    network prefix present in the category (health and health_3tier).
    """
    for prefix in ("Designated Network", "In-Network", "Out-of-Network"):
        md = f"{prefix} Major Diagnostics"
        ct = f"{prefix} CT scan, PT scan, MRI"
        if md in field_names and ct in field_names:
            fields[md] = fields.get(ct)


# Placeholder phrases the VLM occasionally invents instead of returning null.
_PLACEHOLDER_VALUES: frozenset[str] = frozenset({
    "not specified", "no specific value provided", "not provided",
    "not mentioned", "unknown", "see document",
})


def _nulls_to_empty(fields: dict) -> dict[str, str]:
    """Coerce all field values to str for API output.

    Rules:
      - None  → ""  (field not on this plan)
      - str   → str (pass through, including NOT_FOUND sentinel)
      - dict  → "Key: Value / Key: Value" (VLM sometimes returns nested objects
                for fields with multiple sub-types, e.g. Root Canal costs per
                tooth type; flatten to a readable string rather than crashing)
      - other → str(v)  (integers, booleans, etc. coerced defensively)
    """
    result: dict[str, str] = {}
    for k, v in fields.items():
        if v is None:
            result[k] = ""
        elif isinstance(v, str):
            result[k] = "" if v.strip().lower() in _PLACEHOLDER_VALUES else v
        elif isinstance(v, dict):
            parts = [f"{sk}: {sv}" for sk, sv in v.items() if sv is not None]
            result[k] = " / ".join(parts)
        elif isinstance(v, list):
            result[k] = " / ".join(str(x) for x in v if x is not None)
        else:
            result[k] = str(v)
    return result


@app.post(
    "/extract_json_v2",
    summary="Extract insurance fields from a PDF",
    description=(
        "Extract structured data from an insurance benefit PDF.\n\n"
        "**Default response** (`include_metadata=false`): the response body is the "
        "extracted fields object directly — a flat JSON object where every key is a "
        "field name and every value is a string (empty string when the field is not "
        "present on the plan).\n\n"
        "**With `?include_metadata=true`**: the full pipeline response is returned, "
        "including `fields`, `low_confidence_fields`, `pages_used`, "
        "`processing_time_seconds`, and `stage_timings`."
    ),
    responses={
        200: {
            "description": (
                "Extracted data. Shape depends on `include_metadata`:\n"
                "- `false` (default): `{\"Field Name\": \"value\", ...}` — flat fields object\n"
                "- `true`: full metadata wrapper with `fields`, `pages_used`, `stage_timings`, etc."
            )
        },
        400: {"description": "Bad request (invalid category, empty file, non-PDF)"},
        401: {"description": "Missing or invalid API token"},
        413: {"description": "File exceeds maximum allowed size"},
        502: {"description": "Upstream service error (docling or VLM)"},
        504: {"description": "Pipeline exceeded timeout"},
    },
)
async def extract_json_v2(
    _token: Annotated[str, Depends(verify_token)],
    file: UploadFile = File(..., description="PDF document"),
    category: str = Form(
        ...,
        description=f"Insurance category. Valid values: {', '.join(VALID_CATEGORIES_SORTED)}",
    ),
    include_metadata: bool = Query(
        default=False,
        description=(
            "When true, wrap the response in a metadata envelope containing "
            "`fields`, `low_confidence_fields`, `pages_used`, "
            "`processing_time_seconds`, and `stage_timings`. "
            "Default false returns the fields object directly."
        ),
    ),
):
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

    if include_metadata:
        return JSONResponse(result.model_dump())
    return JSONResponse(result.fields)


@app.post("/llm_test", response_model=LlmTestResponse)
async def llm_test(
    _token: Annotated[str, Depends(verify_token)],
    prompt: str = Form(..., description="Prompt to send to the VLM"),
) -> LlmTestResponse:
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    try:
        content = await complete_prompt(prompt)
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="VLM service unreachable")
    except httpx.HTTPStatusError as exc:
        body_preview = exc.response.text[:300].strip()
        raise HTTPException(
            status_code=502,
            detail=f"VLM service returned HTTP {exc.response.status_code}: {body_preview}",
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
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="docling-service unreachable")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"docling-service returned HTTP {exc.response.status_code}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"docling-service returned invalid response: {exc}")
    except Exception:
        logger.exception("pipeline[1/4]: unexpected docling error")
        raise HTTPException(status_code=502, detail="docling-service failed unexpectedly")
    docling_seconds = time.monotonic() - t0
    logger.info(f"pipeline[1/4]: done in {docling_seconds:.2f}s")

    ocr_mode_used = docling_result.get("_ocr_mode_used", settings.DOCLING_OCR_MODE)
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
    # For RX categories, pharmacy fields are extracted by a dedicated structured
    # RX call (app/rx_extractor.py) that runs concurrently with the main call.
    # The main call never sees RX fields.
    logger.info("pipeline[4/4]: VLM extraction — start")
    t0 = time.monotonic()
    vlm_fields = vlm_field_names(field_names)
    rx_enabled = category in RX_EXTRACTOR_CATEGORIES
    rx_fields: dict[str, str] | None = None
    if rx_enabled:
        owned = set(rx_owned_fields(category))
        vlm_fields = [f for f in vlm_fields if f not in owned]
        rx_page_numbers = select_rx_pages(pages) or selected_page_numbers
        try:
            rx_images = (
                rendered_images
                if rx_page_numbers == selected_page_numbers
                else render_pages(pdf_bytes, rx_page_numbers)
            )
        except Exception:
            logger.exception("pipeline[4/4]: RX page rendering failed — "
                             "falling back to category pages")
            rx_page_numbers = selected_page_numbers
            rx_images = rendered_images
        rx_markdowns = [
            {"page_number": n, "markdown": page_markdown_by_num.get(n, "")}
            for n in rx_page_numbers
        ]
        logger.info(f"pipeline[4/4]: RX extraction pages={rx_page_numbers}")

    try:
        main_coro = extract_fields(
            category=category,
            field_names=vlm_fields,
            page_markdowns=selected_markdowns,
            page_images=rendered_images,
        )
        if rx_enabled:
            gathered = await asyncio.gather(
                main_coro,
                extract_rx_fields(category, rx_markdowns, rx_images),
                return_exceptions=True,
            )
            if isinstance(gathered[0], BaseException):
                raise gathered[0]
            vlm_result = gathered[0]
            if isinstance(gathered[1], BaseException):
                logger.error(
                    "pipeline[4/4]: structured RX extraction failed — "
                    "RX fields will be empty",
                    exc_info=gathered[1],
                )
                rx_fields = {f: "" for f in rx_owned_fields(category)}
            else:
                rx_fields = gathered[1]
        else:
            vlm_result = await main_coro
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="VLM service unreachable")
    except httpx.HTTPStatusError as exc:
        body_preview = exc.response.text[:300].strip()
        raise HTTPException(
            status_code=502,
            detail=f"VLM service returned HTTP {exc.response.status_code}: {body_preview}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"VLM returned invalid JSON: {exc}")
    except Exception:
        logger.exception("pipeline[4/4]: unexpected VLM error")
        raise HTTPException(status_code=502, detail="VLM extraction failed unexpectedly")
    vlm_extraction_seconds = time.monotonic() - t0
    logger.info(f"pipeline[4/4]: done in {vlm_extraction_seconds:.2f}s")

    # ── Post-processing: compute derived fields, then serialize ───────────────
    raw_fields = vlm_result.get("fields", {})
    processed_fields = apply_post_processing(raw_fields, field_names)
    if rx_fields is not None:
        suppress_medical_deductible_echo(rx_fields, raw_fields)
        processed_fields.update(rx_fields)
    _mirror_ct_into_major_diagnostics(processed_fields, field_names)
    serialized_fields = _nulls_to_empty(processed_fields)
    serialized_fields = {k: serialized_fields[k] for k in field_names if k in serialized_fields}

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
        fields=serialized_fields,
        low_confidence_fields=vlm_result.get("low_confidence_fields", []),
        pages_used=selected_page_numbers,
        processing_time_seconds=round(total_seconds, 3),
        stage_timings=stage_timings,
        ocr_mode=ocr_mode_used,
    )
