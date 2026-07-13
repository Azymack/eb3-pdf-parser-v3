"""VLM extraction via vLLM's OpenAI-compatible chat completions endpoint.

Uses guided_json decoding to constrain the response to the category's field schema,
eliminating post-hoc JSON repair.

System prompt composition
-------------------------
Rather than one monolithic prompt for every category, the system prompt is
assembled from three blocks:

  _BASE_PROMPT          — universal extraction instructions (all categories)
  _SINGLE_COLUMN_PROMPT — in/out-of-network table guidance (dental, vision, health, health_3tier)

Prescription drug (RX) fields are NOT extracted here: health and health_3tier
RX fields are handled by the dedicated structured extraction in rx_extractor.py.

Adding category-specific guidance in future is a one-line change to the
relevant frozenset plus a new prompt constant.
"""
import json
import logging
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0)

# ---------------------------------------------------------------------------
# Prompt composition constants
# ---------------------------------------------------------------------------

# Categories that have per-service In-Network / Out-of-Network cost columns.
_NETWORK_TABLE_CATEGORIES: frozenset[str] = frozenset({
    "dental", "vision", "health", "health_3tier",
})

# Block 1 — universal (all categories).
# {display_category} is filled in at build time.
_BASE_PROMPT = (
    "You are an insurance document extraction specialist. "
    "Extract the specified fields from the provided {display_category} insurance plan document. "
    "Use null for any field not present or not applicable. "
    "List uncertain or ambiguous fields in low_confidence_fields. "
    "You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences. "
    "Start your response with {{ and end with }}."
)

# Block 2 — in/out-of-network single-column table guidance.
# Relevant for categories that have separate In-Network and Out-of-Network fields
# (dental, vision, health, health_3tier).
_SINGLE_COLUMN_PROMPT = (
    "\n\nIMPORTANT — single-column benefit tables: Some insurance documents present benefit "
    "sections (e.g., Preventive Services, Basic Services, Major Services) with a SINGLE "
    "'What You Pay' column rather than separate In-Network and Out-of-Network columns. "
    "This layout means the stated cost applies equally to BOTH networks. "
    "When a benefit section has only one cost column and the document elsewhere shows "
    "out-of-network deductible or annual maximum values (confirming OON coverage exists), "
    "populate BOTH the In-Network AND the Out-of-Network fields for each benefit row "
    "with that same single value. "
    "Do NOT leave Out-of-Network blank just because no separate Out-of-Network column "
    "is visible in the service table — check whether the plan has OON cost-share details "
    "and, if so, treat a missing OON service column as 'same as In-Network'."
)

def _build_system_prompt(category: str, display_category: str) -> str:
    """Compose a category-appropriate system prompt from the relevant blocks."""
    parts = [_BASE_PROMPT.format(display_category=display_category)]
    if category in _NETWORK_TABLE_CATEGORIES:
        parts.append(_SINGLE_COLUMN_PROMPT)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Schema and message builders
# ---------------------------------------------------------------------------

async def _chat_completion(messages: list[dict], *, extra_body: dict | None = None) -> str:
    """POST to the VLM chat completions endpoint and return message content."""
    settings = get_settings()
    endpoint = f"{settings.VLM_BASE_URL}/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": settings.VLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "seed": 0,  # pin sampling seed — reduces run-to-run output variance
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.ConnectError as exc:
        logger.error("vlm_client: VLM unreachable", extra={"error": str(exc)})
        raise

    if response.status_code != 200:
        logger.error(
            "vlm_client: VLM returned error",
            extra={"status_code": response.status_code, "body_preview": response.text[:500]},
        )
        response.raise_for_status()

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"VLM returned unexpected response shape: {exc}") from exc


async def complete_prompt(prompt: str) -> str:
    """Send a plain-text prompt to the VLM and return the model's reply."""
    logger.info("vlm_client: llm_test prompt")
    return await _chat_completion([{"role": "user", "content": prompt}])


def _build_extraction_schema(field_names: list[str]) -> dict:
    """JSON Schema passed to vLLM's guided_json parameter.

    Each category field is a nullable string. low_confidence_fields is a list
    of field names where the model flagged uncertainty.
    """
    field_props = {
        name: {"anyOf": [{"type": "string"}, {"type": "null"}]}
        for name in field_names
    }
    return {
        "type": "object",
        "properties": {
            "fields": {
                "type": "object",
                "properties": field_props,
                "additionalProperties": False,
            },
            "low_confidence_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["fields", "low_confidence_fields"],
        "additionalProperties": False,
    }


def _build_messages(
    category: str,
    field_names: list[str],
    page_markdowns: list[dict],
    page_images: list[dict],
) -> list[dict]:
    markdown_sections = "\n\n---\n\n".join(
        f"## Page {p['page_number']}\n\n{p['markdown']}"
        for p in page_markdowns
    )
    fields_list = "\n".join(f"- {f}" for f in field_names)
    display_category = category.replace("_", " ")

    system_prompt = _build_system_prompt(category, display_category)

    user_text = (
        f"Extract all fields for this {display_category} insurance plan.\n\n"
        f"**Fields to extract:**\n{fields_list}\n\n"
        f"**Document text (per page):**\n\n{markdown_sections}\n\n"
        'Return ONLY raw JSON with this exact structure: '
        '{"fields": {"Field Name": "value or null", ...}, "low_confidence_fields": ["Field Name", ...]}'
    )

    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in page_images:
        mime = img.get("mime_type", "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img['image_b64']}"},
        })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


async def extract_fields(
    category: str,
    field_names: list[str],
    page_markdowns: list[dict],
    page_images: list[dict],
) -> dict[str, Any]:
    """Call the VLM and return {"fields": {...}, "low_confidence_fields": [...]}.

    Raises:
        httpx.ConnectError: VLM service is not reachable
        httpx.HTTPStatusError: VLM returned non-2xx
        ValueError: response content is not parseable JSON
    """
    settings = get_settings()
    schema = _build_extraction_schema(field_names)
    messages = _build_messages(category, field_names, page_markdowns, page_images)

    logger.info(
        "vlm_client: starting extraction",
        extra={
            "category": category,
            "endpoint": f"{settings.VLM_BASE_URL}/v1/chat/completions",
        },
    )

    try:
        raw_content = await _chat_completion(messages, extra_body={"guided_json": schema})
        # Strip markdown code fences in case the model wraps its output despite instructions.
        cleaned = raw_content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "vlm_client: JSON parse failed",
            extra={"content_preview": (raw_content or "")[:300]},
        )
        raise ValueError(f"VLM returned unparseable content: {exc}") from exc

    logger.info(
        "vlm_client: extraction complete",
        extra={"low_confidence_count": len(result.get("low_confidence_fields", []))},
    )
    return result
