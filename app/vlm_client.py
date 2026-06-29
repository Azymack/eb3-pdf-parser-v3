"""VLM extraction via vLLM's OpenAI-compatible chat completions endpoint.

Uses guided_json decoding to constrain the response to the category's field schema,
eliminating post-hoc JSON repair.
"""
import json
import logging
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=5.0)


async def _chat_completion(messages: list[dict], *, extra_body: dict | None = None) -> str:
    """POST to the VLM chat completions endpoint and return message content."""
    settings = get_settings()
    proxy = settings.outbound_proxy_url
    endpoint = f"{settings.VLM_BASE_URL}/v1/chat/completions"

    payload: dict[str, Any] = {
        "model": settings.VLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
    }
    if extra_body:
        payload["extra_body"] = extra_body

    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=_TIMEOUT) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except httpx.ProxyError as exc:
        logger.error("vlm_client: proxy unreachable", extra={"error": str(exc)})
        raise
    except httpx.ConnectError as exc:
        logger.error(
            "vlm_client: VLM unreachable (two-hop path: proxy→service)",
            extra={"error": str(exc)},
        )
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
    logger.info(
        "vlm_client: llm_test prompt",
        extra={"via_proxy": get_settings().outbound_proxy_url is not None},
    )
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

    system_prompt = (
        f"You are an insurance document extraction specialist. "
        f"Extract the specified fields from the provided {display_category} insurance plan document. "
        f"Use null for any field not present or not applicable. "
        f"List uncertain or ambiguous fields in low_confidence_fields. "
        f"You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences. "
        f"Start your response with {{ and end with }}.\n\n"
        f"IMPORTANT — single-column benefit tables: Some insurance documents present benefit "
        f"sections (e.g., Preventive Services, Basic Services, Major Services) with a SINGLE "
        f"'What You Pay' column rather than separate In-Network and Out-of-Network columns. "
        f"This layout means the stated cost applies equally to BOTH networks. "
        f"When a benefit section has only one cost column and the document elsewhere shows "
        f"out-of-network deductible or annual maximum values (confirming OON coverage exists), "
        f"populate BOTH the In-Network AND the Out-of-Network fields for each benefit row "
        f"with that same single value. "
        f"Do NOT leave Out-of-Network blank just because no separate Out-of-Network column "
        f"is visible in the service table — check whether the plan has OON cost-share details "
        f"and, if so, treat a missing OON service column as 'same as In-Network'.\n\n"
        f"IMPORTANT — RX tier fields (In-Network RX, Out-of-Network RX, "
        f"In-Network Mail Order RX, Out-of-Network Mail Order RX, and Designated Network variants): "
        f"These fields capture ALL prescription drug tiers in a single string. "
        f"List every tier the carrier shows for that network direction, separated by ' / '. "
        f"Use the format 'Label: cost' for each tier, using the carrier's own tier labels exactly "
        f"(e.g. 'Tier 1 (Generic)', 'Preferred Brand', 'Non-preferred Brand', 'Tier 1a', 'Specialty'). "
        f"RETAIL vs MAIL ORDER separation: For the Retail RX field use ONLY the retail/30-day cost per tier. "
        f"For the Mail Order RX field use ONLY the mail-order/90-day cost per tier. "
        f"When a table row shows both retail and mail-order costs in the same cell or row, "
        f"put only the retail cost in the Retail RX field and only the mail-order cost in the Mail Order RX field — "
        f"never include both in the same field. "
        f"If the document shows no separate mail-order pricing column or section for a given network direction, "
        f"return empty string (not null, not 'Not covered') for that network's Mail Order RX field. "
        f"Do NOT copy retail costs into the Mail Order RX field as a substitute. "
        f"EMPTY vs NOT COVERED: If an entire network direction has no drug benefit "
        f"(drugs are simply not covered for that network), return empty string for that network's RX field — "
        f"do not write 'Not covered' as the field value. "
        f"NEVER write the word 'null' as a string value — use empty string instead. "
        f"Omit individual tiers that explicitly say 'Not covered' but keep the field populated "
        f"with the remaining covered tiers. "
        f"Example — 4-tier PPO, retail/mail-order columns are separate: "
        f"In-Network RX = 'Tier 1 (Generic): $10 / Tier 2 (Preferred Brand): $45 / "
        f"Tier 3 (Non-preferred): $65 / Tier 4 (Specialty): 50% up to $150' "
        f"In-Network Mail Order RX = 'Tier 1: $20 / Tier 2: $90 / Tier 3: $130' "
        f"Out-of-Network RX = '' (empty — OON drugs not covered) "
        f"Out-of-Network Mail Order RX = '' (empty). "
        f"Example — 5-tier HMO with single cost column (no separate mail order): "
        f"In-Network RX = 'Generic: $15 / Preferred Brand: Deductible, then $50 / "
        f"Non-preferred Brand: Deductible, then $75 / Preferred Specialty: Deductible, then 50% up to $100 / "
        f"Non-preferred Specialty: Deductible, then 50% up to $150' "
        f"In-Network Mail Order RX = '' (empty — no separate mail order column). "
        f"Example — 3-tier plan where only Designated Network has mail order (In-Network does not): "
        f"Designated Network Mail Order RX = 'Tier 1: $25 / Tier 2: $75 / Tier 3: $125' "
        f"In-Network Mail Order RX = '' (empty — no In-Network mail order benefit shown)."
    )

    user_text = (
        f"Extract all fields for this {display_category} insurance plan.\n\n"
        f"**Fields to extract:**\n{fields_list}\n\n"
        f"**Document text (per page):**\n\n{markdown_sections}\n\n"
        f'Return ONLY raw JSON with this exact structure: '
        f'{{"fields": {{"Field Name": "value or null", ...}}, "low_confidence_fields": ["Field Name", ...]}}'
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

    Routes through the outbound proxy when USE_OUTBOUND_PROXY=true (same as docling-service).
    Raises:
        httpx.ProxyError: proxy is unreachable
        httpx.ConnectError: proxy reachable but VLM service is not
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
            "via_proxy": settings.outbound_proxy_url is not None,
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
