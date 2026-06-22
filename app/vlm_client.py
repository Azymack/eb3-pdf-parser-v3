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
        f"List uncertain or ambiguous fields in low_confidence_fields."
    )

    user_text = (
        f"Extract all fields for this {display_category} insurance plan.\n\n"
        f"**Fields to extract:**\n{fields_list}\n\n"
        f"**Document text (per page):**\n\n{markdown_sections}"
    )

    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in page_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img['image_b64']}"},
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
        httpx.ConnectError: VLM service unreachable
        httpx.HTTPStatusError: VLM returned non-2xx
        ValueError: response content is not parseable JSON
    """
    settings = get_settings()
    schema = _build_extraction_schema(field_names)
    messages = _build_messages(category, field_names, page_markdowns, page_images)

    payload = {
        "model": settings.VLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "extra_body": {"guided_json": schema},
    }

    endpoint = f"{settings.VLM_BASE_URL}/v1/chat/completions"
    logger.info("vlm_client: starting extraction", extra={"category": category, "endpoint": endpoint})

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if response.status_code != 200:
        logger.error(
            "vlm_client: VLM returned error",
            extra={"status_code": response.status_code, "body_preview": response.text[:500]},
        )
        response.raise_for_status()

    try:
        raw_content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(raw_content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError(f"VLM returned unparseable content: {exc}") from exc

    logger.info(
        "vlm_client: extraction complete",
        extra={"low_confidence_count": len(result.get("low_confidence_fields", []))},
    )
    return result
