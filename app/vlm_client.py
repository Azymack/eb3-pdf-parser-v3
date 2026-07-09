"""VLM extraction via vLLM's OpenAI-compatible chat completions endpoint.

Uses guided_json decoding to constrain the response to the category's field schema,
eliminating post-hoc JSON repair.

System prompt composition
-------------------------
Rather than one monolithic prompt for every category, the system prompt is
assembled from three blocks:

  _BASE_PROMPT          — universal extraction instructions (all categories)
  _SINGLE_COLUMN_PROMPT — in/out-of-network table guidance (dental, vision, health, health_3tier)
  _RX_PROMPT            — prescription drug field guidance (health, health_3tier only)

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

# Categories that have prescription drug RX fields.
_RX_CATEGORIES: frozenset[str] = frozenset({
    "health", "health_3tier",
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

# Block 3 — prescription drug RX field guidance.
# ONLY for health and health_3tier. Never sent to dental, vision, std, ltd, etc.
_RX_PROMPT = (
    "\n\nIMPORTANT — RX fields: two supply channels.\n\n"
    "RETAIL / 30-DAY SUPPLY → In-Network RX, Out-of-Network RX, Designated Network RX.\n"
    "  Format: 'Label: cost' per tier separated by ' / ', using the carrier's own tier labels exactly.\n"
    "  Example: 'Tier 1 (Generic): $10 / Tier 2 (Preferred Brand): $45 / Tier 3 (Non-preferred): $65'\n\n"
    "MAIL ORDER / EXTENDED SUPPLY → In-Network Mail Order RX, Out-of-Network Mail Order RX, "
    "Designated Network Mail Order RX.\n"
    "  Format: cost values ONLY separated by ' / ', NO tier labels, in the same tier order as the retail field.\n"
    "  Example: '$25 / $90 / $130'  <- three tiers, cost values only, no labels.\n"
    "  Identifying mail order: look for 'mail order', 'mail-order service', '90-day supply', "
    "'90-day fill', '100-day supply', or a dedicated mail order column.\n"
    "  When a document cell shows both retail and mail-order in the same cell "
    "(e.g. '$10 / $25 per fill' or '$10 retail / $25 mail'), put the 30-day cost in the Retail field "
    "and the extended-supply cost in the Mail Order field.\n"
    "  If no separate mail-order/extended-supply pricing exists for a network direction, return empty string "
    "(not null, not 'Not covered') for that Mail Order field. "
    "Do NOT copy retail costs into Mail Order fields as a substitute.\n\n"
    "LIST-FORMAT RX and MULTI-PAGE COLLECTION: Some documents list prescription tiers as individual "
    "rows rather than a column-based table. Each row names a specific tier AND a supply channel, e.g.:\n"
    "  'Most generic items (Tier 1) at a Plan Pharmacy ... $20 (30-day)'  <- retail row\n"
    "  'Most generic items (Tier 1) refills through our mail-order service ... $40 (100-day)'  <- mail order row\n"
    "  'Most brand-name items (Tier 2) at a Plan Pharmacy ... $100 (30-day)'  <- retail row\n"
    "In this format:\n"
    "  - Rows containing 'at a Plan Pharmacy', 'retail pharmacy', '30-day supply', or similar "
    "are RETAIL rows -> collect their tier labels and costs into the Retail RX field.\n"
    "  - Rows containing 'mail-order service', 'mail order', '90-day', '100-day', or similar "
    "are MAIL ORDER rows -> collect their costs (no labels) into the Mail Order RX field.\n"
    "  - CRITICAL: RX sections often span multiple pages with a '(continues)' marker at the bottom. "
    "You MUST scan ALL provided pages and collect every retail row and every mail-order row "
    "regardless of which page they appear on. Do not stop at a page boundary.\n\n"
    "PREFERRED vs NON-PREFERRED ROWS MUST NOT BE COLLAPSED: Many plans list separate rows for "
    "Generic (Preferred), Generic (Non-Preferred), Brand (Preferred), Brand (Non-Preferred), "
    "Specialty (Preferred), Specialty (Non-Preferred). Preserve every row as its own tier label "
    "in the consolidated RX field. Do NOT collapse these into only Generic / Brand / Specialty.\n"
    "This rule applies independently to each network direction (In-Network, Out-of-Network, "
    "Designated Network): if Out-of-Network has non-preferred rows, include them explicitly in "
    "Out-of-Network RX too.\n"
    "Treat spelling/format variants as equivalent labels: non-preferred, non preferred, "
    "nonpreferred, non-preffered, speciality vs specialty.\n\n"
    "EMPTY vs NOT COVERED:\n"
    "  - If ALL tiers for a network direction are explicitly 'Not covered' (e.g. the Out-of-Network "
    "pharmacy column says 'Not covered' for every tier), write 'Not covered' as the field value.\n"
    "  - If only SOME tiers say 'Not covered', omit those tiers but keep the field populated "
    "with the remaining covered tiers.\n"
    "  - If no mail-order pricing exists at all for a network direction (not the same as 'Not covered'), "
    "return empty string. NEVER write 'null' as a string value.\n\n"
    "MULTIPLE COST ROWS WITHIN ONE TIER: Some plans list several cost variants inside a single "
    "tier cell — e.g. Generic drugs shown as 'Preventive: No Charge / Condition Care Rx: $4 / "
    "All Other Generic: $20'. Report ALL cost values for the tier, joined by ' / ', WITHOUT "
    "the sub-labels — but SKIP 'Preventive' rows (the No Charge preventive-mandate sub-tier). "
    "Example: that Generic cell becomes 'Generic: $4 / $20' and a Preferred Brand cell "
    "'Condition Care Rx: $50 / All Other Preferred Brand: Deductible + $100' becomes "
    "'Preferred Brand: $50 / Deductible + $100'. "
    "Keep the tier label ONCE at the front; never repeat sub-program names.\n\n"
    "PER-TIER RETAIL vs MAIL ORDER ATTRIBUTION: Decide tier by tier. A tier contributes to "
    "Mail Order RX ONLY if the document explicitly shows a mail-order price for THAT tier — "
    "a value marked '(mail order)', a mail-order column entry, or a 90/100-day-supply price. "
    "If a tier shows only a retail price, or its Limitations column says something like "
    "'Up to a 30-day supply (retail)', that tier has NO mail-order benefit: skip it in "
    "Mail Order RX entirely — never copy its retail cost there. "
    "Example: Generic '$5 (retail), $10 (mail order)' / Brand '$15 (retail), $30 (mail order)' / "
    "Specialty '10% coinsurance up to $250' with limitation 'Up to a 30-day supply (retail)' "
    "-> In-Network Mail Order RX = '$10 / $30' (specialty omitted — retail only).\n\n"
    "NEVER CALCULATE MAIL ORDER PRICES: Only populate Mail Order RX with dollar amounts "
    "explicitly PRINTED in the document as mail-order prices. Limitation notes like "
    "'Up to 30 day supply for retail, 90 day supply for mail order at 2 times the retail amount' "
    "or '90-day supply at 2.5x copay' are multiplication RULES, not price lists — when the only "
    "mail-order information is such a rule, Mail Order RX MUST be the empty string ''. "
    "Do NOT compute $8 from '$4 at 2 times', do NOT double any retail price. "
    "If you cannot point to a printed mail-order dollar amount in a table cell, "
    "the answer is '' (empty).\n\n"
    "90-DAY SUPPLY AT RETAIL PHARMACY ≠ MAIL ORDER: A '90-day supply' or 'extended supply' "
    "option available at a retail/network pharmacy is NOT mail order. "
    "Only populate Mail Order RX fields when the document shows a dedicated mail-order pharmacy "
    "service (phrases like 'mail order', 'mail-order service', 'home delivery pharmacy', "
    "'CVS Caremark mail service'). If no such mail-order service exists, return empty string.\n\n"
    "Example -- 4-tier PPO with separate retail (30-day) and mail-order (90-day) columns:\n"
    "  In-Network RX = 'Tier 1 (Generic): $10 / Tier 2 (Preferred Brand): $45 / "
    "Tier 3 (Non-preferred): $65 / Tier 4 (Specialty): 50% up to $150'\n"
    "  In-Network Mail Order RX = '$20 / $90 / $130'  (cost-only, 90-day prices, no labels)\n"
    "  Out-of-Network RX = 'Not covered' (OON pharmacy not covered)  "
    "Out-of-Network Mail Order RX = 'Not covered'\n\n"
    "Example -- 5-tier HMO with single cost column (no separate mail order section):\n"
    "  In-Network RX = 'Generic: $15 / Preferred Brand: Deductible, then $50 / "
    "Non-preferred Brand: Deductible, then $75 / Preferred Specialty: Deductible, then 50% up to $100 / "
    "Non-preferred Specialty: Deductible, then 50% up to $150'\n"
    "  In-Network Mail Order RX = ''  (no 90-day column found)\n\n"
    "Example -- 3-tier plan where only Designated Network has mail order:\n"
    "  Designated Network Mail Order RX = '$25 / $75 / $125'  (cost-only, no labels)\n"
    "  In-Network Mail Order RX = ''  (no In-Network mail order benefit shown)\n\n"
    "Example -- list-format plan spanning two pages (each tier is a separate row):\n"
    "  Page 1 rows: Tier 1 at pharmacy $20/30-day | Tier 1 mail-order $40/100-day | "
    "Tier 2 at pharmacy $100/30-day | (continues)\n"
    "  Page 2 rows: Tier 2 mail-order $200/100-day | Tier 4 at pharmacy 20% coinsurance/30-day\n"
    "  -> In-Network RX = 'Tier 1 (Generic): $20 / Tier 2 (Brand): $100 / "
    "Tier 4 (Specialty): 20% Coinsurance'  (ALL retail rows from ALL pages)\n"
    "  -> In-Network Mail Order RX = '$40 / $200'  (ALL mail-order rows from ALL pages, cost-only)"

    "\n\nPREFERRED NETWORK PHARMACY: Some plans have a 'Preferred Network Pharmacy' column "
    "separate from the standard 'In-Network Pharmacy' column.\n"
    "  Preferred Network RX: If that column exists, extract its 30-day RETAIL costs only "
    "(ignore any home delivery / 90-day costs in the same cell). "
    "Use the same 'Label: cost' format as In-Network RX. "
    "If no Preferred Network Pharmacy column exists, return null.\n"
    "  In-Network RX: always extract from the 'In-Network Pharmacy' column, 30-day retail only.\n"
    "  In-Network Mail Order RX: for plans with a Preferred Network column, the home delivery (90-day) "
    "costs appear in the Preferred Network column — extract them here, cost-only, no labels.\n"
    "  Example (Anthem, 3-col): Preferred col for Tier 2 says '$80 copay (retail) and $200 copay (home delivery)' "
    "and In-Network col says '$90 copay (retail only)':\n"
    "    Preferred Network RX = 'Tier 2 - Typically Preferred Brand: $80' (retail only)\n"
    "    In-Network RX        = 'Tier 2 - Typically Preferred Brand: $90' (retail only)\n"
    "    In-Network Mail Order RX = '$200' (home delivery from Preferred col, cost-only)"
    "\n\nFINAL CHECK before answering: for EVERY value you wrote in a Mail Order RX field, "
    "verify that the exact dollar amount appears verbatim in the document as a printed "
    "mail-order price. If any amount does not appear verbatim (e.g. you doubled a retail "
    "price because of a '2 times the retail amount' note), set that Mail Order RX field "
    "to '' instead."
)


def _build_system_prompt(category: str, display_category: str) -> str:
    """Compose a category-appropriate system prompt from the relevant blocks."""
    parts = [_BASE_PROMPT.format(display_category=display_category)]
    if category in _NETWORK_TABLE_CATEGORIES:
        parts.append(_SINGLE_COLUMN_PROMPT)
    if category in _RX_CATEGORIES:
        parts.append(_RX_PROMPT)
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
