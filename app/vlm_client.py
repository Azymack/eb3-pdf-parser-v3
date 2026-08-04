"""VLM extraction via vLLM's OpenAI-compatible chat completions endpoint.

Uses guided_json decoding to constrain the response to the category's field schema,
eliminating post-hoc JSON repair.

System prompt composition
-------------------------
Rather than one monolithic prompt for every category, the system prompt is
assembled from layered blocks:

  _BASE_PROMPT                        — universal extraction instructions (all categories)
  _SINGLE_COLUMN_PROMPT               — in/out-of-network table guidance (dental, vision, health*)
  _HEALTH_FIELD_DEFINITIONS_PROMPT    — service-field semantics (health, health_3tier)
  _HEALTH_3TIER_NETWORK_PROMPT        — three provider-level columns (health_3tier)
  _DENTAL_FIELD_DEFINITIONS_PROMPT    — class-grouped dental tables (dental)
  _VISION_FIELD_DEFINITIONS_PROMPT    — lens / eyeglasses copay semantics (vision)
  _DISABILITY_FIELD_DEFINITIONS_PROMPT — benefit formula + elimination period (ltd, std)
  _LIFE_FIELD_DEFINITIONS_PROMPT      — employee/spouse/child columns (sup_life, term_life)

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

# Health categories get service-field definitions (documents name the same
# benefit rows very differently across carriers).
_HEALTH_CATEGORIES: frozenset[str] = frozenset({
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

# Block 3 — health service-field definitions.
# Carriers name the same benefit rows differently; match fields by MEANING.
_HEALTH_FIELD_DEFINITIONS_PROMPT = (
    "\n\nFIELD DEFINITIONS — match these fields by meaning, not by exact wording; "
    "documents name the same benefit rows differently:\n"
    "- 'Inpatient Surgery': the hospital-stay FACILITY FEE (hospital room / room & "
    "board). Row names include 'If you have a hospital stay - Facility fee (e.g., "
    "hospital room)', 'Hospital Stay', 'Inpatient Hospital Services'. Do NOT use "
    "the physician/surgeon fees row, and NEVER copy the Outpatient Surgery value "
    "here — inpatient and outpatient facility fees are different rows.\n"
    "- 'Outpatient Surgery': the outpatient-surgery FACILITY FEE row (e.g. "
    "'Facility fee (e.g., ambulatory surgery center)'), not physician/surgeon "
    "fees.\n"
    "- 'Newborn Delivery': the 'Childbirth/delivery facility services' row (the "
    "delivery/facility cost), not childbirth professional services.\n"
    "- 'CT scan, PT scan, MRI': the ADVANCED IMAGING row — 'Imaging (CT/PET "
    "scans, MRIs)', 'Advanced Diagnostic Imaging (for example: MRI, PET and CAT "
    "scans)', 'MRI/CT/PET scans'.\n"
    "- 'Major Diagnostics': the basic diagnostic test row — 'Diagnostic test "
    "(x-ray, blood work)', laboratory and x-ray services. Keep this separate from "
    "'CT scan, PT scan, MRI': x-ray/blood work here, advanced imaging there.\n"
    "- 'PCP Visit': the IN-PERSON 'Primary care visit to treat an injury or "
    "illness' cost. Do NOT use virtual/telehealth visit pricing and do NOT use "
    "the preventive care/screening row (often 'No Charge') — those are different "
    "rows. Example: a cell '10% coinsurance. Virtual visits: 0% coinsurance' → "
    "PCP Visit is '10% coinsurance'.\n"
    "- 'Specialist Visit': the in-person specialist office visit cost, same "
    "exclusions as PCP Visit.\n"
    "NEVER output placeholder text such as 'No specific value provided', "
    "'Not provided', or 'See document' as a field value — use null when a value "
    "is not stated.\n"
    "MULTIPLE PLACE-OF-SERVICE SUB-ROWS: when a benefit lists separate costs per "
    "setting (e.g. Office / Freestanding Radiology Center / Outpatient Hospital "
    "for imaging, or Hospital / Ambulatory Surgical Center for outpatient "
    "surgery), include ALL settings in the field value, labeled and joined with "
    "' / '. Example: 'Office: 20% coinsurance / Freestanding Radiology Center: "
    "20% coinsurance / Outpatient Hospital: $500 then 20% coinsurance'. NEVER "
    "keep only the first setting's cost.\n"
    "NO OUT-OF-NETWORK COVERAGE: when the plan's out-of-network service columns "
    "show 'Not covered' (HMO/EPO-style plans), set the Out-of-Network Deductible, "
    "OOP Max, and Coinsurance fields to 'Not covered' — do NOT copy the "
    "In-Network amounts into them. The single-column rule above applies ONLY "
    "when the plan actually covers out-of-network care."
)


# Block 3b — health_3tier provider-level column mapping.
# The main call previously had NO definition of what the three field prefixes
# mean, so the model guessed — often copying the first (cheapest) column into
# the In-Network fields and leaving the middle column unused.
_HEALTH_3TIER_NETWORK_PROMPT = (
    "\n\nTHREE PROVIDER LEVELS: this plan type prices services at THREE "
    "provider levels, shown as three cost columns. Documents name the levels "
    "differently — 'Designated Network / Non-Designated Network / "
    "Out-of-Network', 'In-Network Tier 1 / In-Network Tier 2 / Out-of-Network "
    "Provider', 'Tier 1 - <brand> Network / Tier 2 - <brand> Network / "
    "Non-Network Provider', 'Level 1 Pharmacy-RX Only / In-Network Provider / "
    "Out-of-Network Provider'. Map the columns by POSITION and cost level, "
    "never by name:\n"
    "- FIRST cost column (cheapest level — 'You will pay the least', Tier 1, "
    "Designated, Level 1) → the 'Designated Network ...' fields.\n"
    "- MIDDLE cost column (Tier 2, Non-Designated, standard network — 'You "
    "will pay more') → the 'In-Network ...' fields.\n"
    "- LAST cost column ('You will pay the most', Out-of-Network, "
    "Non-Network, Non-Participating) → the 'Out-of-Network ...' fields.\n"
    "NEVER copy the first column's value into an 'In-Network ...' field when "
    "the middle column prints its own value — each column fills only its own "
    "fields, for EVERY benefit row (deductible, out-of-pocket limit, "
    "coinsurance, office visits, urgent care, surgery, imaging, delivery). "
    "CRITICAL — 'Not Applicable' / 'N/A' / 'Not applicable' in the FIRST "
    "column: copy that EXACTLY into the Designated Network field for that "
    "row. Do NOT substitute the middle (In-Network) column's cost. Anthem "
    "'Level 1 Pharmacy-RX Only' tables print Not Applicable for almost every "
    "medical benefit in column 1 — Designated Network PCP/Specialist/ER/"
    "surgery/imaging/delivery/coinsurance must stay 'Not Applicable' (or "
    "null), while In-Network gets the real medical cost from column 2. Only "
    "prescription-drug rows use real Level 1 pharmacy prices in column 1.\n"
    "When a deductible or out-of-pocket limit is stated ONCE covering the "
    "first two levels together (e.g. 'For Tier 1 and Tier 2 Participating "
    "providers $6,000 person / $12,000 family'), fill BOTH the Designated "
    "Network and In-Network deductible/OOP fields with that same amount. "
    "When the overall deductible is labeled only for In-Network Providers "
    "(and Level 1 is pharmacy-only), leave Designated Network deductible "
    "null — do not copy the In-Network medical deductible into Designated."
)

# Block 4 — dental class-grouped benefit tables.
_DENTAL_FIELD_DEFINITIONS_PROMPT = (
    "\n\nDENTAL CLASS TABLES: benefit summaries often group procedures under "
    "coverage classes with ONE percentage per class (e.g. 'Class I - "
    "Diagnostic/Preventive Services: 100%', 'Class II - Basic Services: 80%', "
    "'Class III - Major Services: 50%'), listing many procedures under each "
    "class heading while printing the percentage only once. The class "
    "percentage applies to EVERY procedure listed under that class — never "
    "leave a listed procedure's field empty just because the percentage is not "
    "printed on its own row. The percentage may be printed beside ANY row in "
    "the middle of the class (e.g. beside 'Nonsurgical Periodontics'); every "
    "procedure from one class heading down to the NEXT class heading belongs "
    "to that class, including rows printed AFTER the percentage (e.g. "
    "'Complex Oral Surgery' listed after the 80% row but before 'Class III' "
    "is still Class II → 80%). Common groupings:\n"
    "- Class I (Diagnostic/Preventive): Exams, Bitewing X-rays and All Other "
    "X-rays (→ X-Rays), Cleanings, Fluoride, Sealants, Space Maintainers, "
    "Palliative Treatment.\n"
    "- Class II (Basic): Basic Restorative (→ Fillings), Simple Extractions, "
    "Endodontics (→ Root Canal), Nonsurgical/Surgical Periodontics "
    "(→ Periodontal Gum Disease), Complex Oral Surgery (→ Oral Surgery), "
    "Posterior Resins.\n"
    "- Class III (Major): Inlays/Onlays/Crowns (→ Crowns), Prosthetics "
    "(→ BOTH Dentures and Bridges), Implants.\n"
    "When one percentage column applies to both networks (or separate "
    "In-Network/Non-Network columns show the same value), fill In-Network and "
    "Out-of-Network fields alike. Use null only for procedures the document "
    "does not cover anywhere."
)

# Block 5 — vision lens field semantics.
# Testers expect the member's lens / eyeglasses CO-PAY (often $10/$20/$25), not
# a MATERIALS table cell of $0 that only means "no extra charge beyond that
# copay". Davis Vision / Aflac-style PDFs print BOTH: CO-PAYMENTS "Eyeglass
# lenses $10" AND MATERIALS "Clear plastic single-vision… : $0".
_VISION_FIELD_DEFINITIONS_PROMPT = (
    "\n\nVISION LENS FIELDS: 'Single Vision Lens', 'Lined Bi-Focal Lens', "
    "'Lined Tri-Focal Lens', and 'Lenticular Lens' are the member's "
    "OUT-OF-POCKET cost for obtaining that STANDARD lens (usually a "
    "copay). These four fields are ONLY for eyeglass lenses — never copy "
    "their values into 'Contact Lens Allowance' or 'Frame Allowance'.\n"
    "PREFERRED SOURCE (use this first):\n"
    "- An 'Eyeglass lenses' / 'Prescription eyeglasses' / 'Prescription "
    "glasses' / 'Eyewear' CO-PAYMENT that covers standard lenses, OR\n"
    "- Wording like '100% after $10 copay', 'covered in full after $20 "
    "eyewear copay', 'at no additional cost after $25 copay', "
    "'Included in Prescription Glasses', 'Included in prescription "
    "eyeglass copay', or 'Eyeglasses (lenses and frame) … $25' that "
    "covers single vision / lined bifocal / lined trifocal / "
    "lenticular.\n"
    "When ONE shared eyeglasses/lens/eyewear / Prescription Glasses "
    "copay covers all standard lens types, put that SAME copay amount "
    "in ALL four In-Network lens fields that the document lists under "
    "that benefit (e.g. '$10' or '$25'). Do this even when individual "
    "lens-type rows leave the In-Network cell blank and only print "
    "Out-of-Network 'Up to $…' / 'Amount over: $…' amounts.\n"
    "'COPAY APPLIES' CELLS (Guardian/VSP benefit summaries): when an "
    "In-Network cell literally says 'Copay applies' / 'Copay Applies', "
    "resolve it to the NUMERIC copay from the Co-Pay section of the "
    "same plan (e.g. 'First Services Provided $0.00', 'Exams $10', "
    "'Materials $25') — output that amount (e.g. '$0' or '$0.00'), "
    "NEVER the literal phrase 'Copay applies', and NEVER the "
    "Out-of-Network 'Amount over: $…' figure (those OON dollars go "
    "only in Out-of-Network fields).\n"
    "DO NOT use a MATERIALS / lens-options cell of '$0' for these "
    "fields when a separate eyeglasses/lens/eyewear CO-PAYMENT also "
    "exists. That '$0' only means no EXTRA materials charge after the "
    "lenses copay — the member still pays the lenses copay. Example: "
    "CO-PAYMENTS 'Eyeglass lenses: $10' + MATERIALS 'Clear plastic "
    "single-vision, lined bifocal, trifocal or lenticular lenses: $0' "
    "→ all four lens fields are '$10', NOT '$0'.\n"
    "Only output '$0' when the document truly states a $0 member cost "
    "for those lenses (including a $0 co-pay / First Services "
    "Provided) AND there is no larger lenses/eyeglasses/eyewear "
    "copay.\n"
    "EYE EXAM: use the routine exam / WellVision Exam copay only "
    "(e.g. '$10'). Do NOT append retinal screening amounts such as "
    "'Up to $39' into Eye Exam.\n"
    "CONTACT LENS / FRAME ALLOWANCE: use the contact-lens material "
    "ALLOWANCE (e.g. '$130', '$150', 'Up to $180') and frame "
    "ALLOWANCE separately. Never fill Contact Lens Allowance with "
    "the eyeglasses lens copay, exam copay, or contact "
    "fitting/evaluation copay — a separate 'Contact lenses instead "
    "of Eyeglasses: $25' COPAY is not the allowance.\n"
    "NEVER use lens add-on / enhancement rows (tinting, scratch "
    "coating, polycarbonate upgrades, progressive, UV, AR, "
    "high-index, polarized, photochromic, scratch-protection plans) "
    "or the contact-lens fitting row as the value for these four "
    "standard lens fields."
)


# Block 6 — LTD / STD benefit formula + elimination period.
_DISABILITY_CATEGORIES: frozenset[str] = frozenset({"ltd", "std"})

_DISABILITY_FIELD_DEFINITIONS_PROMPT = (
    "\n\nDISABILITY BENEFIT FIELDS:\n"
    "- 'Monthly Benefit' (LTD) / 'Weekly Benefit' (STD): always combine "
    "the PERCENTAGE of earnings/salary AND the MAXIMUM dollar amount into "
    "ONE value. Examples: '60% of monthly earnings up to a maximum of "
    "$10,000', '60% of monthly earnings $10,000 (Maximum Amount)', "
    "'60% of salary to maximum $10000/month', '60% of weekly earnings up "
    "to a maximum of $1,000/week'. NEVER return only the max dollar "
    "amount (e.g. '$10,000' or '$6,000') when a percentage is also "
    "stated. Monthly Volume / Coverage amount rows that list "
    "'60% of monthly earnings' on one line and 'Maximum Amount $10,000' "
    "on another MUST be joined.\n"
    "- Do NOT confuse Guarantee Issue / Guaranteed Issue dollar amounts "
    "(e.g. 'We Guarantee Issue $6000 in coverage') with the Monthly/"
    "Weekly Benefit — GI is a separate field.\n"
    "- 'Elimination Period' (aka Waiting Period / When benefits begin): "
    "when Accident and Illness are listed SEPARATELY on the document, "
    "format as 'Accident: Day 91 / Illness: Day 91' (or Day 181, etc.). "
    "Include BOTH even when the day count is the same. If the document "
    "shows only a single elimination/waiting period (e.g. '90 Days') with "
    "NO Accident/Illness split, keep that single value — do NOT invent "
    "Accident/Illness labels.\n"
    "- 'Disability Definition': use the plan's definition of disability / "
    "own-occupation vs any-occupation wording — never Evidence of "
    "Insurability (EOI) / health-statement text."
)

# Block 7 — supplemental / term life coverage columns.
_LIFE_CATEGORIES: frozenset[str] = frozenset({"sup_life", "term_life"})

_LIFE_FIELD_DEFINITIONS_PROMPT = (
    "\n\nLIFE COVERAGE FIELDS:\n"
    "Voluntary / supplemental life summaries often use a 3-column chart "
    "(Employee Life Benefits | Spouse Life Benefits | Child Life "
    "Benefits). Map each column to the matching field; never copy a "
    "value from one column into another.\n"
    "- 'Child(ren) Life Insurance Coverage': use the Child Life Benefits "
    "Benefit Amount / purchase options (e.g. '$2,000, $4,000, or "
    "$10,000 for children age 14 days or older; $1,000 for children "
    "under 14 days'). Do NOT use 'Not Applicable' from the Child "
    "Minimum/Maximum/Age Reductions/Proof of Good Health rows when "
    "actual child benefit amounts are listed in Benefit Amount.\n"
    "- 'Employee Life Insurance Coverage' / 'Spouse Life Insurance "
    "Coverage': include the purchase increments and/or min–max when "
    "stated (e.g. 'increments of $10,000; $10,000–$300,000').\n"
    "- 'Age Reduction Schedule': employee/spouse age-reduction wording; "
    "ignore a Child-column 'Not Applicable' for this field."
)


def _build_system_prompt(category: str, display_category: str) -> str:
    """Compose a category-appropriate system prompt from the relevant blocks."""
    parts = [_BASE_PROMPT.format(display_category=display_category)]
    if category in _NETWORK_TABLE_CATEGORIES:
        parts.append(_SINGLE_COLUMN_PROMPT)
    if category in _HEALTH_CATEGORIES:
        parts.append(_HEALTH_FIELD_DEFINITIONS_PROMPT)
    if category == "health_3tier":
        parts.append(_HEALTH_3TIER_NETWORK_PROMPT)
    if category == "dental":
        parts.append(_DENTAL_FIELD_DEFINITIONS_PROMPT)
    if category == "vision":
        parts.append(_VISION_FIELD_DEFINITIONS_PROMPT)
    if category in _DISABILITY_CATEGORIES:
        parts.append(_DISABILITY_FIELD_DEFINITIONS_PROMPT)
    if category in _LIFE_CATEGORIES:
        parts.append(_LIFE_FIELD_DEFINITIONS_PROMPT)
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
        # The OpenAI Python client merges `extra_body` into the request JSON.
        # We POST with httpx, so those keys (e.g. guided_json) must be top-level.
        # Nesting them under an "extra_body" field is ignored by vLLM, which
        # silently disables guided decoding and makes outputs non-deterministic.
        payload.update(extra_body)

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
