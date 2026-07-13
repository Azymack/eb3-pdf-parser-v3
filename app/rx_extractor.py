"""Structured RX (prescription drug) extraction.

Replaces the consolidated-string + regex-parsing pipeline for pharmacy fields.

Why
---
Carrier tier vocabulary is open-ended ("Level 1", "Value generic drugs",
"Preferred Drugs", "Tier 1a") and per-cell channel formats vary endlessly
("$10 (retail) and $25 (home delivery)", "30-day supply: $25; 90-day supply:
$62.50", "Retail/Mail Order (1-30 days) $10. Mail Order (31-90 days) $20.").
Regex label-mapping and string-splitting in post-processing can never keep up —
every new carrier needed a new rule, and new rules broke old documents.

Architecture
------------
The VLM does the SEMANTIC work in one dedicated, RX-only call:
  - transcribe the pharmacy table row by row (one entry per printed tier row)
  - classify each row into a standard tier slot (generic / preferred_brand /
    non_preferred_brand / preferred_specialty / non_preferred_specialty)
  - split each cell into retail vs mail-order costs, per network column

Python then does only DETERMINISTIC assembly:
  - join row costs into the fixed output fields per network
  - build the combined Mail Order RX values (cost-only, tier order)
  - propagate "Not covered" for out-of-network pharmacies

Standard tier slots (the product's fixed field layout):
  generic                → "<net> Generic RX"   (Tier 1)
  preferred_brand        → "<net> Brand RX"     (Tier 2)
  non_preferred_brand    → "<net> Tier 3 RX"
  preferred_specialty    → "<net> Tier 4 RX"
  non_preferred_specialty→ "<net> Tier 5 RX"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .post_process import _strip_rx_suffix
from .vlm_client import _chat_completion

logger = logging.getLogger(__name__)

# Categories that use the structured RX extractor.
RX_EXTRACTOR_CATEGORIES: frozenset[str] = frozenset({"health", "health_3tier"})

_STANDARD_TIERS: tuple[str, ...] = (
    "generic",
    "preferred_brand",
    "non_preferred_brand",
    "preferred_specialty",
    "non_preferred_specialty",
)

_TIER_SUFFIX: dict[str, str] = {
    "generic": "Generic RX",
    "preferred_brand": "Brand RX",
    "non_preferred_brand": "Tier 3 RX",
    "preferred_specialty": "Tier 4 RX",
    "non_preferred_specialty": "Tier 5 RX",
}

# (display prefix, schema key prefix) per category, in output order.
_NETWORKS: dict[str, list[tuple[str, str]]] = {
    "health": [
        ("In-Network", "in_network"),
        ("Out-of-Network", "out_of_network"),
    ],
    "health_3tier": [
        ("Designated Network", "designated_network"),
        ("In-Network", "in_network"),
        ("Out-of-Network", "out_of_network"),
    ],
}

_NOISE_VALUES: frozenset[str] = frozenset({
    "null", "none", "n/a", "na", "not applicable", "not available",
    "not provided", "not shown", "-", "--",
})

_NO_DEDUCTIBLE_VALUES: frozenset[str] = frozenset({
    "$0", "0", "none", "no deductible", "does not apply", "not applicable",
    "n/a", "null", "no", "deductible does not apply",
})


def rx_owned_fields(category: str) -> list[str]:
    """All output fields produced by this module for the category."""
    fields: list[str] = []
    for display, _key in _NETWORKS[category]:
        fields.append(f"{display} RX Deductible")
        fields.append(f"{display} RX")
        fields.extend(f"{display} {suffix}" for suffix in _TIER_SUFFIX.values())
        fields.append(f"{display} Mail Order RX")
    fields.append("Preferred Network RX")
    return fields


# ---------------------------------------------------------------------------
# VLM schema + prompt
# ---------------------------------------------------------------------------

def _nullable_string() -> dict:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def _build_rx_schema(category: str) -> dict:
    net_keys = [key for _, key in _NETWORKS[category]]
    row_props: dict[str, Any] = {
        "label": {"type": "string"},
        "standard_tier": {
            "type": "string",
            "enum": list(_STANDARD_TIERS) + ["other"],
        },
    }
    for key in net_keys:
        row_props[f"{key}_retail"] = _nullable_string()
        row_props[f"{key}_mail_order"] = _nullable_string()
    if category == "health":
        row_props["preferred_pharmacy_retail"] = _nullable_string()
        row_props["preferred_pharmacy_mail_order"] = _nullable_string()

    top_props: dict[str, Any] = {
        "drug_rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": row_props,
                "required": list(row_props.keys()),
                "additionalProperties": False,
            },
        },
        "mail_order_service": {"type": "boolean"},
        "out_of_network_pharmacy": {
            "type": "string",
            "enum": [
                "covered",
                "not_covered",
                "emergency_or_reimbursement_only",
                "not_shown",
            ],
        },
    }
    for key in net_keys:
        top_props[f"rx_deductible_{key}"] = _nullable_string()

    return {
        "type": "object",
        "properties": top_props,
        "required": list(top_props.keys()),
        "additionalProperties": False,
    }


_RX_SYSTEM_PROMPT_HEALTH_NETWORKS = (
    "NETWORK COLUMNS:\n"
    "- in_network_*: the standard network pharmacy column ('Network Provider', "
    "'In-Network Pharmacy', 'Participating').\n"
    "- preferred_pharmacy_*: ONLY when the document has a SEPARATE preferred-level "
    "pharmacy column in addition to the standard one (e.g. 'Preferred Network Pharmacy' "
    "next to 'In-Network Pharmacy', or in-cell dual pricing like "
    "'Preferred - $10 / Participating - $20' where the Preferred amount goes to "
    "preferred_pharmacy_retail and the Participating amount goes to in_network_retail). "
    "null when no separate preferred pharmacy level exists.\n"
    "- out_of_network_*: the out-of-network / non-participating pharmacy column.\n"
)

_RX_SYSTEM_PROMPT_3TIER_NETWORKS = (
    "NETWORK COLUMNS:\n"
    "- designated_network_*: the designated / first-tier network column.\n"
    "- in_network_*: the standard in-network column.\n"
    "- out_of_network_*: the out-of-network / non-participating column.\n"
)


def _build_rx_system_prompt(category: str) -> str:
    networks_block = (
        _RX_SYSTEM_PROMPT_3TIER_NETWORKS
        if category == "health_3tier"
        else _RX_SYSTEM_PROMPT_HEALTH_NETWORKS
    )
    return (
        "You are an insurance document extraction specialist. Extract the "
        "PRESCRIPTION DRUG (pharmacy) benefit structure from this health insurance "
        "plan document. Respond with ONLY valid JSON matching the required schema.\n\n"
        "DRUG ROWS: output one drug_rows entry for EVERY printed drug tier row of the "
        "pharmacy cost-share section, in document order. Pharmacy sections often "
        "continue across pages — scan ALL provided pages and include every row.\n\n"
        "label: the row's tier label exactly as printed. Examples: 'Generic drugs', "
        "'Tier 2', 'Level 1: Preferred generic drugs and certain lower cost preferred "
        "brand name drugs', 'Value generic drugs (Tier 1)', 'Preferred Drugs', "
        "'Specialty (Non-Preferred)'.\n\n"
        "standard_tier: classify each row by the MEANING of its label/description — "
        "what kinds of drugs it covers — not by its printed number:\n"
        "- generic: generic drug rows, including value/low-cost generic, preferred "
        "generic, non-preferred generic, and rows described as 'preferred generic "
        "drugs and certain lower cost preferred brand name drugs'.\n"
        "- preferred_brand: preferred/formulary brand drugs. A row named just 'Brand' "
        "or 'Brand-name drugs' is preferred_brand when there is no separate "
        "non-preferred brand row; if a separate 'Preferred Brand' row exists, a plain "
        "'Brand' row means non_preferred_brand.\n"
        "- non_preferred_brand: non-preferred / non-formulary brand drugs "
        "('Non-preferred brand', 'Non-formulary', 'Preferred Drugs'-then-'Non "
        "Preferred Drugs' pairs: the Non Preferred row).\n"
        "- preferred_specialty: specialty drugs (a lone 'Specialty' row, or the "
        "preferred specialty row when split).\n"
        "- non_preferred_specialty: non-preferred specialty drugs.\n"
        "- other: preventive/$0-mandate sub-rows, vaccine rows, or rows that are not "
        "drug cost tiers.\n"
        "Multiple rows MAY map to the same standard_tier (e.g. preferred + "
        "non-preferred generic both map to generic; 'Value generic drugs (Tier 1)' and "
        "'Generic drugs (Tier 2)' both map to generic). Never skip a printed row.\n"
        "AmeriHealth-style naming: 'Preferred Drugs' (brand formulary) → "
        "preferred_brand; 'Non Preferred Drugs' → non_preferred_brand.\n"
        "For combined multi-class rows, the printed tier number decides: Tier 1 → "
        "generic, Tier 2 → preferred_brand, Tier 3 → non_preferred_brand, Tier 4 → "
        "preferred_specialty, Tier 5 → non_preferred_specialty. Example: 'Typically "
        "Non-Preferred Brand and Generic drugs (Tier 3)' → non_preferred_brand.\n\n"
        "COST VALUES: copy the printed cost expression, keeping meaningful qualifiers "
        "('$10', '$3 copay', '50% coinsurance', '30% coinsurance up to $250', 'No "
        "charge after deductible', 'Deductible, then $50', '100% until deductible is "
        "met. After deductible $5/prescription', 'Not covered'). Do NOT include "
        "limitation narrative, day-supply wording, or channel wording in the value. "
        "Use null when the document shows no price for that network+channel. NEVER "
        "write the string 'null'.\n"
        "If a cell contains only narrative text about HOW or WHERE prescriptions may "
        "be filled (e.g. 'Prescriptions may be filled at an out-of-network pharmacy "
        "in emergency situations only ... submit a reimbursement form') and no cost "
        "expression, that is NOT a cost — use null for that cell.\n\n"
        "CHANNELS: retail = standard supply (usually 30-day) at a pharmacy; "
        "mail_order = mail order / home delivery / 90-day-supply pricing.\n"
        "ONE OUTPUT ROW PER PRINTED ROW: when a single printed row shows prices for "
        "two supply durations or channels, NEVER emit two drug_rows entries for it — "
        "emit ONE entry with the 30-day/retail price in retail and the longer-supply/"
        "mail price in mail_order. When one cell shows both, split it:\n"
        "- '$10/prescription (retail) and $25/prescription (home delivery)' → retail "
        "'$10', mail_order '$25'\n"
        "- '30-day supply: $25 copay; 90-day supply: $62.50 copay' → retail '$25', "
        "mail_order '$62.50' (when the plan offers mail order for that tier)\n"
        "- 'Retail/Mail Order (1-30 days supply) $10/Fill. Mail Order (31-90 days "
        "supply) $20/Fill.' → retail '$10/Fill', mail_order '$20/Fill'\n"
        "- '$4 copayment/prescription-Retail & mail order 30-day supply. $12 "
        "copayment/prescription-Retail 84-90-day supply & mail order 31-90-day "
        "supply.' → retail '$4', mail_order '$12'\n"
        "- 'No charge after deductible retail No charge after deductible mail order' "
        "→ retail 'No charge after deductible', mail_order 'No charge after "
        "deductible'\n"
        "- 'Retail: $10 / Mail-Order: $20' → retail '$10', mail_order '$20'\n"
        "- A row showing only a retail price → mail_order null. Do NOT copy a retail "
        "price into mail_order.\n"
        "- NEVER compute a mail-order price from a multiplier rule, and NEVER copy "
        "the retail price into mail_order because of one. Examples of multiplier "
        "rules that mean mail_order = null: '90-day supply at 2 times the retail "
        "amount', '90-day supply at 2.5x copay', '(2 copays apply to certain 90-day "
        "supply mail orders)'. Example: '$5/prescription to out-of-pocket limit. "
        "(2 copays apply to certain 90-day supply mail orders)' → retail '$5', "
        "mail_order null.\n"
        "- If the out-of-network cell says mail order / home delivery is 'Not "
        "covered', record out_of_network_mail_order as 'Not covered'.\n"
        "- Record out_of_network_mail_order ONLY when the document explicitly "
        "prints an out-of-network mail-order price. Most plans do not offer "
        "mail order through out-of-network pharmacies — never copy in-network "
        "mail-order prices there.\n\n"
        + networks_block +
        "If the pharmacy section has a single cost column that applies to both "
        "networks AND the plan clearly covers out-of-network pharmacy elsewhere, "
        "record the same cost for in_network and out_of_network. If a network column "
        "prints 'Not covered' for a row, record 'Not covered' for that row.\n\n"
        "TOP-LEVEL FIELDS:\n"
        "- rx_deductible_<network>: a deductible EXPLICITLY LABELED as applying to "
        "prescription drugs / pharmacy benefits (e.g. 'Prescription Drug deductible: "
        "$250/person or $500/family'). null when there is none or when it is $0. "
        "Do NOT copy the plan's overall medical deductible here, even when drug "
        "costs say 'after deductible' — that refers to the medical deductible and "
        "means rx_deductible is null.\n"
        "- mail_order_service: true when the plan offers a mail-order / home-delivery "
        "pharmacy service.\n"
        "- out_of_network_pharmacy: 'not_covered' when the document shows "
        "out-of-network prescriptions are not covered; "
        "'emergency_or_reimbursement_only' when OON fills are allowed only in "
        "emergencies or via reimbursement claims; 'covered' when OON prices are "
        "printed; 'not_shown' when the document does not address it."
    )


def _build_rx_messages(
    category: str,
    page_markdowns: list[dict],
    page_images: list[dict],
) -> list[dict]:
    markdown_sections = "\n\n---\n\n".join(
        f"## Page {p['page_number']}\n\n{p['markdown']}"
        for p in page_markdowns
    )
    user_text = (
        "Extract the prescription drug (pharmacy) benefit structure from this "
        "health insurance plan document.\n\n"
        f"**Document text (per page):**\n\n{markdown_sections}\n\n"
        "Return ONLY raw JSON matching the required schema."
    )
    content: list[dict] = [{"type": "text", "text": user_text}]
    for img in page_images:
        mime = img.get("mime_type", "image/jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{img['image_b64']}"},
        })
    return [
        {"role": "system", "content": _build_rx_system_prompt(category)},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Deterministic assembly
# ---------------------------------------------------------------------------

# A real cost expression contains at least one of these. Narrative cells
# ("Prescriptions may be filled at an out-of-network pharmacy in emergency
# situations only...") contain none and must be dropped, not displayed.
_COST_MARKER = re.compile(
    r"[\$%]|\d|no charge|not covered|covered in full|deductible|coinsurance|copay",
    re.IGNORECASE,
)


def _clean_cost(value: Any) -> str:
    """Normalize a single cost cell value from the VLM."""
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v or v.lower().strip(".") in _NOISE_VALUES:
        return ""
    if v.lower().strip(" .") in ("not covered", "not covered."):
        return "Not covered"
    if not _COST_MARKER.search(v):
        return ""
    return _strip_rx_suffix(v)


def _clean_deductible(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    v = value.strip()
    if not v or v.lower().strip(" .") in _NO_DEDUCTIBLE_VALUES | _NOISE_VALUES:
        return ""
    # A real RX deductible states an amount. Descriptions like "Subject to
    # combined medical and prescription drug deductible" mean there is no
    # separate RX deductible.
    if not _DOLLARS.search(v):
        return ""
    return v


def _dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def _is_implausible_mail_cost(value: str) -> bool:
    """Coinsurance above 100% is a computed multiplier (e.g. 45% x 2.5 = 112.5%),
    never a printed mail-order price."""
    return any(float(m.group(1)) > 100 for m in _PERCENT.finditer(value))


_DOLLARS = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def _dollar_amounts(value: str) -> set[str]:
    return {
        m.group(1).replace(",", "").removesuffix(".00").removesuffix(".0")
        for m in _DOLLARS.finditer(value or "")
    }


def suppress_medical_deductible_echo(
    rx_fields: dict[str, str],
    medical_fields: dict[str, Any],
) -> None:
    """Clear RX deductibles that merely repeat the plan's medical deductible.

    An integrated deductible ('after deductible' referring to the plan
    deductible) means there is NO separate RX deductible — the field must be
    empty. The VLM sometimes echoes the medical amounts anyway; the main
    extraction's deductible fields let us detect that deterministically.
    """
    # Only true deductible-amount fields — NOT "Deductible Explanation", which
    # often mentions the RX deductible amount itself and would wrongly
    # suppress it.
    med_amounts: set[str] = set()
    for key, val in medical_fields.items():
        if isinstance(val, str) and (
            key.endswith("Single Deductible") or key.endswith("Family Deductible")
        ):
            med_amounts.update(_dollar_amounts(val))
    if not med_amounts:
        return
    for field in rx_fields:
        if field.endswith("RX Deductible") and rx_fields[field]:
            amounts = _dollar_amounts(rx_fields[field])
            if amounts and amounts <= med_amounts:
                rx_fields[field] = ""


def assemble_rx_fields(category: str, data: dict) -> dict[str, str]:
    """Map the structured VLM RX result onto the flat output fields.

    Pure function — no I/O — so it is directly unit-testable.
    """
    fields: dict[str, str] = {f: "" for f in rx_owned_fields(category)}
    rows = [r for r in (data.get("drug_rows") or []) if isinstance(r, dict)]
    oon_status = (data.get("out_of_network_pharmacy") or "").strip()
    merge_preferred = category == "health"

    # Bucket rows by standard tier, preserving document order.
    slots: dict[str, list[dict]] = {t: [] for t in _STANDARD_TIERS}
    for row in rows:
        tier = row.get("standard_tier")
        if tier in slots:
            slots[tier].append(row)

    preferred_parts: list[str] = []

    for display, key in _NETWORKS[category]:
        fields[f"{display} RX Deductible"] = _clean_deductible(
            data.get(f"rx_deductible_{key}")
        )

        per_tier: dict[str, str] = {}
        consolidated: list[str] = []
        mail_costs: list[str] = []
        mail_not_covered = False

        for tier in _STANDARD_TIERS:
            retail_vals: list[str] = []
            for row in slots[tier]:
                retail = _clean_cost(row.get(f"{key}_retail"))
                mail = _clean_cost(row.get(f"{key}_mail_order"))
                if merge_preferred and key == "in_network":
                    pref_retail = _clean_cost(row.get("preferred_pharmacy_retail"))
                    if pref_retail:
                        if display == "In-Network" and tier in _TIER_SUFFIX:
                            label = (row.get("label") or "").strip()
                            preferred_parts.append(
                                f"{label}: {pref_retail}" if label else pref_retail
                            )
                        if retail and pref_retail != retail:
                            retail = f"{pref_retail} / {retail}"
                        elif not retail:
                            retail = pref_retail
                    if not mail:
                        mail = _clean_cost(row.get("preferred_pharmacy_mail_order"))
                if retail:
                    retail_vals.append(retail)
                    label = (row.get("label") or "").strip()
                    consolidated.append(f"{label}: {retail}" if label else retail)
                if mail:
                    if mail == "Not covered":
                        mail_not_covered = True
                    elif not _is_implausible_mail_cost(mail):
                        mail_costs.append(mail)
            if retail_vals:
                per_tier[tier] = " / ".join(_dedupe_keep_order(retail_vals))

        # Enum-based "not covered" must not override printed OON costs the model
        # extracted anyway (contradictory output) — costs win.
        has_real_costs = any(v and v != "Not covered" for v in per_tier.values())
        network_not_covered = key == "out_of_network" and (
            (oon_status == "not_covered" and not has_real_costs)
            or (bool(per_tier) and all(v == "Not covered" for v in per_tier.values()))
        )
        if network_not_covered:
            fields[f"{display} RX"] = "Not covered"
            fields[f"{display} Mail Order RX"] = "Not covered"
            for suffix in _TIER_SUFFIX.values():
                fields[f"{display} {suffix}"] = "Not covered"
            fields[f"{display} RX Deductible"] = ""
            continue

        for tier, suffix in _TIER_SUFFIX.items():
            fields[f"{display} {suffix}"] = per_tier.get(tier, "")
        fields[f"{display} RX"] = " / ".join(consolidated)
        if mail_costs:
            fields[f"{display} Mail Order RX"] = " / ".join(mail_costs)
        elif mail_not_covered:
            fields[f"{display} Mail Order RX"] = "Not covered"

    if preferred_parts:
        fields["Preferred Network RX"] = " / ".join(preferred_parts)

    return fields


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def extract_rx_fields(
    category: str,
    page_markdowns: list[dict],
    page_images: list[dict],
) -> dict[str, str]:
    """Run the structured RX extraction and return the flat RX output fields.

    Raises the same exceptions as the main VLM call (httpx errors, ValueError)
    — the caller decides how to degrade.
    """
    schema = _build_rx_schema(category)
    messages = _build_rx_messages(category, page_markdowns, page_images)

    logger.info("rx_extractor: starting structured RX extraction",
                extra={"category": category})
    raw = await _chat_completion(messages, extra_body={"guided_json": schema})
    cleaned = (
        raw.strip()
        .removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("rx_extractor: JSON parse failed",
                     extra={"content_preview": cleaned[:300]})
        raise ValueError(f"RX extraction returned unparseable content: {exc}") from exc

    fields = assemble_rx_fields(category, data)
    logger.info(
        "rx_extractor: extraction complete",
        extra={
            "rows": len(data.get("drug_rows") or []),
            "oon_status": data.get("out_of_network_pharmacy"),
            "populated": sum(1 for v in fields.values() if v),
        },
    )
    return fields
