"""Post-extraction field normalization and computation.

Two passes happen after VLM extraction:

Pass 1 — Normalization (RX field cleanup):
  1. "Not covered" whole-field values → empty string.
  2. Out-of-Network RX fields → empty for HMO/EPO plans.

Pass 2 — Computed fields (derived, never sent to VLM):
  The per-tier RX fields (In-Network Generic RX, In-Network Brand RX, etc.)
  are computed by splitting the consolidated In-Network RX / Out-of-Network RX
  strings into positional tiers.

  Tier mapping (positional, 0-indexed within the consolidated string):
    0 → Generic RX
    1 → Brand RX
    2 → Tier 3 RX
    3 → Tier 4 RX
    4 → Tier 5 RX

  The label prefix ("Generic: ", "Tier 1 (Generic): ", etc.) is stripped;
  only the cost portion is kept in each per-tier field.
"""
from __future__ import annotations
import re

# Matches a semicolon used as a TIER separator: "; AlphaLabel: "
# Does NOT match semicolons inside values like "; 90-day supply: " (digit after space)
# or "; maintenance drugs only): " (no colon-space after the label word).
_TIER_SEMICOLON = re.compile(r';\s+(?=[A-Za-z][^;:/]+:\s)')

_COMPUTED_TIER_FIELDS: frozenset[str] = frozenset({
    "In-Network Generic RX", "Out-of-Network Generic RX",
    "In-Network Brand RX", "Out-of-Network Brand RX",
    "In-Network Tier 3 RX", "Out-of-Network Tier 3 RX",
    "In-Network Tier 4 RX", "Out-of-Network Tier 4 RX",
    "In-Network Tier 5 RX", "Out-of-Network Tier 5 RX",
})

COMPUTED_FIELD_NAMES: frozenset[str] = _COMPUTED_TIER_FIELDS

_NOT_COVERED_PHRASES: frozenset[str] = frozenset({
    "not covered",
    "not covered.",
    "not applicable",
    "not available",
    "n/a",
    "null",
})

_OON_RX_FIELDS: frozenset[str] = frozenset({
    "Out-of-Network RX",
    "Out-of-Network Mail Order RX",
    "Out-of-Network RX Deductible",
})

_HMO_EPO_NETWORKS: frozenset[str] = frozenset({"hmo", "epo"})

# Positional mapping: index → per-tier field suffix
_TIER_SUFFIXES: tuple[str, ...] = (
    "Generic RX",
    "Brand RX",
    "Tier 3 RX",
    "Tier 4 RX",
    "Tier 5 RX",
)


def vlm_field_names(field_names: list[str]) -> list[str]:
    """Return field_names with computed fields removed — VLM never sees these."""
    return [f for f in field_names if f not in COMPUTED_FIELD_NAMES]


def _split_tiers(consolidated: str) -> list[str]:
    """Split a consolidated RX string into per-tier cost values.

    Two normalization steps before splitting:
    1. Semicolons used as tier separators (e.g. '; Preferred Brand Drugs: ')
       are replaced with ' / ' so both separator styles are handled uniformly.
       Semicolons inside values (e.g. '; 90-day supply: ', '; maintenance...')
       are left alone — they won't match because the word after the semicolon
       either starts with a digit or lacks a colon-space suffix.
    2. Only parts with an alphabetic label prefix are kept (e.g. 'Tier 1: $10',
       'Generic: $15').  Bare value fragments like '$20 (mail order)' that the
       VLM sometimes embeds after the retail cost are skipped.
    """
    if not consolidated:
        return []
    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    tiers = []
    for part in normalized.split(" / "):
        part = part.strip()
        if not part:
            continue
        if ": " in part:
            label = part.split(": ", 1)[0]
            if label and label[0].isalpha():
                tiers.append(part.split(": ", 1)[1])
    return tiers


def _tier_cost(consolidated: str, index: int) -> str:
    tiers = _split_tiers(consolidated)
    return tiers[index] if index < len(tiers) else ""


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],  # noqa: ARG001
) -> dict[str, str | None]:
    """Normalize RX fields and compute per-tier derived fields."""
    result = dict(fields)

    # Pass 1a: "Not covered" / "null" whole-field → empty string for any RX field.
    for key, val in result.items():
        if "RX" in key and isinstance(val, str):
            if val.strip().lower() in _NOT_COVERED_PHRASES:
                result[key] = ""

    # Pass 1b: HMO/EPO plans have no OON drug benefit — clear OON RX fields.
    network_type = (result.get("Network Type") or "").strip().lower()
    if any(net in network_type for net in _HMO_EPO_NETWORKS):
        for field in _OON_RX_FIELDS:
            if field in result:
                result[field] = ""

    # Pass 2: compute per-tier fields from consolidated strings.
    for net_prefix in ("In-Network", "Out-of-Network"):
        consolidated = result.get(f"{net_prefix} RX") or ""
        for i, suffix in enumerate(_TIER_SUFFIXES):
            field_name = f"{net_prefix} {suffix}"
            if field_name in output_field_names:
                result[field_name] = _tier_cost(consolidated, i)

    return result
