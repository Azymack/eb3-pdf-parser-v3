"""Post-extraction field normalization and computation.

Three passes happen after VLM extraction:

Pass 1 — Normalization (RX field cleanup):
  1. "Not covered" whole-field values → empty string.
  2. Out-of-Network RX fields → empty for HMO/EPO plans.

Pass 2 — Computed fields (derived, never sent to VLM):
  The per-tier RX fields are computed from the consolidated In-Network RX /
  Out-of-Network RX strings using LABEL-BASED mapping (not positional).

Pass 3 — Mail Order RX label stripping:
  Mail Order RX fields display 90-day costs as plain values only ("$20 / $80"),
  not as "Label: cost" pairs.  Any labels the VLM included are stripped here.

  Tier assignment rules (in priority order):
    Non-Preferred Brand  → Tier 3 RX  (ALWAYS, even if doc numbers it Tier 2)
    Non-Preferred Generic → Brand RX   (Tier 2 in split-generic plans)
    Non-Preferred Specialty → Tier 5 RX
    Specialty / Preferred Specialty → Tier 4 RX
    Generic / Preferred Generic → Generic RX (Tier 1)
    Brand / Preferred Brand → Brand RX (Tier 2)
    Bare "Tier N" labels without a descriptor → positional by number

  This means a document that skips Preferred Brand entirely and only has
  Generic + Non-Preferred Brand + Specialty produces:
    Generic RX=$X, Brand RX="", Tier 3 RX=$Y, Tier 4 RX=$Z
  rather than the wrong positional: Generic=$X, Brand=$Y, Tier 3=$Z.
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

# Mail Order RX fields — display as cost-values only ("$20 / $80"), no tier labels.
_MAIL_ORDER_RX_FIELDS: frozenset[str] = frozenset({
    "In-Network Mail Order RX",
    "Out-of-Network Mail Order RX",
    "Designated Network Mail Order RX",
})

# Positional mapping: 0-based index → per-tier field suffix
_TIER_SUFFIXES: tuple[str, ...] = (
    "Generic RX",    # index 0 — Tier 1
    "Brand RX",      # index 1 — Tier 2
    "Tier 3 RX",     # index 2
    "Tier 4 RX",     # index 3
    "Tier 5 RX",     # index 4
)


def vlm_field_names(field_names: list[str]) -> list[str]:
    """Return field_names with computed fields removed — VLM never sees these."""
    return [f for f in field_names if f not in COMPUTED_FIELD_NAMES]


# ── Label-based tier mapping ──────────────────────────────────────────────────

def _label_to_tier_index(label: str) -> int | None:
    """Map a drug tier label string to a 0-based tier index.

    Returns:
        0  Generic RX   (Tier 1)
        1  Brand RX     (Tier 2)
        2  Tier 3 RX    (Non-Preferred Brand — ALWAYS here regardless of doc numbering)
        3  Tier 4 RX    (Specialty / Preferred Specialty)
        4  Tier 5 RX    (Non-Preferred Specialty)
        None  — label unrecognized; caller decides what to do

    Priority order matters: Non-Preferred Brand is checked before Brand so that
    a label like "Non-Preferred Brand Drugs" doesn't accidentally match the
    generic 'brand' check.
    """
    # Normalize: lowercase, collapse whitespace
    n = re.sub(r'\s+', ' ', label.lower().strip())
    # Strip "drugs" suffix ("Generic Drugs" → "Generic")
    n = re.sub(r'\s+drugs?$', '', n).strip()
    # Strip bare parenthetical tier numbers "(tier N)" — they carry no semantic info
    # beyond what the label text already says, e.g. "Tier 1 (Generic)" → "tier 1 generic"
    # Leave "(preferred)" / "(non-preferred)" etc. since they carry meaning.
    n = re.sub(r'\s*\(tier\s*\d+\)\s*', ' ', n).strip()
    n = re.sub(r'\s{2,}', ' ', n)

    # ── Priority 1: Non-Preferred Brand → ALWAYS Tier 3 ──────────────────────
    # Catches: "Non-Preferred Brand", "Non-Preferred Brand Drugs",
    #          "Tier 2 (Non-Preferred Brand)", "Non Preferred Brand"
    if re.search(r'non.preferred\s+brand', n):
        return 2

    # ── Priority 2: Non-Preferred Generic → Tier 2 ───────────────────────────
    # Occurs in plans that split generics into Preferred/Non-Preferred.
    # Must be checked before the generic catch-all below.
    if re.search(r'non.preferred\s+generic', n):
        return 1

    # ── Priority 3: Non-Preferred Specialty → Tier 5 ─────────────────────────
    # Catches: "Non-Preferred Specialty", "Specialty (Non-Preferred)"
    if re.search(r'non.preferred\s+specialty', n) or (
        'specialty' in n and re.search(r'non.preferred', n)
    ):
        return 4

    # ── Tier 5 by number ─────────────────────────────────────────────────────
    if re.fullmatch(r'tier\s*5', n):
        return 4

    # ── Specialty (anything remaining with "specialty") → Tier 4 ─────────────
    # Catches: "Specialty", "Preferred Specialty", "Specialty Drugs",
    #          "Typically Preferred Specialty"
    if 'specialty' in n:
        return 3

    # ── Tier 4 by number or alias ─────────────────────────────────────────────
    if re.fullmatch(r'tier\s*4', n) or n in (
        'tier 4 - typically preferred specialty',
        'tier 4 – typically preferred specialty',
        'typically preferred specialty',
        'preferred specialty',
    ):
        return 3

    # ── Generic → Tier 1 ─────────────────────────────────────────────────────
    # Catches: "Generic", "Preferred Generic", "Generic (Tier 1)"
    # Must come AFTER non-preferred generic check above.
    if 'generic' in n:
        return 0

    # ── Tier 1 by number or alias ─────────────────────────────────────────────
    if re.fullmatch(r'tier\s*1[a-z]?', n) or n in (
        'tier 1 - typically generic',
        'tier 1 – typically generic',
    ):
        return 0

    # ── Brand → Tier 2 ───────────────────────────────────────────────────────
    # Catches: "Brand", "Brand Name", "Preferred Brand", "Brand Name Drugs"
    # Must come AFTER non-preferred brand check above.
    if 'brand' in n:
        return 1

    # ── Tier 2 by number or alias ─────────────────────────────────────────────
    if re.fullmatch(r'tier\s*2', n) or n in (
        'tier 2 - typically preferred brand',
        'tier 2 – typically preferred brand',
    ):
        return 1

    # ── Tier 3 by number or alias ─────────────────────────────────────────────
    if re.fullmatch(r'tier\s*3', n) or n in (
        'tier 3 - typically non-preferred brand',
        'tier 3 – typically non-preferred brand',
    ):
        return 2

    return None  # unrecognized


def _strip_tier_labels(value: str) -> str:
    """Strip tier labels from a mail-order RX value string.

    Converts 'Tier 1: $20 / Tier 2: $80 / Tier 3: $130' → '$20 / $80 / $130'.
    Values already in cost-only format are returned unchanged.
    Empty strings are passed through as-is.

    Semicolons used as tier separators are normalised to ' / ' first.
    """
    if not value:
        return value
    normalized = _TIER_SEMICOLON.sub(" / ", value)
    parts = []
    for part in normalized.split(" / "):
        part = part.strip()
        if not part:
            continue
        if ": " in part:
            _, _, cost = part.partition(": ")
            parts.append(cost.strip())
        else:
            parts.append(part)
    return " / ".join(parts)


def _extract_tier_values(consolidated: str) -> dict[int, str]:
    """Parse a consolidated RX string into {tier_index: cost_value}.

    Uses label-based mapping via _label_to_tier_index.  Entries whose labels
    cannot be mapped are skipped rather than placed positionally — assigning an
    unknown label to the wrong tier is worse than leaving the slot empty.

    First labeled match per slot wins (guards against duplicate tier entries
    the VLM occasionally emits).
    """
    if not consolidated:
        return {}

    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    result: dict[int, str] = {}

    for part in normalized.split(" / "):
        part = part.strip()
        if not part or ": " not in part:
            continue
        label, _, value = part.partition(": ")
        label = label.strip()
        value = value.strip()
        if not label or not label[0].isalpha():
            continue

        tier_idx = _label_to_tier_index(label)
        if tier_idx is not None and tier_idx not in result:
            result[tier_idx] = value

    return result


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],
) -> dict[str, str | None]:
    """Normalize RX fields and compute per-tier derived fields."""
    result = dict(fields)

    # Pass 1a: "Not covered" / "null" whole-field -> empty string for any RX field.
    for key, val in result.items():
        if "RX" in key and isinstance(val, str):
            if val.strip().lower() in _NOT_COVERED_PHRASES:
                result[key] = ""

    # Pass 1b: HMO/EPO plans have no OON drug benefit -- clear OON RX fields.
    network_type = (result.get("Network Type") or "").strip().lower()
    if any(net in network_type for net in _HMO_EPO_NETWORKS):
        for field in _OON_RX_FIELDS:
            if field in result:
                result[field] = ""

    # Pass 2: compute per-tier fields using label-based mapping.
    for net_prefix in ("In-Network", "Out-of-Network"):
        consolidated = result.get(f"{net_prefix} RX") or ""
        tier_values = _extract_tier_values(consolidated)
        for i, suffix in enumerate(_TIER_SUFFIXES):
            field_name = f"{net_prefix} {suffix}"
            if field_name in output_field_names:
                result[field_name] = tier_values.get(i, "")

    # Pass 3: strip tier labels from Mail Order RX fields.
    # Mail Order fields display 90-day costs as "$20 / $80", not "Tier 1: $20 / Tier 2: $80".
    for field in _MAIL_ORDER_RX_FIELDS:
        val = result.get(field)
        if isinstance(val, str) and val:
            result[field] = _strip_tier_labels(val)

    return result
