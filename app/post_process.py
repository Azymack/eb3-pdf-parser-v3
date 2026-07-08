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
    # Strip "all other" prefix ("All Other Generic" → "Generic") — sub-row qualifier
    n = re.sub(r'^all\s+other\s+', '', n).strip()
    # Strip bare parenthetical tier numbers "(tier N)" — they carry no semantic info
    # beyond what the label text already says, e.g. "Tier 1 (Generic)" → "tier 1 generic"
    # Leave "(preferred)" / "(non-preferred)" etc. since they carry meaning.
    n = re.sub(r'\s*\(tier\s*\d+\)\s*', ' ', n).strip()
    n = re.sub(r'\s{2,}', ' ', n)

    # ── Exclude preventive sub-tier labels → skip ────────────────────────────
    # "Preventive" drugs are a special $0 sub-category within a tier (e.g.,
    # "Generic (Preventive): No Charge" alongside "Generic: $20"). The standard
    # tier cost is what we want; skipping preventive labels lets the next entry win.
    if re.search(r'\bpreventive\b', n):
        return None

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

    # ── Fallback: leading tier number with an unrecognized descriptor ─────────
    # Catches labels like "Tier 3 (Non-preferred)" where the descriptor alone
    # matched nothing above. The document's own tier number decides the slot.
    m = re.match(r'^tier\s*([1-5])\b', n)
    if m:
        return int(m.group(1)) - 1

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


def _split_sublabel(value: str) -> tuple[str | None, str]:
    """If value starts with an alpha sub-label ('SubLabel: cost'), return
    (sublabel, cost); otherwise (None, value).  Sub-labels containing '$' or
    '%' are treated as part of the cost, not a label."""
    if ": " in value and value and value[0].isalpha():
        sub, _, cost = value.partition(": ")
        if "$" not in sub and "%" not in sub:
            return sub.strip(), cost.strip()
    return None, value


def _is_preventive(label: str) -> bool:
    return bool(re.search(r'\bpreventive\b', label.lower()))


def _join_tier_value(current: str, addition: str) -> str:
    return addition if not current else current + " / " + addition


def _extract_tier_values(consolidated: str) -> dict[int, str]:
    """Parse a consolidated RX string into {tier_index: cost_value}.

    Uses label-based mapping via _label_to_tier_index.  Entries whose labels
    cannot be mapped are skipped rather than placed positionally.

    First labeled match per slot wins (guards against duplicate tier entries).

    Unlabeled parts that follow a successfully mapped labeled entry are appended
    to that entry's value with ' / '.  This handles three-column pharmacy tables
    where a tier's Preferred Network and In-Network retail costs appear in
    consecutive ' / '-separated segments with no second label:
      'Tier 2 (Brand): $80 / $90'  ->  tier index 1 value = '$80 / $90'

    SUB-ROW MODE: when a mapped tier's own value contains a nested sub-label
    ('Generic: Preventive: No Charge' or 'Preferred Brand: Condition Care Rx: $50'),
    the tier has program sub-rows.  While sub-row mode is active for a tier:
      - unrecognized labels ('Condition Care Rx: $4') append their COST to the tier
      - duplicate mappings to the same tier ('All Other Generic: $20') append too
      - 'Preventive' sub-rows are skipped (the $0 preventive-mandate sub-tier)
    Plain tiers (no nested sub-label) keep strict first-match-wins semantics and
    unrecognized labels reset continuation, as before.

    Example — Florida Blue:
      'Generic: Preventive: No Charge / Condition Care Rx: $4 / All Other Generic: $20'
      → {0: '$4 / $20'}
    """
    if not consolidated:
        return {}

    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    result: dict[int, str] = {}
    last_mapped_idx: int | None = None  # most recent successfully mapped tier index
    subrow_mode = False                 # last mapped tier has program sub-rows

    for part in normalized.split(" / "):
        part = part.strip()
        if not part:
            continue

        if ": " in part and part[0].isalpha():
            label, _, value = part.partition(": ")
            label = label.strip()
            value = value.strip()
            if not label:
                continue
            tier_idx = _label_to_tier_index(label)
            if tier_idx is not None:
                sub, cost = _split_sublabel(value)
                nested = sub is not None
                if tier_idx not in result:
                    if nested and _is_preventive(sub):
                        result[tier_idx] = ""   # anchored; awaiting non-preventive sub-rows
                    else:
                        result[tier_idx] = cost if nested else value
                    last_mapped_idx = tier_idx
                    subrow_mode = nested
                elif subrow_mode and last_mapped_idx == tier_idx:
                    # Sub-row re-using the tier label (e.g. 'All Other Generic: $20')
                    if not (nested and _is_preventive(sub)):
                        result[tier_idx] = _join_tier_value(result[tier_idx], cost)
                else:
                    # Duplicate label outside sub-row mode — first match wins
                    last_mapped_idx = None
                    subrow_mode = False
            else:
                # Unrecognized label
                if subrow_mode and last_mapped_idx is not None:
                    # Program sub-row (e.g. 'Condition Care Rx: $4') — append cost
                    if not _is_preventive(label):
                        result[last_mapped_idx] = _join_tier_value(result[last_mapped_idx], value)
                else:
                    last_mapped_idx = None
        else:
            # Unlabeled part — append to the most recent mapped tier if any
            if last_mapped_idx is not None:
                result[last_mapped_idx] = _join_tier_value(result[last_mapped_idx], part)

    # Drop tiers that ended up empty (e.g. only a Preventive sub-row was present)
    return {idx: v for idx, v in result.items() if v}


def _extract_retail_only(value: str) -> str:
    """Strip home-delivery cost from a Preferred Network tier value.

    Preferred Network column cells often encode both retail and home delivery
    costs in a single string, e.g.:
      "$20 copay (retail) and $40 copay (home delivery)" → "$20 copay"
      "30% coinsurance (retail and home delivery)"       → "30% coinsurance"
      "$20 copay"  (no qualifier)                        → "$20 copay"
    """
    # Pattern 1: "X (retail) and Y (home delivery)" — keep X only
    m = re.match(r'^(.*?)\s*\(retail\)\s+and\b.*', value, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Pattern 2: "X (retail and home delivery)" — strip the qualifier
    cleaned = re.sub(r'\s*\(retail and home delivery\)', '', value, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_home_delivery(value: str) -> str:
    """Extract the home delivery (mail order) cost from a Preferred Network tier value.

      "$20 copay (retail) and $40 copay (home delivery)" → "$40 copay"
      "30% coinsurance (retail and home delivery)"       → "30% coinsurance" (same rate)
      "$20 copay"  (no qualifier)                        → "" (no mail order info)
    """
    # Pattern 1: "X (retail) and Y (home delivery)" — keep Y
    m = re.search(r'\(retail\)\s+and\s+(.*?)\s*\(home delivery\)', value, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Pattern 2: "X (retail and home delivery)" — same cost applies to mail order
    m = re.match(r'^(.*?)\s*\(retail and home delivery\)', value, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _strip_rx_suffix(value: str) -> str:
    """Strip per-prescription descriptor suffixes from RX cost values.

    Operates per-part on slash-separated strings so each cost is cleaned
    independently before rejoining.

    Examples:
      "$20 copay per prescription, deductible does not apply" → "$20"
      "$80 copay per prescription after deductible is met"   → "$80"
      "30% coinsurance up to $400 per prescription..."       → "30% coinsurance up to $400"
      "$80 copay... / $90 copay..."                          → "$80 / $90"
    """
    if not value:
        return value
    parts = [p.strip() for p in value.split(" / ")]
    stripped: list[str] = []
    for p in parts:
        # "$X copay per prescription[,  ...]" → "$X"
        s = re.sub(r'\s*\bcopay\b\s+per\s+prescription\b.*$', '', p, flags=re.IGNORECASE).strip()
        # "30% coinsurance up to $X per prescription[...]" → "30% coinsurance up to $X"
        s = re.sub(r'\s*\bper\s+prescription\b.*$', '', s, flags=re.IGNORECASE).strip()
        # "Deductible + $300 Copay" → "Deductible + $300"  (bare trailing "copay")
        s = re.sub(r'\s+\bcopay\b$', '', s, flags=re.IGNORECASE).strip()
        stripped.append(s or p)   # never return an empty part
    return " / ".join(stripped)


def _merge_preferred_and_inn_tier_values(
    preferred_rx: str, inn_rx: str
) -> dict[int, str]:
    """Merge Preferred Network and In-Network retail RX costs per tier.

    For plans with three pharmacy columns (Preferred Network | In-Network | OON),
    the VLM extracts each column separately.  This function combines them:
      - same cost in both columns → single value
      - different costs           → "pref / inn"
      - only one column has a value → use that value

    Preferred Network cells may mix retail + home delivery costs. After isolating
    the retail portion, _strip_rx_suffix removes "copay per prescription..."
    verbosity so that semantically identical values compare as equal regardless of
    whether the VLM included the full descriptor.
    """
    pref_raw = _extract_tier_values(preferred_rx)
    inn_raw  = _extract_tier_values(inn_rx)
    # Retail-only + strip verbosity for clean comparison and output
    pref = {idx: _strip_rx_suffix(_extract_retail_only(v)) for idx, v in pref_raw.items()}
    inn  = {idx: _strip_rx_suffix(v)                       for idx, v in inn_raw.items()}
    result: dict[int, str] = {}
    for idx in sorted(set(pref) | set(inn)):
        p = pref.get(idx, "")
        i = inn.get(idx, "")
        if p and i:
            result[idx] = p if p == i else f"{p} / {i}"
        else:
            result[idx] = p or i
    return result


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],
) -> dict[str, str | None]:
    """Normalize RX fields and compute per-tier derived fields."""
    result = dict(fields)

    # Capture OON "Not covered" status BEFORE Pass 1a clears it.
    # This lets Pass 2 propagate "Not covered" to per-tier OON fields.
    oon_rx_not_covered = (
        (result.get("Out-of-Network RX") or "").strip().lower() in _NOT_COVERED_PHRASES
    )

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
        # Also prevent Pass 2 from propagating "Not covered" to per-tier OON fields.
        # HMO/EPO per-tier OON fields should be "" (not applicable), not "Not covered".
        oon_rx_not_covered = False

    # Pass 2: compute per-tier fields using label-based mapping.
    # If Preferred Network RX is present (3-column plan), merge with In-Network RX.
    preferred_rx = result.get("Preferred Network RX") or ""
    for net_prefix in ("In-Network", "Out-of-Network"):
        # Propagate "Not covered" to all per-tier OON fields when applicable.
        if net_prefix == "Out-of-Network" and oon_rx_not_covered:
            for suffix in _TIER_SUFFIXES:
                field_name = f"Out-of-Network {suffix}"
                if field_name in output_field_names:
                    result[field_name] = "Not covered"
            continue

        consolidated = result.get(f"{net_prefix} RX") or ""
        if net_prefix == "In-Network" and preferred_rx:
            tier_values = _merge_preferred_and_inn_tier_values(preferred_rx, consolidated)
        else:
            tier_values = _extract_tier_values(consolidated)
        for i, suffix in enumerate(_TIER_SUFFIXES):
            field_name = f"{net_prefix} {suffix}"
            if field_name in output_field_names:
                v = tier_values.get(i, "")
                # Strip "copay per prescription..." verbosity from 2-column plan values
                # (3-column values are already stripped inside _merge_preferred_and_inn_tier_values)
                result[field_name] = _strip_rx_suffix(v) if v else v

    # Pass 3: strip tier labels from Mail Order RX fields.
    # Mail Order fields display 90-day costs as "$20 / $80", not "Tier 1: $20 / Tier 2: $80".
    for field in _MAIL_ORDER_RX_FIELDS:
        val = result.get(field)
        if isinstance(val, str) and val:
            result[field] = _strip_tier_labels(val)

    # Pass 4: For 3-column plans, derive Mail Order deterministically from the
    # home-delivery costs embedded in the Preferred Network column.  This overrides
    # whatever Pass 3 produced for In-Network Mail Order RX and is more reliable
    # than VLM extraction (which sometimes picks the wrong column for Tier 4).
    if preferred_rx:
        pref_raw = _extract_tier_values(preferred_rx)
        hd_parts: list[str] = []
        for idx in sorted(pref_raw):
            hd = _extract_home_delivery(pref_raw[idx])
            if hd:
                hd_parts.append(_strip_rx_suffix(hd))
        if hd_parts and "In-Network Mail Order RX" in output_field_names:
            result["In-Network Mail Order RX"] = " / ".join(hd_parts)
        # Propagate "Not covered" to OON Mail Order when the OON column is not covered.
        if oon_rx_not_covered and "Out-of-Network Mail Order RX" in output_field_names:
            result["Out-of-Network Mail Order RX"] = "Not covered"

    return result
