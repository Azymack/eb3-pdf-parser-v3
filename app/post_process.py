"""Post-extraction field normalization and computation.

Three passes happen after VLM extraction:

Pass 1 — Normalization (RX field cleanup):
  1. "Not covered" whole-field values → empty string.
  2. Out-of-Network RX fields → empty for HMO/EPO plans.

Pass 2 — Computed fields (derived, never sent to VLM):
  Per-tier RX fields are computed from consolidated RX strings using label-based
  tier mapping.  Each tier cell is channel-split (_extract_retail_only /
  _extract_mail_from_tier_value) so retail and mail-order costs in the same
  cell are routed to the correct output fields.

  health (2-col / Preferred+INN): In-Network per-tier merges Preferred + INN
  columns when costs differ ("pref / inn").

  health_3tier (Designated+INN+OON): each network column feeds its own per-tier
  fields — no cross-column merge.

Pass 2b — Mail-order retail dedup (INN/OON only):
  Drop retail-only coinsurance the VLM wrongly copied into mail-order fields.

Pass 3 — Mail Order RX label stripping:
  Mail Order RX fields display 90-day costs as plain values only ("$20 / $80"),
  not as "Label: cost" pairs.  Any labels the VLM included are stripped here.

  Tier assignment rules (in priority order):
    Non-Preferred Brand  → Tier 3 RX  (ALWAYS, even if doc numbers it Tier 2)
    Brand (non-preferred) suffix form → Tier 3 RX  ("Brand Drugs (Non-Preferred)")
    Non-Preferred Generic → Generic RX (merged with Preferred Generic)
    Generic (non-preferred) suffix form → Generic RX (merged)
    Non-Preferred Specialty → Tier 5 RX
    Specialty Drugs row (standalone, below Tier 4) → Tier 5 RX (joined sub-rows)
    Specialty / Preferred Specialty → Tier 4 RX
    Generic / Preferred Generic → Generic RX (Tier 1)
    Brand / Preferred Brand → Brand RX (Tier 2)
    Bare "Tier N" labels without a descriptor → positional by number

  When a plan splits Generic into Preferred + Non-Preferred rows, both map to
  Generic RX and their costs are joined with " / ".
"""
from __future__ import annotations
import re

# Matches a semicolon used as a TIER separator: "; AlphaLabel: "
# Does NOT match semicolons inside values like "; 90-day supply: " (digit after space)
# or "; maintenance drugs only): " (no colon-space after the label word).
_TIER_SEMICOLON = re.compile(r';\s+(?=[A-Za-z][^;:/]+:\s)')

_COMPUTED_TIER_FIELDS: frozenset[str] = frozenset({
    "Designated Network Generic RX", "Designated Network Brand RX",
    "Designated Network Tier 3 RX", "Designated Network Tier 4 RX",
    "Designated Network Tier 5 RX",
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

def _normalize_tier_label(label: str) -> str:
    """Lowercase, collapse whitespace, normalize common carrier typos."""
    n = re.sub(r'\s+', ' ', label.lower().strip())
    n = n.replace('speciality', 'specialty')
    n = n.replace('preffered', 'preferred')
    # Canonicalize non-preferred variants: nonpreferred / non preferred / non-preffered
    n = re.sub(r'\bnon[\s-]*pref+erred\b', 'non preferred', n)
    n = re.sub(r'\s+drugs?$', '', n).strip()
    n = re.sub(r'^all\s+other\s+', '', n).strip()
    n = re.sub(r'\s*\(tier\s*\d+\)\s*', ' ', n).strip()
    return re.sub(r'\s{2,}', ' ', n)


def _is_generic_label(label: str) -> bool:
    n = _normalize_tier_label(label)
    return 'generic' in n and 'brand' not in n


def _is_preferred_tier_label(label: str) -> bool:
    n = _normalize_tier_label(label)
    return bool(re.search(r'\bpreferred\b', n)) and not re.search(r'\bnon\s+preferred\b', n)


def _is_non_preferred_tier_label(label: str) -> bool:
    return bool(re.search(r'\bnon\s+preferred\b', _normalize_tier_label(label)))


def _explicit_tier_number(label: str) -> int | None:
    """Extract document tier number from '(Tier N)' or leading 'Tier N' in the label."""
    m = re.search(r'\(\s*tier\s*(\d+)\s*\)', label, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r'^\s*tier\s*(\d+)\b', label, re.I)
    if m:
        return int(m.group(1))
    return None


def _is_specialty_drugs_row_label(label: str, context_text: str = "") -> bool:
    """Standalone 'Specialty Drugs' row below Tier 4 — not 'Specialty Drugs (Tier 4)'."""
    if _explicit_tier_number(label) is not None:
        return False
    raw = re.sub(r'\s+', ' ', label.lower().strip())
    if not re.fullmatch(r'specialty\s+drugs?', raw):
        return False
    # Wellmark SBC: numbered Tier 1-4 rows precede the separate Specialty Drugs row.
    return bool(
        re.search(r'\btier\s*1\b', context_text)
        and re.search(r'\btier\s*4\b', context_text)
    )


def _is_plain_tier_label(label: str, kind: str) -> bool:
    """Label contains kind but neither preferred nor non-preferred qualifier."""
    n = _normalize_tier_label(label)
    return (
        kind in n
        and not re.search(r'\bpreferred\b', n)
        and not re.search(r'\bnon\s+preferred\b', n)
    )


def _contextual_tier_index(label: str, tier_idx: int | None, context_text: str) -> int | None:
    """Refine ambiguous plain labels when preferred counterparts exist in same RX block."""
    if tier_idx is None:
        return None
    if (
        _is_plain_tier_label(label, "brand")
        and "preferred brand" in context_text
        and "non preferred brand" not in context_text
    ):
        return 2  # plain Brand means non-preferred Brand when Preferred Brand row exists
    if (
        _is_plain_tier_label(label, "specialty")
        and "preferred specialty" in context_text
        and "non preferred specialty" not in context_text
    ):
        return 4  # plain Specialty means non-preferred Specialty when Preferred row exists
    return tier_idx


def _should_merge_duplicate_tier(label: str, tier_idx: int, prev_label: str) -> bool:
    """Merge only when Generic is split into separate Preferred + Non-Preferred rows."""
    if tier_idx != 0 or not _is_generic_label(label) or not _is_generic_label(prev_label):
        return False
    return (
        (_is_preferred_tier_label(label) and _is_non_preferred_tier_label(prev_label))
        or (_is_non_preferred_tier_label(label) and _is_preferred_tier_label(prev_label))
        or (_is_plain_tier_label(label, "generic") and _is_preferred_tier_label(prev_label))
        or (_is_preferred_tier_label(label) and _is_plain_tier_label(prev_label, "generic"))
    )


def _label_to_tier_index(label: str, context_text: str = "") -> int | None:
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
    n = _normalize_tier_label(label)

    # ── Exclude preventive sub-tier labels → skip ────────────────────────────
    # "Preventive" drugs are a special $0 sub-category within a tier (e.g.,
    # "Generic (Preventive): No Charge" alongside "Generic: $20"). The standard
    # tier cost is what we want; skipping preventive labels lets the next entry win.
    if re.search(r'\bpreventive\b', n):
        return None

    # ── Priority 1: Non-Preferred Brand → ALWAYS Tier 3 ──────────────────────
    # Catches: "Non-Preferred Brand", "Brand Drugs (Non-Preferred)", "Brand (non-preferred)"
    if re.search(r'non\s+preferred\s+brand', n) or (
        re.search(r'\bbrand\b', n) and re.search(r'\bnon\s+preferred\b', n)
    ):
        return 2

    # ── Priority 2: Non-Preferred Generic → Generic RX (merge with Preferred) ─
    # Catches: "Non-Preferred Generic", "Generic Drugs (Non-Preferred)"
    if re.search(r'non\s+preferred\s+generic', n) or (
        re.search(r'\bgeneric\b', n) and re.search(r'\bnon\s+preferred\b', n)
    ):
        return 0

    # ── Priority 3: Non-Preferred Specialty → Tier 5 ─────────────────────────
    # Catches: "Non-Preferred Specialty", "Specialty (Non-Preferred)", "Speciality (non-preferred)"
    if re.search(r'non\s+preferred\s+specialty', n) or (
        'specialty' in n and re.search(r'\bnon\s+preferred\b', n)
    ):
        return 4

    # ── Tier 5 by number ─────────────────────────────────────────────────────
    if re.fullmatch(r'tier\s*5', n):
        return 4

    # ── Standalone Specialty Drugs row (Wellmark SBC below Tier 4) → Tier 5 ─
    if _is_specialty_drugs_row_label(label, context_text):
        return 4

    # ── Specialty (anything remaining with "specialty") → Tier 4 ─────────────
    # Catches: "Specialty", "Preferred Specialty", "Specialty Drugs (Tier 4)",
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


def _merge_split_tier_value(current: str, addition: str) -> str:
    """Join preferred + non-preferred rows; collapse when costs are identical."""
    if not addition:
        return current
    if not current:
        return addition
    if current.strip() == addition.strip():
        return current
    return current + " / " + addition


# Parenthetical channel markers — carriers use many synonyms.
_MAIL_CHANNEL = re.compile(r'\(\s*(?:mail[\s-]*order|home\s+delivery)\s*\)', re.I)
_RETAIL_CHANNEL = re.compile(r'\(\s*retail(?:\s+only)?\s*\)', re.I)


def _split_retail_mail(value: str) -> tuple[str, str | None]:
    """Split a tier value with explicit retail + mail/home-delivery qualifiers.

    Returns (retail_value, mail_value_or_None):
      "$5 / prescription (retail), $10 / prescription (mail order)" → ("$5", "$10")
      "$15 (retail), $30 (home delivery)"                          → ("$15", "$30")
      "$90 copay (retail only)"                                      → ("$90 copay", None)
      "10% coinsurance up to $250 / prescription"                   → (unchanged, None)

    Splits when BOTH retail and mail/home-delivery markers appear (comma- or
    semicolon-separated).  Lone retail markers are left for _extract_retail_only.
    """
    if not value:
        return value, None
    if not (_RETAIL_CHANNEL.search(value) and _MAIL_CHANNEL.search(value)):
        return value, None
    # "X (retail) and Y (home delivery)" — handled by _extract_home_delivery
    if re.search(r'\(\s*retail\s*\)\s+and\b', value, re.I):
        return value, None
    m_re = re.search(r'([^,;]*?)\s*\(\s*retail(?:\s+only)?\s*\)', value, re.I)
    m_mo = re.search(r'([^,;]*?)\s*\(\s*(?:mail[\s-]*order|home\s+delivery)\s*\)', value, re.I)
    retail = (m_re.group(1) if m_re else value).strip(" ,")
    mail = (m_mo.group(1) if m_mo else "").strip(" ,")
    retail = re.sub(r'\s*/\s*prescription$', '', retail, flags=re.I).strip()
    mail = re.sub(r'\s*/\s*prescription$', '', mail, flags=re.I).strip()
    return retail, mail or None


def _extract_retail_only(value: str) -> str:
    """Isolate the retail / 30-day cost from a per-tier cell value.

    Handles Anthem-style dual costs and lone channel qualifiers:
      "$20 copay (retail) and $40 copay (home delivery)" → "$20 copay"
      "30% coinsurance (retail and home delivery)"       → "30% coinsurance"
      "$90 copay (retail only)"                          → "$90 copay"
      "$5 / prescription (retail)"                       → "$5"
      "$20 copay"                                        → "$20 copay"
    """
    if not value:
        return value
    # Dual-cost: "X (retail) and Y (home delivery|mail order)" — keep X only
    m = re.match(r'^(.*?)\s*\(\s*retail\s*\)\s+and\b.*', value, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Same rate for both channels
    m = re.match(r'^(.*?)\s*\(retail and home delivery\)', value, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Lone qualifiers — strip marker, keep cost
    cleaned = re.sub(r'\s*\(\s*retail\s+only\s*\)', '', value, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'\s*/\s*prescription\s*\(\s*retail\s*\)\s*$', '', cleaned, flags=re.I).strip()
    cleaned = re.sub(r'\s*\(\s*retail\s*\)\s*$', '', cleaned, flags=re.IGNORECASE).strip(" ,")
    return cleaned


def _extract_home_delivery(value: str) -> str:
    """Extract mail-order / home-delivery cost from a per-tier cell value.

      "$20 copay (retail) and $40 copay (home delivery)" → "$40 copay"
      "$15 (retail), $30 (mail order)"                  → "$30"
      "30% coinsurance (retail and home delivery)"       → "30% coinsurance"
      "$20 copay (retail only)"                          → ""
    """
    if not value:
        return ""
    if re.search(r'\(\s*retail\s+only\s*\)', value, re.I):
        return ""
    m = re.search(
        r'\(\s*retail\s*\)\s+and\s+(.*?)\s*\(\s*(?:home\s+delivery|mail[\s-]*order)\s*\)',
        value, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    m = re.match(r'^(.*?)\s*\(retail and home delivery\)', value, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _split_unlabeled_retail_mail(value: str) -> tuple[str, str | None]:
    """Split VLM output where retail and mail are adjacent with no channel markers.

    Anthem designated-column pattern:
      '$5 / $10'  → retail '$5', mail '$10'
      '$50 / $125' → retail '$50', mail '$125'

    Does NOT split cross-network retail merges like '$80 / $90' (Preferred+INN) —
    call only for single-network columns where dual flat-dollar pairs mean channels.
    """
    parts = [p.strip() for p in value.split(" / ") if p.strip()]
    if len(parts) != 2:
        return value, None
    a, b = parts
    if "%" in a or "%" in b or "coinsurance" in a.lower() or "coinsurance" in b.lower():
        return value, None
    if re.match(r'^\$\d', a) and re.match(r'^\$\d', b):
        return a, b
    return value, None


def _normalize_tier_retail_value(value: str, *, split_retail_mail_pairs: bool = False) -> str:
    """Retail portion of a tier cell, channel markers removed and suffix stripped."""
    if not value:
        return value
    retail, _ = _split_retail_mail(value)
    retail = _extract_retail_only(retail)
    if split_retail_mail_pairs:
        r, m = _split_unlabeled_retail_mail(retail)
        if m:
            retail = r
    return _strip_rx_suffix(retail)


def _extract_mail_from_tier_value(value: str, *, split_retail_mail_pairs: bool = False) -> str:
    """Mail-order portion of a tier cell, or empty string when none."""
    if not value:
        return ""
    hd = _extract_home_delivery(value)
    if hd:
        return _strip_rx_suffix(hd)
    _, mail = _split_retail_mail(value)
    if mail:
        return _strip_rx_suffix(mail)
    if split_retail_mail_pairs:
        _, mail = _split_unlabeled_retail_mail(value)
        if mail:
            return _strip_rx_suffix(mail)
    return ""


def _extract_tier_values_with_mail(
    consolidated: str,
    *,
    split_retail_mail_pairs: bool = False,
) -> tuple[dict[int, str], dict[int, str]]:
    """Parse consolidated RX into per-tier retail and mail-order dicts."""
    raw = _extract_tier_values(consolidated)
    retail: dict[int, str] = {}
    mail: dict[int, str] = {}
    for idx, val in raw.items():
        retail[idx] = _normalize_tier_retail_value(
            val, split_retail_mail_pairs=split_retail_mail_pairs,
        )
        m = _extract_mail_from_tier_value(
            val, split_retail_mail_pairs=split_retail_mail_pairs,
        )
        if m:
            mail[idx] = m
    return retail, mail


def _build_mail_order_from_consolidated(
    consolidated: str,
    *,
    split_retail_mail_pairs: bool = False,
) -> str:
    """Join per-tier mail-order costs found in a consolidated RX string."""
    raw = _extract_tier_values(consolidated)
    parts: list[str] = []
    for idx in sorted(raw):
        m = _extract_mail_from_tier_value(
            raw[idx], split_retail_mail_pairs=split_retail_mail_pairs,
        )
        if m:
            parts.append(m)
    return " / ".join(parts)


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
    context_text = _normalize_tier_label(consolidated)

    # Newline-separated output: each line is one complete tier entry and its
    # value is kept whole — ' / ' inside a line is part of the value (e.g.
    # '10% coinsurance up to $250 / prescription'), NOT a tier separator.
    if "\n" in consolidated:
        result_nl: dict[int, str] = {}
        labels_nl: dict[int, str] = {}
        for line in consolidated.splitlines():
            line = line.strip()
            if not line or ": " not in line or not line[0].isalpha():
                continue
            label, _, value = line.partition(": ")
            label = label.strip()
            tier_idx = _contextual_tier_index(
                label,
                _label_to_tier_index(label, context_text),
                context_text,
            )
            if tier_idx is not None:
                if tier_idx not in result_nl:
                    result_nl[tier_idx] = value.strip()
                    labels_nl[tier_idx] = label
                elif _should_merge_duplicate_tier(label, tier_idx, labels_nl[tier_idx]):
                    result_nl[tier_idx] = _merge_split_tier_value(
                        result_nl[tier_idx], value.strip(),
                    )
        return result_nl

    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    result: dict[int, str] = {}
    tier_labels: dict[int, str] = {}
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
            tier_idx = _contextual_tier_index(
                label,
                _label_to_tier_index(label, context_text),
                context_text,
            )
            if tier_idx is not None:
                sub, cost = _split_sublabel(value)
                nested = sub is not None
                if tier_idx not in result:
                    if nested and _is_preventive(sub):
                        result[tier_idx] = ""   # anchored; awaiting non-preventive sub-rows
                    else:
                        result[tier_idx] = cost if nested else value
                    tier_labels[tier_idx] = label
                    last_mapped_idx = tier_idx
                    subrow_mode = nested
                elif subrow_mode and last_mapped_idx == tier_idx:
                    # Sub-row re-using the tier label (e.g. 'All Other Generic: $20')
                    if not (nested and _is_preventive(sub)):
                        result[tier_idx] = _join_tier_value(result[tier_idx], cost)
                elif _should_merge_duplicate_tier(label, tier_idx, tier_labels[tier_idx]):
                    addition = cost if nested else value
                    result[tier_idx] = _merge_split_tier_value(result[tier_idx], addition)
                    last_mapped_idx = tier_idx
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


def _strip_rx_suffix(value: str) -> str:
    """Strip per-prescription descriptor suffixes from RX cost values.

    Operates per-part on slash-separated strings so each cost is cleaned
    independently before rejoining.

    Examples:
      "$20 copay per prescription, deductible does not apply" → "$20"
      "$5/prescription, deductible does not apply"            → "$5"
      "$80 copay per prescription after deductible is met"   → "$80"
      "30% coinsurance up to $400 per prescription..."       → "30% coinsurance up to $400"
      "30% coinsurance up to $250/prescription"              → unchanged
      "$80 copay... / $90 copay..."                          → "$80 / $90"
    """
    if not value:
        return value
    parts = [p.strip() for p in value.split(" / ")]
    stripped: list[str] = []
    for p in parts:
        # "$X/prescription[, deductible...]" → "$X"  (flat-copay shorthand; not coinsurance)
        m = re.match(r'^(\$\d+(?:\.\d+)?)/prescription\b.*$', p, flags=re.IGNORECASE)
        if m:
            stripped.append(m.group(1))
            continue
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
    pref = {idx: _normalize_tier_retail_value(v) for idx, v in pref_raw.items()}
    inn  = {idx: _normalize_tier_retail_value(v) for idx, v in inn_raw.items()}
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
            # RX Deductible fields: a zero/none deductible means "no RX deductible"
            # -> empty string ("$0", "0", "None", "No deductible" are all noise).
            elif key.endswith("RX Deductible") and val.strip().lower() in (
                "$0", "0", "none", "no deductible", "does not apply"
            ):
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
    # health (2-col): Preferred Network RX + In-Network RX merge into In-Network per-tier.
    # health_3tier: Designated Network RX, In-Network RX, and Out-of-Network RX each
    # feed their own per-tier fields — no cross-column merge.
    preferred_rx = result.get("Preferred Network RX") or ""
    designated_tier_fields = any(
        f"Designated Network {suffix}" in output_field_names
        for suffix in _TIER_SUFFIXES
    )
    net_prefixes: tuple[str, ...] = (
        ("Designated Network", "In-Network", "Out-of-Network")
        if designated_tier_fields
        else ("In-Network", "Out-of-Network")
    )
    for net_prefix in net_prefixes:
        # Propagate "Not covered" to all per-tier OON fields when applicable.
        if net_prefix == "Out-of-Network" and oon_rx_not_covered:
            for suffix in _TIER_SUFFIXES:
                field_name = f"Out-of-Network {suffix}"
                if field_name in output_field_names:
                    result[field_name] = "Not covered"
            continue

        consolidated = result.get(f"{net_prefix} RX") or ""
        split_pairs = designated_tier_fields and net_prefix == "Designated Network"
        if (
            net_prefix == "In-Network"
            and preferred_rx
            and not designated_tier_fields
        ):
            tier_values = _merge_preferred_and_inn_tier_values(preferred_rx, consolidated)
        else:
            tier_values, _explicit_mail = _extract_tier_values_with_mail(
                consolidated, split_retail_mail_pairs=split_pairs,
            )
        for i, suffix in enumerate(_TIER_SUFFIXES):
            field_name = f"{net_prefix} {suffix}"
            if field_name in output_field_names:
                result[field_name] = tier_values.get(i, "")

    # Pass 2b: drop retail coinsurance values the VLM copied into Mail Order.
    # A mail-order part that verbatim-equals a retail per-tier value AND is a
    # coinsurance expression is almost certainly a retail-only tier (typically
    # Specialty, "Up to a 30-day supply (retail)") wrongly attributed to mail
    # order — flat-dollar duplicates are left alone since 90-day mail prices
    # can legitimately coincide with retail dollars.
    def _canon(s: str) -> str:
        s = re.sub(r'\s*/?\s*(?:per\s+)?prescription\b', '', s.lower())
        return re.sub(r'\s+', ' ', s).strip(' ,.')

    retail_tier_values = {
        _canon(result.get(f"{np} {sfx}") or "")
        for np in ("In-Network", "Out-of-Network", "Designated Network")
        for sfx in _TIER_SUFFIXES
    } - {""}
    for mo_field in _MAIL_ORDER_RX_FIELDS:
        # Designated mail order often legitimately repeats retail coinsurance rates
        # for specialty tiers — skip the retail-dedup heuristic for that column.
        if mo_field == "Designated Network Mail Order RX":
            continue
        val = result.get(mo_field)
        if isinstance(val, str) and val and retail_tier_values:
            kept = [
                p for p in (part.strip() for part in val.split(" / "))
                if p and _canon(p) and not (
                    ("%" in p or "coinsurance" in p.lower())
                    and _canon(p) in retail_tier_values
                )
            ]
            result[mo_field] = " / ".join(kept)

    # Pass 3: strip tier labels from Mail Order RX fields.
    # Mail Order fields display 90-day costs as "$20 / $80", not "Tier 1: $20 / Tier 2: $80".
    for field in _MAIL_ORDER_RX_FIELDS:
        val = result.get(field)
        if isinstance(val, str) and val:
            result[field] = _strip_tier_labels(val)

    # Pass 4: Derive mail order from channel markers in each network's RX column.
    mail_sources: list[tuple[str, str, str, bool]] = []
    for net_prefix in net_prefixes:
        rx = result.get(f"{net_prefix} RX") or ""
        mo_field = f"{net_prefix} Mail Order RX"
        if mo_field in output_field_names:
            split_pairs = designated_tier_fields and net_prefix == "Designated Network"
            mail_sources.append((net_prefix, rx, mo_field, split_pairs))
    if (
        preferred_rx
        and not designated_tier_fields
        and "In-Network Mail Order RX" in output_field_names
    ):
        mail_sources = [
            ("In-Network", preferred_rx, "In-Network Mail Order RX", False),
            *[
                (np, rx, mo, sp)
                for np, rx, mo, sp in mail_sources
                if mo != "In-Network Mail Order RX"
            ],
        ]
    for net_prefix, consolidated, mo_field, split_pairs in mail_sources:
        derived = _build_mail_order_from_consolidated(
            consolidated, split_retail_mail_pairs=split_pairs,
        )
        if derived:
            result[mo_field] = derived
        elif designated_tier_fields and net_prefix != "Designated Network":
            # health_3tier INN/OON columns with no embedded mail → no mail benefit.
            # Clears VLM mail wrongly copied from the Designated column.
            result[mo_field] = ""

    return result
