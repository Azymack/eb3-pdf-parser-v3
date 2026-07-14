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
    Specialty Drugs embedded in tier cells → Tier 5 RX (joined, retail cells cleaned)
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

# Comma between drug ROW labels (Premera-style): "Generic drugs: $20, Preferred brand drugs: $35"
# Does NOT match UHC in-cell channels: "..., Mail-Order: $50" or "Tier 1: ..., Specialty Drugs: $25"
_TIER_COMMA_BOUNDARY = re.compile(
    r',\s+(?='
    r'(?:Generic\s+drugs?|Preferred\s+brand(?:\s+drugs?)?|'
    r'Non[-\s]+preferred\s+brand(?:\s+drugs?)?|Specialty\s+drugs?)'
    r'\s*:'
    r')',
    re.I,
)

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
    # Anthem "Typically X (Tier N)" and "Tier N (Typically X)" rows — the document's
    # tier number is authoritative for these combined drug-class labels.
    if re.match(r'typically\b', label, re.I):
        m = re.search(r'\(\s*tier\s*([1-5])\s*\)', label, re.I)
        if m:
            return int(m.group(1)) - 1
    m = re.match(r'tier\s*([1-5])\s*\(\s*typically\b', label, re.I)
    if m:
        return int(m.group(1)) - 1

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


def _derived_mail_pollutes_retail(derived: str) -> bool:
    return "(retail)" in derived.lower()


def _should_prefer_existing_mail_order(existing: str, derived: str) -> bool:
    """Keep VLM mail-only values when consolidated derivation mixes in retail costs."""
    if not existing or not derived:
        return False
    if _derived_mail_pollutes_retail(derived) and not _derived_mail_pollutes_retail(existing):
        return True
    return False


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


# Parenthetical specialty costs inside a tier cell, e.g. "(Specialty Drugs: $150 copay)".
_PAREN_SPECIALTY_DRUGS = re.compile(
    r'\(\s*Specialty\s+Drugs?\s*\*?\s*:?\s*([^)]+?)\s*\)',
    re.IGNORECASE,
)


def _normalize_specialty_cost_fragment(value: str) -> str:
    if not value:
        return ""
    return _strip_rx_suffix(_normalize_tier_retail_value(value))


def _strip_embedded_specialty_from_value(value: str) -> str:
    """Remove inline (Specialty Drugs: $X) fragments from a retail tier value."""
    if not value:
        return value
    parts: list[str] = []
    for part in value.split(" / "):
        cleaned = _PAREN_SPECIALTY_DRUGS.sub("", part).strip(" ,")
        if cleaned:
            parts.append(cleaned)
    return " / ".join(parts)


_UHC_CHANNEL_BOUNDARY = re.compile(
    r"\s*/\s*(?=(?:Retail|Mail[\s-]*Order|Specialty\s+Drugs?)\s*\**\s*(?::|\$))",
    re.I,
)
_UHC_CHANNEL_MARKER = re.compile(
    r"(?:^|[,/]\s*)(Retail|Mail[\s-]*Order|Specialty\s+Drugs?)\s*\**\s*:\s*",
    re.I,
)
_UHC_CHANNEL_SEG = re.compile(
    r"^(?:.*?\s)?(Retail|Mail[\s-]*Order|Specialty\s+Drugs?)\s*\**\s*"
    r"(?::\s*(.+)|\$\s*(.+))$",
    re.I,
)

_BOILERPLATE_TIER_VALUE = re.compile(
    r"^deductible\s+does\s+not\s+apply\.?$",
    re.I,
)


def _value_has_uhc_channels(value: str) -> bool:
    """UHC SBC cells: 'Retail: $25 / Mail-Order: $50' or '$25, Mail-Order: $50'."""
    return bool(re.search(
        r"(?:Retail|Mail[\s-]*Order|Specialty\s+Drugs?)\s*\**\s*(?::|\$)",
        value, re.I,
    ))


def _has_uhc_channel_markers(value: str) -> bool:
    return _value_has_uhc_channels(value)


def _buf_has_uhc_channel_cell(buf: list[str]) -> bool:
    return _value_has_uhc_channels(" / ".join(buf))


def _parse_uhc_multi_channel_cell(value: str) -> tuple[str, str | None, list[str]]:
    """Split one tier cell into retail, mail-order, and specialty channel costs.

    Handles slash-separated ('Retail: $25 / Mail-Order: $50') and comma-separated
    ('$25 copay, Mail-Order: $50 copay, Specialty Drugs: $25') UHC SBC layouts.
    """
    value = value.strip()
    if not value or not _value_has_uhc_channels(value):
        return value, None, []

    markers = list(_UHC_CHANNEL_MARKER.finditer(value))
    if markers:
        retail_parts: list[str] = []
        mail_parts: list[str] = []
        specialty_parts: list[str] = []
        if markers[0].start() > 0:
            leading = value[:markers[0].start()].strip(" ,.")
            if leading and not _BOILERPLATE_TIER_VALUE.match(leading.rstrip(".")):
                retail_parts.append(leading)
        for i, m in enumerate(markers):
            name = re.sub(r"[\s-]+", " ", m.group(1).lower()).strip()
            chunk = value[m.end(): markers[i + 1].start() if i + 1 < len(markers) else len(value)]
            chunk = chunk.strip(" ,/")
            if not chunk:
                continue
            if name == "retail":
                retail_parts.append(chunk)
            elif name.startswith("mail"):
                mail_parts.append(chunk)
            elif name.startswith("specialty"):
                specialty_parts.append(chunk)
        retail = " / ".join(retail_parts)
        mail = " / ".join(mail_parts) if mail_parts else None
        return retail, mail, specialty_parts

    return _parse_uhc_slash_channel_cell(value)


def _parse_uhc_slash_channel_cell(value: str) -> tuple[str, str | None, list[str]]:
    """Fallback: 'Retail $10 / Mail-Order $20 / Specialty Drugs $10' (no colons)."""
    retail_parts: list[str] = []
    mail_parts: list[str] = []
    specialty_parts: list[str] = []
    for part in _UHC_CHANNEL_BOUNDARY.split(value) if _UHC_CHANNEL_BOUNDARY.search(value) else [value]:
        part = part.strip()
        if not part:
            continue
        channel, cost = _match_uhc_channel_segment(part)
        if channel == "retail":
            retail_parts.append(cost)
        elif channel == "mail":
            mail_parts.append(cost)
        elif channel == "specialty":
            specialty_parts.append(cost)
        elif not retail_parts and not mail_parts and not specialty_parts:
            retail_parts.append(part)
    retail = " / ".join(retail_parts) if retail_parts else value
    mail = " / ".join(mail_parts) if mail_parts else None
    return retail, mail, specialty_parts


def _split_on_uhc_channel_segments(value: str) -> list[str]:
    """Split a tier cell into channel segments for legacy callers."""
    retail, mail, specialty = _parse_uhc_multi_channel_cell(value)
    segments: list[str] = []
    if retail:
        segments.append(retail)
    if mail:
        segments.append(f"Mail-Order: {mail}")
    for sp in specialty:
        segments.append(f"Specialty Drugs: {sp}")
    return segments or [value]


def _match_uhc_channel_segment(segment: str) -> tuple[str | None, str]:
    segment = segment.strip()
    m = _UHC_CHANNEL_SEG.match(segment)
    if not m:
        return None, segment
    name = re.sub(r'[\s-]+', ' ', m.group(1).lower()).strip()
    cost = (m.group(2) or f"${m.group(3)}").strip()
    if name.startswith("mail"):
        return "mail", cost
    if name.startswith("specialty"):
        return "specialty", cost
    return "retail", cost


def _split_uhc_channel_segment(segment: str) -> tuple[str | None, str]:
    return _match_uhc_channel_segment(segment)


def _split_uhc_retail_mail(value: str) -> tuple[str, str | None]:
    """UHC per-tier cell — retail and mail-order only (specialty handled separately)."""
    retail, mail, _ = _parse_uhc_multi_channel_cell(value)
    if _value_has_uhc_channels(value):
        return retail, mail
    return value, None


def _strip_inline_specialty_words(value: str) -> tuple[str, list[str]]:
    """Pull Specialty Drugs channel segments out of a tier cell value."""
    retail, _, specialty = _parse_uhc_multi_channel_cell(value)
    if _value_has_uhc_channels(value):
        return retail, specialty
    return value, []


def _segment_tier_drug_label(seg: str, context_text: str = "") -> str | None:
    """Return label when segment starts a new mapped drug tier row."""
    if ": " not in seg or not seg or not seg[0].isalpha():
        return None
    label = seg.partition(": ")[0].strip()
    if _label_to_tier_index(label, context_text) is not None:
        return label
    return None


def _normalize_uhc_specialty_seg(seg: str) -> str:
    """PDF footnote markers: 'Specialty Drugs**: $10' -> 'Specialty Drugs: $10'."""
    return re.sub(r"Specialty\s+Drugs?\s*\*+\s*:", "Specialty Drugs:", seg.strip(), flags=re.I)


def _is_standalone_uhc_channel_segment(seg: str, context_text: str = "") -> bool:
    """Retail/Mail-Order channel continuation — not a new tier or Specialty Drugs row."""
    s = seg.strip()
    if re.match(r"^Tier\s*\d+", s, re.I):
        return False
    if _segment_tier_drug_label(s, context_text):
        return False
    channel, _ = _match_uhc_channel_segment(s)
    return channel in ("retail", "mail")


def _is_inline_specialty_segment(seg: str, buf: list[str]) -> bool:
    """Single-tier Specialty Drugs cost inside a Retail/Mail-Order cell — not Tier 5 row."""
    if not buf or not _buf_has_uhc_channel_cell(buf):
        return False
    seg = _normalize_uhc_specialty_seg(seg)
    channel, cost = _match_uhc_channel_segment(seg)
    if channel != "specialty":
        m = re.match(r"^Specialty\s+Drugs?\s*:\s*(.+)$", seg, re.I)
        if not m:
            return False
        cost = m.group(1).strip()
    return bool(cost) and not re.search(r"\s/\s+\$", cost)


def _apply_embedded_specialty_tier5(tier_values: dict[int, str]) -> dict[int, str]:
    """Move (Specialty Drugs: $X) fragments from tier 1-4 cells into Tier 5 RX."""
    if not tier_values:
        return tier_values

    result = dict(tier_values)
    if result.get(4, "").strip():
        for idx in list(result):
            if idx <= 3:
                result[idx] = _strip_embedded_specialty_from_value(result[idx])
        return result

    specialty_parts: list[str] = []
    for idx in sorted(k for k in result if k <= 3):
        val = result[idx]
        for part in val.split(" / "):
            for match in _PAREN_SPECIALTY_DRUGS.finditer(part):
                specialty_parts.append(match.group(1).strip())
        cleaned, inline_specialty = _strip_inline_specialty_words(val)
        specialty_parts.extend(inline_specialty)
        result[idx] = _strip_embedded_specialty_from_value(cleaned)

    if specialty_parts:
        result[4] = " / ".join(specialty_parts)
    return result


def _canon_tier_part(part: str) -> str:
    """Normalize a slash-separated tier fragment for duplicate comparison."""
    s = re.sub(r"\s*/?\s*(?:per\s+)?prescription\b", "", part.lower())
    return re.sub(r"\s+", " ", s).strip(" ,.")


def _dedupe_adjacent_tier_parts(value: str) -> str:
    """Collapse consecutive duplicate values after Preferred/Participating splits."""
    if not value:
        return value
    parts = [p.strip() for p in value.split(" / ") if p.strip()]
    out: list[str] = []
    for part in parts:
        if not out or _canon_tier_part(part) != _canon_tier_part(out[-1]):
            out.append(part)
    return " / ".join(out)


def _finalize_tier_value_dict(
    raw_tiers: dict[int, str],
    *,
    split_retail_mail_pairs: bool = False,
) -> dict[int, str]:
    """Normalize parsed tier values, including Tier 5 specialty aggregates."""
    result: dict[int, str] = {}
    for idx, val in raw_tiers.items():
        if idx <= 3:
            normalized = _normalize_tier_retail_value(
                val, split_retail_mail_pairs=split_retail_mail_pairs,
            )
            if idx == 0:
                normalized = _dedupe_adjacent_tier_parts(normalized)
            result[idx] = normalized
        else:
            parts = [
                _normalize_specialty_cost_fragment(part)
                for part in val.split(" / ")
                if part.strip()
            ]
            result[idx] = " / ".join(part for part in parts if part)
    return result


def _preferred_rx_redundant_with_inn(preferred_rx: str, inn_rx: str) -> bool:
    """True when Preferred Network RX duplicates a prefix of In-Network RX."""
    p, i = preferred_rx.strip(), inn_rx.strip()
    if not p:
        return True
    if p == i:
        return True
    if not i.startswith(p):
        return False
    rest = i[len(p):].lstrip()
    if rest and not rest.startswith("/"):
        return False
    pref_tiers = _extract_tier_values(p)
    inn_tiers = _extract_tier_values(i)
    return all(inn_tiers.get(k) == v for k, v in pref_tiers.items())


def _parse_retail_tier_values(
    consolidated: str,
    *,
    split_retail_mail_pairs: bool = False,
) -> dict[int, str]:
    raw = _apply_embedded_specialty_tier5(_extract_tier_values(consolidated))
    return _finalize_tier_value_dict(
        raw, split_retail_mail_pairs=split_retail_mail_pairs,
    )


# Parenthetical channel markers — carriers use many synonyms.
_MAIL_CHANNEL = re.compile(r'\(\s*(?:mail[\s-]*order|home\s+delivery)\s*\)', re.I)
_RETAIL_CHANNEL = re.compile(r'\(\s*retail(?:\s+only)?\s*\)', re.I)
_SUPPLY_RETAIL = re.compile(r'/\s*retail\s+supply\b', re.I)
_SUPPLY_MAIL = re.compile(r'/\s*mail\s+service\s+supply\b', re.I)


def _has_supply_channel_markers(value: str) -> bool:
    return bool(_SUPPLY_RETAIL.search(value) or _SUPPLY_MAIL.search(value))


def _extract_cost_from_supply_segment(segment: str) -> str:
    """'$10 / retail supply for ...' -> '$10'; '50% coinsurance / retail supply' -> '50% coinsurance'."""
    segment = segment.strip()
    for pattern in (r'^(.*?)\s*/\s*retail\s+supply\b', r'^(.*?)\s*/\s*mail\s+service\s+supply\b'):
        m = re.match(pattern, segment, re.I)
        if m:
            return m.group(1).strip()
    return segment


def _strip_oon_mail_service_tail(clause: str) -> str:
    return re.sub(r'\s+and\s+all\s+charges\s+for\s+mail\s+service.*$', '', clause, flags=re.I).strip()


def _split_retail_mail_supply_clause(clause: str) -> tuple[str, str | None]:
    """Split one BCBS-style clause with '/ retail supply' and '/ mail service supply'."""
    clause = _strip_oon_mail_service_tail(clause.strip())
    if not clause:
        return "", None

    if re.search(
        r'/\s*retail\s+supply\b.*?\bor\b.*?/\s*mail\s+service\s+supply\b',
        clause, re.I,
    ):
        retail_seg, mail_seg = re.split(r'\s+or\s+', clause, maxsplit=1, flags=re.I)
        return (
            _extract_cost_from_supply_segment(retail_seg),
            _extract_cost_from_supply_segment(mail_seg) or None,
        )

    if _SUPPLY_RETAIL.search(clause) and not _SUPPLY_MAIL.search(clause):
        if re.search(r'\s+or\s+', clause, re.I):
            parts = [
                _extract_cost_from_supply_segment(seg)
                for seg in re.split(r'\s+or\s+', clause, flags=re.I)
                if seg.strip()
            ]
            return " / ".join(parts), None
        return _extract_cost_from_supply_segment(clause), None

    if _SUPPLY_MAIL.search(clause) and not _SUPPLY_RETAIL.search(clause):
        return "", _extract_cost_from_supply_segment(clause)

    if _SUPPLY_RETAIL.search(clause):
        return _extract_cost_from_supply_segment(clause), None
    return clause, None


def _split_retail_mail_supply_value(value: str) -> tuple[str, str | None]:
    """BCBS MA cells: '$10 / retail supply or $20 / mail service supply; ...'."""
    retail_parts: list[str] = []
    mail_parts: list[str] = []
    for clause in re.split(r'\s*;\s*', value):
        if not clause.strip():
            continue
        retail, mail = _split_retail_mail_supply_clause(clause)
        if retail:
            retail_parts.append(retail)
        if mail and not _is_not_covered_mail_fragment(mail):
            mail_parts.append(mail)
    if not retail_parts and not mail_parts:
        return value, None
    return " / ".join(retail_parts), (" / ".join(mail_parts) if mail_parts else None)


def _is_specialty_subrow_label(label: str) -> bool:
    """Short sub-labels inside a Specialty Drugs cell — not tier boundaries."""
    n = _normalize_tier_label(label)
    if n in {"generic", "preferred", "non-preferred"}:
        return True
    return bool(re.match(r'^low[- ]cost\s+generic\b', n))


def _specialty_value_has_inline_subrows(value: str) -> bool:
    for segment in value.split(" / "):
        sub, _ = _split_sublabel(segment.strip())
        if sub is not None and _is_specialty_subrow_label(sub):
            return True
    return False


def _collapse_inline_subrow_costs(value: str) -> str:
    """'Low-Cost Generic: $10 / Generic: $45 / Preferred: 50%' -> '$10 / $45 / 50%'."""
    parts: list[str] = []
    for segment in (s.strip() for s in value.split(" / ") if s.strip()):
        sub, cost = _split_sublabel(segment)
        if sub is not None and "$" not in sub and "%" not in sub:
            if _is_preventive(sub):
                continue
            parts.append(cost)
        else:
            parts.append(segment)
    return " / ".join(parts)


def _is_not_covered_value(value: str) -> bool:
    return value.strip().lower() in _NOT_COVERED_PHRASES


def _is_not_covered_mail_fragment(value: str) -> bool:
    return _is_not_covered_value(value)


def _is_oon_pharmacy_not_covered(oon_rx: str) -> bool:
    """True when the whole OON pharmacy benefit or every parsed tier is Not covered."""
    if not oon_rx:
        return False
    if _is_not_covered_value(oon_rx):
        return True
    normalized = _normalize_comma_tier_separators(oon_rx)
    tiers = _extract_tier_values(normalized)
    return bool(tiers) and all(_is_not_covered_value(v) for v in tiers.values())


def _should_normalize_comma_tier_separators(consolidated: str) -> bool:
    """True for simple multi-row drug tables comma-separated on one line."""
    if not re.search(
        r',\s*(?:Generic\s+drugs?|Preferred\s+brand(?:\s+drugs?)?|'
        r'Non[-\s]+preferred\s+brand(?:\s+drugs?)?)\s*:',
        consolidated,
        re.I,
    ):
        return False
    # UHC per-tier cells: "Tier 1: $25, Mail-Order: $50, Specialty Drugs: $25"
    if re.match(r'\s*Tier\s*\d+', consolidated, re.I):
        return False
    return True


def _normalize_comma_tier_separators(consolidated: str) -> str:
    """Convert comma-separated drug rows to ' / ' tier separators."""
    if not consolidated or not _should_normalize_comma_tier_separators(consolidated):
        return consolidated
    return _TIER_COMMA_BOUNDARY.sub(" / ", consolidated)


def _split_consolidated_tier_parts(consolidated: str) -> list[str]:
    """Split consolidated RX on tier boundaries without breaking in-cell '/ retail supply'
    or Specialty Drugs sub-rows (Generic / Preferred / Non-Preferred)."""
    if not consolidated:
        return []
    consolidated = _normalize_comma_tier_separators(consolidated)
    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    context_text = _normalize_tier_label(consolidated)
    segments = [s.strip() for s in normalized.split(" / ") if s.strip()]
    parts: list[str] = []
    buf: list[str] = []
    inside_specialty_cell = False

    def flush() -> None:
        nonlocal buf
        if buf:
            parts.append(" / ".join(buf))
            buf = []

    for seg in segments:
        if buf and _is_standalone_uhc_channel_segment(seg, context_text):
            buf.append(seg)
            continue
        if buf and _is_inline_specialty_segment(seg, buf):
            buf.append(seg)
            continue
        is_labeled = ": " in seg and seg[0].isalpha()
        if is_labeled:
            label, _, _ = seg.partition(": ")
            label = label.strip()
            tier_idx = _label_to_tier_index(label, context_text)
            if tier_idx is not None:
                is_specialty_row = "specialty" in _normalize_tier_label(label)
                if inside_specialty_cell and _is_specialty_subrow_label(label):
                    buf.append(seg)
                    continue
                flush()
                buf = [seg]
                inside_specialty_cell = is_specialty_row
                continue
            flush()
            buf = [seg]
            continue
        if buf:
            buf.append(seg)
        else:
            buf = [seg]
    flush()
    return parts


def _should_merge_generic_slot(label: str, tier_idx: int, prev_label: str) -> bool:
    """Merge distinct generic rows (e.g. Low-Cost Generic + Generic) into Generic RX."""
    return (
        tier_idx == 0
        and _is_generic_label(label)
        and _is_generic_label(prev_label)
        and label != prev_label
    )


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
    if _has_supply_channel_markers(value):
        return _split_retail_mail_supply_value(value)
    if _has_uhc_channel_markers(value):
        return _split_uhc_retail_mail(value)
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
    if _has_supply_channel_markers(value):
        retail, _ = _split_retail_mail_supply_value(value)
        return retail
    if _has_uhc_channel_markers(value):
        retail, _ = _split_uhc_retail_mail(value)
        return retail
    # Dual-cost: "X (retail) and Y (home delivery|mail order)" — keep X only
    m = re.match(r'^(.*?)\s*\(\s*retail\s*\)\s+and\b.*', value, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Same rate for both channels
    m = re.match(r'^(.*?)\s*\(retail and home delivery\)', value, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Lone qualifiers — strip marker, keep cost
    cleaned = re.sub(
        r"\s*\(\s*retail\s*\)\s*,?\s*deductible does not apply\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r'\s*\(\s*retail\s+only\s*\)', '', cleaned, flags=re.IGNORECASE).strip()
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
    if _has_supply_channel_markers(value):
        _, mail = _split_retail_mail_supply_value(value)
        return mail or ""
    if _has_uhc_channel_markers(value):
        _, mail = _split_uhc_retail_mail(value)
        return mail or ""
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


def _split_preferred_participating_value(value: str) -> str:
    """BCBS SBC: 'Preferred - No Charge Participating - $10/prescription' -> 'No Charge / $10'."""
    if not value or not re.search(r"\bPreferred\s*-", value, re.I):
        return value
    parts = [p.strip() for p in value.split(" / ") if p.strip()]
    out: list[str] = []
    for part in parts:
        m = re.search(
            r"Preferred\s*-\s*(.+?)\s+Participating\s*-\s*(.+)$",
            part,
            re.I,
        )
        if m:
            out.append(f"{m.group(1).strip()} / {m.group(2).strip()}")
        else:
            out.append(part)
    return " / ".join(out)


def _collect_retail_canon_atoms(values: list[str]) -> set[str]:
    """Canonical tokens from per-tier retail values for mail-order dedup."""
    atoms: set[str] = set()
    for val in values:
        if not val:
            continue
        for part in val.split(" / "):
            part = part.strip()
            if not part:
                continue
            cp = re.sub(r"\s*/?\s*(?:per\s+)?prescription\b", "", part.lower())
            cp = re.sub(r"\s+", " ", cp).strip(" ,.")
            if cp:
                atoms.add(cp)
            for m in re.finditer(r"\$\d+(?:\.\d+)?", part):
                atoms.add(m.group(0).lower())
    return atoms


def _should_drop_retail_echo_from_mail(part: str, retail_atoms: set[str]) -> bool:
    """Drop mail-order parts that duplicate a retail per-tier value (VLM copy error)."""
    part = part.strip()
    if not part:
        return True
    cp = re.sub(r"\s*/?\s*(?:per\s+)?prescription\b", "", part.lower())
    cp = re.sub(r"\s+", " ", cp).strip(" ,.")
    if cp in ("no charge", "$0", "0"):
        return True
    if cp in retail_atoms:
        return True
    m = re.fullmatch(r"\$\d+(?:\.\d+)?", part)
    if m and m.group(0).lower() in retail_atoms:
        return True
    return False


def _strip_boilerplate_tier_value(value: str) -> str:
    """Drop disclaimer-only tier values the VLM mistook for a cost (UHC Tier 1)."""
    if not value:
        return value
    stripped = value.strip()
    if not re.search(r"[\$%]", stripped) and _BOILERPLATE_TIER_VALUE.match(stripped):
        return ""
    return value


def _normalize_tier_retail_value(value: str, *, split_retail_mail_pairs: bool = False) -> str:
    """Retail portion of a tier cell, channel markers removed and suffix stripped."""
    if not value:
        return value
    value = _strip_boilerplate_tier_value(value)
    if not value:
        return value
    value = _split_preferred_participating_value(value)
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
    retail = _finalize_tier_value_dict(
        _apply_embedded_specialty_tier5(raw),
        split_retail_mail_pairs=split_retail_mail_pairs,
    )
    mail: dict[int, str] = {}
    for idx, val in raw.items():
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
    consolidated = _normalize_comma_tier_separators(consolidated)
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
                    stored = value.strip()
                    if tier_idx in (3, 4) and "specialty" in _normalize_tier_label(label):
                        if _specialty_value_has_inline_subrows(stored):
                            stored = _collapse_inline_subrow_costs(stored)
                    result_nl[tier_idx] = stored
                    labels_nl[tier_idx] = label
                elif _should_merge_duplicate_tier(label, tier_idx, labels_nl[tier_idx]):
                    result_nl[tier_idx] = _merge_split_tier_value(
                        result_nl[tier_idx], value.strip(),
                    )
                elif _should_merge_generic_slot(label, tier_idx, labels_nl[tier_idx]):
                    result_nl[tier_idx] = _merge_split_tier_value(
                        result_nl[tier_idx], value.strip(),
                    )
        return result_nl

    normalized = _TIER_SEMICOLON.sub(" / ", consolidated)
    result: dict[int, str] = {}
    tier_labels: dict[int, str] = {}
    last_mapped_idx: int | None = None  # most recent successfully mapped tier index
    subrow_mode = False                 # last mapped tier has program sub-rows

    for part in _split_consolidated_tier_parts(consolidated):
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
                        stored = cost if nested else value
                        if tier_idx in (3, 4) and "specialty" in _normalize_tier_label(label):
                            if _specialty_value_has_inline_subrows(value):
                                stored = _collapse_inline_subrow_costs(value)
                        result[tier_idx] = stored
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
                elif _should_merge_generic_slot(label, tier_idx, tier_labels[tier_idx]):
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
        # "$X copay/prescription" / "$X copayment/prescription" → "$X"
        # (tolerate spaces around the slash: "$25 copay/ prescription")
        m = re.match(r'^(\$\d+(?:\.\d+)?)\s*copay(?:ment)?\s*/\s*prescription\s*$', p, flags=re.IGNORECASE)
        if m:
            stripped.append(m.group(1))
            continue
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


def _inn_has_six_row_pref_nonpref(inn_rx: str) -> bool:
    """True when In-Network RX uses the BCBS 6-row Preferred/Non-Preferred pharmacy table."""
    ctx = _normalize_tier_label(inn_rx)
    if re.search(r"\btier\s*[1-4]\b", ctx) and not re.search(
        r"generic\s*\(\s*preferred\s*\)", ctx
    ):
        return False
    return all(
        re.search(pat, ctx)
        for pat in (
            r"generic(?:\s+drugs?)?\s*\(\s*preferred\s*\)",
            r"generic(?:\s+drugs?)?\s*\(\s*non\s+preferred\s*\)",
            r"brand(?:\s+drugs?)?\s*\(\s*preferred\s*\)",
            r"brand(?:\s+drugs?)?\s*\(\s*non\s+preferred\s*\)",
        )
    )


def _looks_like_pharmacy_cost(value: str) -> bool:
    """True when value looks like a drug cost, not a bare network label."""
    v = value.strip().lower()
    if not v or v in _NOT_COVERED_PHRASES:
        return False
    return bool(re.search(r"[\$%]|coinsurance|copay", v))


def _is_unlabeled_flat_oon_pharmacy(oon_rx: str) -> bool:
    """OON pharmacy string with a cost but no mappable tier labels."""
    if not oon_rx or not _looks_like_pharmacy_cost(oon_rx):
        return False
    if _extract_tier_values(oon_rx):
        return False
    if re.search(
        r"\b(?:generic|brand|tier\s*\d|preferred|non[-\s]+preferred|specialty)\s*:",
        oon_rx,
        re.I,
    ):
        return False
    return True


def _broadcast_flat_oon_pharmacy(
    oon_rx: str,
    inn_tier_values: dict[int, str],
    inn_rx: str,
) -> dict[int, str]:
    """Apply one OON retail rate to Generic/Brand/Tier 3 when VLM omits row labels.

    Aetna-style SBCs show the same Out-of-Network cell (e.g. '50% coinsurance (retail)')
    on Preferred generic, Preferred brand, and Non-preferred rows. The VLM often outputs
    that once without labels; broadcast to INN tier slots 0–2. When INN has a specialty
    tier and OON omits it, specialty is usually 'Not covered' on these layouts.
    """
    if not _is_unlabeled_flat_oon_pharmacy(oon_rx) or not inn_tier_values:
        return {}
    val = _strip_rx_suffix(_normalize_tier_retail_value(oon_rx))
    if not val:
        return {}
    out: dict[int, str] = {}
    for idx in (0, 1, 2):
        if idx in inn_tier_values and inn_tier_values[idx]:
            out[idx] = val
    inn_ctx = _normalize_tier_label(inn_rx)
    if re.search(r"\bspecialty\b", inn_ctx) and not re.search(r"\bspecialty\b", oon_rx, re.I):
        spec_idx = max((i for i in inn_tier_values if i >= 3), default=None)
        if spec_idx is not None:
            if re.search(r"\bnot\s+covered\b", oon_rx, re.I):
                out[spec_idx] = "Not covered"
            elif len(out) >= 1:
                out[spec_idx] = "Not covered"
    return out


def _fix_oon_tier3_brand_split(
    tier_values: dict[int, str],
    oon_rx: str,
    inn_rx: str,
) -> dict[int, str]:
    """Move a second Brand value into Tier 3 for OON 6-row tables.

    In-Network Brand RX often has two retail costs on Brand (Preferred) via
    continuation ('$50 / $70').  On Out-of-Network the same continuation pattern
    is '$70 / $120' where $70 is Brand (Preferred) and $120 is Brand
    (Non-Preferred) — but the VLM may omit the Non-Preferred label and append
    $120 to Brand (Preferred) instead.
    """
    if not _inn_has_six_row_pref_nonpref(inn_rx):
        return tier_values
    if tier_values.get(2):
        return tier_values
    oon_ctx = _normalize_tier_label(oon_rx)
    if re.search(r"\bbrand\s*\(\s*non\s+preferred\s*\)", oon_ctx):
        return tier_values
    brand = tier_values.get(1, "")
    if " / " not in brand:
        return tier_values
    parts = [p.strip() for p in brand.split(" / ") if p.strip()]
    if len(parts) != 2:
        return tier_values
    out = dict(tier_values)
    out[1] = parts[0]
    out[2] = parts[1]
    return out


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
    pref = _finalize_tier_value_dict(_apply_embedded_specialty_tier5(pref_raw))
    inn  = _finalize_tier_value_dict(_apply_embedded_specialty_tier5(inn_raw))
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

    # Capture OON pharmacy "Not covered" BEFORE Pass 1a may rewrite fields.
    oon_rx_original = result.get("Out-of-Network RX") or ""
    oon_pharmacy_not_covered = _is_oon_pharmacy_not_covered(oon_rx_original)

    # Pass 1a: "Not covered" / "null" whole-field -> empty string for RX fields,
    # except consolidated OON pharmacy fields when the entire benefit is Not covered.
    for key, val in result.items():
        if "RX" in key and isinstance(val, str):
            if val.strip().lower() in _NOT_COVERED_PHRASES:
                if oon_pharmacy_not_covered and key in (
                    "Out-of-Network RX",
                    "Out-of-Network Mail Order RX",
                ):
                    result[key] = "Not covered"
                else:
                    result[key] = ""
            # RX Deductible fields: a zero/none deductible means "no RX deductible"
            # -> empty string ("$0", "0", "None", "No deductible" are all noise).
            elif key.endswith("RX Deductible") and val.strip().lower() in (
                "$0", "0", "none", "no deductible", "does not apply"
            ):
                result[key] = ""

    # Pass 1b: HMO/EPO with no OON pharmacy benefit -> clear OON RX fields.
    # When the document explicitly says OON pharmacy is Not covered, preserve that.
    network_type = (result.get("Network Type") or "").strip().lower()
    if any(net in network_type for net in _HMO_EPO_NETWORKS):
        if oon_pharmacy_not_covered:
            for field in ("Out-of-Network RX", "Out-of-Network Mail Order RX"):
                if field in result:
                    result[field] = "Not covered"
            if "Out-of-Network RX Deductible" in result:
                result["Out-of-Network RX Deductible"] = ""
        else:
            for field in _OON_RX_FIELDS:
                if field in result:
                    result[field] = ""
            oon_pharmacy_not_covered = False

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
        if net_prefix == "Out-of-Network" and oon_pharmacy_not_covered:
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
            if _preferred_rx_redundant_with_inn(preferred_rx, consolidated):
                tier_values, _ = _extract_tier_values_with_mail(
                    consolidated, split_retail_mail_pairs=split_pairs,
                )
            else:
                tier_values = _merge_preferred_and_inn_tier_values(
                    preferred_rx, consolidated,
                )
        else:
            tier_values, _explicit_mail = _extract_tier_values_with_mail(
                consolidated, split_retail_mail_pairs=split_pairs,
            )
        if net_prefix == "Out-of-Network":
            tier_values = _fix_oon_tier3_brand_split(
                tier_values,
                consolidated,
                result.get("In-Network RX") or "",
            )
            if not any(tier_values.get(i) for i in range(5)):
                inn_tiers, _ = _extract_tier_values_with_mail(
                    result.get("In-Network RX") or "",
                    split_retail_mail_pairs=False,
                )
                tier_values = _broadcast_flat_oon_pharmacy(
                    consolidated,
                    inn_tiers,
                    result.get("In-Network RX") or "",
                )
        for i, suffix in enumerate(_TIER_SUFFIXES):
            field_name = f"{net_prefix} {suffix}"
            if field_name in output_field_names:
                result[field_name] = tier_values.get(i, "")

    # Pass 2b: drop retail values the VLM wrongly copied into Mail Order.
    # health_3tier mail fields are separate VLM columns — do not dedupe against retail tiers.
    if not designated_tier_fields:
        retail_field_values = [
            result.get(f"{np} {sfx}") or ""
            for np in ("In-Network", "Out-of-Network", "Designated Network")
            for sfx in _TIER_SUFFIXES
        ]
        retail_atoms = _collect_retail_canon_atoms(retail_field_values)
        for mo_field in _MAIL_ORDER_RX_FIELDS:
            if mo_field == "Designated Network Mail Order RX":
                continue
            val = result.get(mo_field)
            if isinstance(val, str) and val:
                kept = [
                    p for p in (part.strip() for part in val.split(" / "))
                    if p and not _should_drop_retail_echo_from_mail(p, retail_atoms)
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
        existing = result.get(mo_field) or ""
        derived = _build_mail_order_from_consolidated(
            consolidated, split_retail_mail_pairs=split_pairs,
        )
        if derived:
            if (
                existing
                and designated_tier_fields
                and net_prefix == "Designated Network"
            ):
                # VLM mail-only column is authoritative when present; consolidated
                # derivation can miss specialty-tier mail or mix in retail costs.
                continue
            if _should_prefer_existing_mail_order(existing, derived):
                continue
            result[mo_field] = derived
        elif designated_tier_fields and net_prefix != "Designated Network":
            # health_3tier INN/OON: clear only empty fields or mail copied from Designated.
            designated_mo = result.get("Designated Network Mail Order RX") or ""
            if not existing:
                result[mo_field] = ""
            elif existing.strip() == designated_mo.strip():
                result[mo_field] = ""
            elif (
                oon_pharmacy_not_covered
                and mo_field == "Out-of-Network Mail Order RX"
            ):
                pass
            # else keep valid network-specific VLM mail order values

    if oon_pharmacy_not_covered:
        if (
            "Out-of-Network RX" in output_field_names
            and _is_not_covered_value(oon_rx_original)
        ):
            result["Out-of-Network RX"] = "Not covered"
        if "Out-of-Network Mail Order RX" in output_field_names:
            mo = result.get("Out-of-Network Mail Order RX") or ""
            if not mo or _is_not_covered_value(mo):
                result["Out-of-Network Mail Order RX"] = "Not covered"

    return result
