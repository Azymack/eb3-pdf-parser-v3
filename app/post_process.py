"""Post-extraction field computation.

Some output fields are derived from other extracted fields rather than read
directly from the document. This module handles those computations so the VLM
is never asked to produce a field that can be fully derived from fields it
already extracts — which would risk the two values disagreeing.
"""
from __future__ import annotations

# Combined Mail Order RX combined fields are computed by joining the individual
# per-tier mail order values. Asking the VLM to both read individual tier values
# AND separately produce a "combined" summary risks the two disagreeing; the
# combined value is fully derivable from the tier data we already extract.
_MAIL_ORDER_COMBINED_SUFFIX = "Mail Order RX"
_MAIL_ORDER_TIER_SUFFIXES: tuple[str, ...] = (
    "Generic Mail Order RX",
    "Brand Mail Order RX",
    "Tier 3 Mail Order RX",
    "Tier 4 Mail Order RX",
    "Tier 5 Mail Order RX",
)

_NETWORK_PREFIXES: tuple[str, ...] = (
    "In-Network",
    "Out-of-Network",
    "Designated Network",  # health_3tier only
)

# The combined Mail Order RX keys that are computed (not VLM-extracted).
COMPUTED_FIELD_NAMES: frozenset[str] = frozenset(
    f"{prefix} {_MAIL_ORDER_COMBINED_SUFFIX}" for prefix in _NETWORK_PREFIXES
)


def vlm_field_names(field_names: list[str]) -> list[str]:
    """Return field_names with computed fields removed for use in the VLM prompt."""
    return [f for f in field_names if f not in COMPUTED_FIELD_NAMES]


def _is_absent(v: str | None) -> bool:
    # Treat null, empty string, and NOT_FOUND as "no value to contribute".
    # NOT_FOUND is skipped because we cannot distinguish "tier not offered"
    # from "tier exists but wasn't located" at combination time; including it
    # would produce misleading strings like "$10 / NOT_FOUND / $30".
    return v is None or v == "" or v == "NOT_FOUND"


def compute_mail_order_fields(
    fields: dict[str, str | None],
    output_field_names: list[str],
) -> dict[str, str]:
    """Compute combined Mail Order RX fields from individual per-tier values.

    Format: non-absent tier values joined by " / " in tier order
    (Generic → Brand → Tier 3 → Tier 4 → Tier 5).
    If all tiers are absent the combined value is "" (field not on this plan).

    Only computes combined fields that appear in output_field_names, so
    categories without mail order fields are unaffected.
    """
    result: dict[str, str] = {}
    output_set = set(output_field_names)
    for prefix in _NETWORK_PREFIXES:
        combined_key = f"{prefix} {_MAIL_ORDER_COMBINED_SUFFIX}"
        if combined_key not in output_set:
            continue
        parts = [
            v
            for suffix in _MAIL_ORDER_TIER_SUFFIXES
            if not _is_absent(v := fields.get(f"{prefix} {suffix}"))
        ]
        result[combined_key] = " / ".join(parts)
    return result


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],
) -> dict[str, str | None]:
    """Apply all post-VLM computation steps.

    Returns a new fields dict augmented with any computed fields.
    The input dict is not mutated.
    """
    computed = compute_mail_order_fields(fields, output_field_names)
    if not computed:
        return fields
    return {**fields, **computed}
