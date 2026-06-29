"""Post-extraction field normalization.

RX tier fields are extracted directly by the VLM. This module normalizes
VLM output to enforce schema invariants the prompt alone cannot guarantee:

  1. "Not covered" whole-field values → empty string.
     The VLM sometimes writes "Not covered" as the entire field value for
     OON fields instead of leaving them empty as instructed.

  2. Out-of-Network RX fields → empty for HMO/EPO plans.
     These plans have no out-of-network drug benefit. When the VLM
     hallucinates OON values (often copying In-Network data), this rule
     clears them.
"""
from __future__ import annotations

COMPUTED_FIELD_NAMES: frozenset[str] = frozenset()

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


def vlm_field_names(field_names: list[str]) -> list[str]:
    """Return field_names with any computed fields removed (currently none)."""
    return [f for f in field_names if f not in COMPUTED_FIELD_NAMES]


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],  # noqa: ARG001
) -> dict[str, str | None]:
    """Normalize RX field values after VLM extraction."""
    result = dict(fields)

    # Rule 1: "Not covered" / "null" whole-field → empty string for any RX field.
    for key, val in result.items():
        if "RX" in key and isinstance(val, str):
            if val.strip().lower() in _NOT_COVERED_PHRASES:
                result[key] = ""

    # Rule 2: HMO/EPO plans have no OON drug benefit — clear OON RX fields.
    network_type = (result.get("Network Type") or "").strip().lower()
    if any(net in network_type for net in _HMO_EPO_NETWORKS):
        for field in _OON_RX_FIELDS:
            if field in result:
                result[field] = ""

    return result
