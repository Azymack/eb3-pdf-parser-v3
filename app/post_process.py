"""Post-extraction field computation.

Currently no fields are computed from other fields — RX tier fields are now
extracted directly by the VLM rather than combined in post-processing.
This module is retained as an extension point for future computed fields.
"""
from __future__ import annotations

COMPUTED_FIELD_NAMES: frozenset[str] = frozenset()


def vlm_field_names(field_names: list[str]) -> list[str]:
    """Return field_names with any computed fields removed (currently none)."""
    return [f for f in field_names if f not in COMPUTED_FIELD_NAMES]


def apply_post_processing(
    fields: dict[str, str | None],
    output_field_names: list[str],  # noqa: ARG001
) -> dict[str, str | None]:
    """Apply post-VLM computation steps. Currently a pass-through."""
    return fields
