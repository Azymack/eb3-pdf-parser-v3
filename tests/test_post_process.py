"""Tests for post_process.py.

The module is currently a pass-through (no computed fields). These tests
confirm that it doesn't mutate or drop fields and that vlm_field_names
returns the full list unchanged.
"""
from app.post_process import (
    COMPUTED_FIELD_NAMES,
    apply_post_processing,
    vlm_field_names,
)
from app.schemas import CATEGORY_FIELDS


def test_computed_field_names_is_empty():
    """No fields are computed — VLM extracts RX tiers directly now."""
    assert COMPUTED_FIELD_NAMES == frozenset()


def test_vlm_field_names_returns_full_list():
    for category in ("health", "health_3tier", "dental", "vision"):
        fields = CATEGORY_FIELDS[category]
        assert vlm_field_names(fields) == fields


def test_apply_post_processing_normalizes_not_covered():
    """'Not covered' whole-field values in RX fields become empty string."""
    fields = {
        "Carrier Name": "Acme",
        "In-Network RX": "Tier 1: $10",
        "Out-of-Network RX": "Not covered",
        "Out-of-Network Mail Order RX": "Not Covered",
        "Plan Year": None,
    }
    result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
    assert result["In-Network RX"] == "Tier 1: $10"
    assert result["Out-of-Network RX"] == ""
    assert result["Out-of-Network Mail Order RX"] == ""
    assert result["Carrier Name"] == "Acme"


def test_apply_post_processing_clears_oon_rx_for_hmo():
    """HMO plans have OON RX fields cleared regardless of VLM output."""
    fields = {
        "Network Type": "HMO",
        "In-Network RX": "Tier 1: $10 / Tier 2: $35",
        "Out-of-Network RX": "Tier 1: $10 / Tier 2: $35",
        "Out-of-Network Mail Order RX": "some value",
        "Out-of-Network RX Deductible": "$500",
    }
    result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
    assert result["In-Network RX"] == "Tier 1: $10 / Tier 2: $35"
    assert result["Out-of-Network RX"] == ""
    assert result["Out-of-Network Mail Order RX"] == ""
    assert result["Out-of-Network RX Deductible"] == ""


def test_apply_post_processing_does_not_mutate():
    fields = {"Carrier Name": "Acme"}
    original = dict(fields)
    apply_post_processing(fields, CATEGORY_FIELDS["dental"])
    assert fields == original


def test_health_has_new_rx_fields():
    """New consolidated RX fields are present; old per-tier fields are gone."""
    health_fields = CATEGORY_FIELDS["health"]
    assert "In-Network RX" in health_fields
    assert "Out-of-Network RX" in health_fields
    assert "In-Network Mail Order RX" in health_fields
    assert "Out-of-Network Mail Order RX" in health_fields
    # Old per-tier fields must be gone
    for old in (
        "In-Network Generic RX", "In-Network Brand RX",
        "In-Network Tier 3 RX", "In-Network Tier 4 RX", "In-Network Tier 5 RX",
        "In-Network Generic Mail Order RX", "In-Network Brand Mail Order RX",
        "In-Network Tier 3 Mail Order RX", "In-Network Tier 4 Mail Order RX",
        "In-Network Tier 5 Mail Order RX",
    ):
        assert old not in health_fields, f"Old field still present: {old!r}"


def test_health_3tier_has_new_rx_fields():
    fields = CATEGORY_FIELDS["health_3tier"]
    assert "Designated Network RX" in fields
    assert "In-Network RX" in fields
    assert "Out-of-Network RX" in fields
    assert "Designated Network Mail Order RX" in fields
    for old in (
        "Designated Network Generic RX", "Designated Network Brand RX",
        "Designated Network Tier 3 RX", "Designated Network Tier 4 RX",
        "Designated Network Tier 5 RX",
        "Designated Network Generic Mail Order RX",
    ):
        assert old not in fields, f"Old field still present: {old!r}"


def test_non_rx_categories_unaffected():
    """Dental/vision/etc. schema is unchanged."""
    dental = CATEGORY_FIELDS["dental"]
    assert "In-Network Cleanings" in dental
    assert "In-Network RX" not in dental
