"""Tests for post_process.py."""
from app.post_process import (
    COMPUTED_FIELD_NAMES,
    apply_post_processing,
    vlm_field_names,
)
from app.schemas import CATEGORY_FIELDS

_COMPUTED_TIER_FIELDS = {
    "In-Network Generic RX", "Out-of-Network Generic RX",
    "In-Network Brand RX", "Out-of-Network Brand RX",
    "In-Network Tier 3 RX", "Out-of-Network Tier 3 RX",
    "In-Network Tier 4 RX", "Out-of-Network Tier 4 RX",
    "In-Network Tier 5 RX", "Out-of-Network Tier 5 RX",
}


def test_computed_field_names_contains_tier_fields():
    """Per-tier RX fields are computed — VLM must not be asked to extract them."""
    for field in _COMPUTED_TIER_FIELDS:
        assert field in COMPUTED_FIELD_NAMES, f"Missing from COMPUTED_FIELD_NAMES: {field!r}"


def test_vlm_field_names_excludes_computed():
    """vlm_field_names strips computed fields so VLM never sees them."""
    for category in ("health", "health_3tier"):
        all_fields = CATEGORY_FIELDS[category]
        vlm_fields = vlm_field_names(all_fields)
        for computed in COMPUTED_FIELD_NAMES:
            assert computed not in vlm_fields, (
                f"{category}: computed field {computed!r} was not stripped"
            )
        # Non-computed fields must still be present
        for f in all_fields:
            if f not in COMPUTED_FIELD_NAMES:
                assert f in vlm_fields


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


def test_apply_post_processing_computes_tier_fields():
    """Per-tier fields are derived positionally from the consolidated RX string."""
    fields = {
        "Network Type": "PPO",
        "In-Network RX": "Generic: $10 / Brand: $35 / Tier 3: $75 / Tier 4: 30%",
        "Out-of-Network RX": "Generic: $20 / Brand: $60",
    }
    output_names = CATEGORY_FIELDS["health"]
    result = apply_post_processing(fields, output_names)
    assert result["In-Network Generic RX"] == "$10"
    assert result["In-Network Brand RX"] == "$35"
    assert result["In-Network Tier 3 RX"] == "$75"
    assert result["In-Network Tier 4 RX"] == "30%"
    assert result["In-Network Tier 5 RX"] == ""
    assert result["Out-of-Network Generic RX"] == "$20"
    assert result["Out-of-Network Brand RX"] == "$60"
    assert result["Out-of-Network Tier 3 RX"] == ""


def test_apply_post_processing_does_not_mutate():
    fields = {"Carrier Name": "Acme"}
    original = dict(fields)
    apply_post_processing(fields, CATEGORY_FIELDS["dental"])
    assert fields == original


def test_health_has_consolidated_and_tier_rx_fields():
    """Health schema has both consolidated and per-tier RX fields."""
    health_fields = CATEGORY_FIELDS["health"]
    for f in ("In-Network RX", "Out-of-Network RX",
              "In-Network Mail Order RX", "Out-of-Network Mail Order RX",
              "In-Network Generic RX", "In-Network Brand RX",
              "In-Network Tier 3 RX", "In-Network Tier 4 RX", "In-Network Tier 5 RX"):
        assert f in health_fields, f"health schema missing: {f!r}"
    # Mail-order tier variants must NOT exist (never added back)
    for old in ("In-Network Generic Mail Order RX", "In-Network Brand Mail Order RX",
                "In-Network Tier 3 Mail Order RX", "Designated Network Generic RX"):
        assert old not in health_fields, f"Old field present: {old!r}"


def test_health_3tier_has_new_rx_fields():
    fields = CATEGORY_FIELDS["health_3tier"]
    assert "Designated Network RX" in fields
    assert "In-Network RX" in fields
    assert "Out-of-Network RX" in fields
    assert "Designated Network Mail Order RX" in fields
    # Designated Network tier fields must NOT exist (not added)
    for old in ("Designated Network Generic RX", "Designated Network Brand RX",
                "Designated Network Tier 3 RX", "Designated Network Generic Mail Order RX"):
        assert old not in fields, f"Old field present: {old!r}"


def test_non_rx_categories_unaffected():
    dental = CATEGORY_FIELDS["dental"]
    assert "In-Network Cleanings" in dental
    assert "In-Network RX" not in dental
