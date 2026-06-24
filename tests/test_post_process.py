"""Unit tests for post-extraction field computation (Mail Order RX combining).

Expected combined values are derived directly from the fixture files in
tests/documents/health/*.json — see health/1, health/2, and health/3 for
non-empty cases, and health/4 / health/5 for the all-empty case.
"""
import pytest

from app.post_process import (
    COMPUTED_FIELD_NAMES,
    apply_post_processing,
    compute_mail_order_fields,
    vlm_field_names,
)
from app.schemas import CATEGORY_FIELDS

HEALTH_FIELDS = CATEGORY_FIELDS["health"]
HEALTH_3TIER_FIELDS = CATEGORY_FIELDS["health_3tier"]


# ---------------------------------------------------------------------------
# compute_mail_order_fields — combination logic
# ---------------------------------------------------------------------------


def test_three_tier_plan_health1_fixture():
    """Generic + Brand + Tier3 filled, Tier4 and Tier5 absent.

    Expected combined: "$20 / $90 / $130" — matches health/1 fixture.
    """
    fields = {
        "In-Network Generic Mail Order RX": "$20",
        "In-Network Brand Mail Order RX": "$90",
        "In-Network Tier 3 Mail Order RX": "$130",
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == "$20 / $90 / $130"
    assert result["Out-of-Network Mail Order RX"] == ""


def test_three_tier_plan_health2_fixture():
    """health/2 fixture pattern: Generic=$10, Brand=$30, Tier3=$30."""
    fields = {
        "In-Network Generic Mail Order RX": "$10",
        "In-Network Brand Mail Order RX": "$30",
        "In-Network Tier 3 Mail Order RX": "$30",
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == "$10 / $30 / $30"
    assert result["Out-of-Network Mail Order RX"] == ""


def test_five_tier_plan_health3_fixture():
    """health/3 fixture pattern: all 5 tiers non-empty.

    Expected combined matches health/3 fixture exactly.
    """
    fields = {
        "In-Network Generic Mail Order RX": "$30",
        "In-Network Brand Mail Order RX": "Deductible, then $100",
        "In-Network Tier 3 Mail Order RX": "Deductible, then $150",
        "In-Network Tier 4 Mail Order RX": "Deductible, then 50% coinsurance up to $200 maximum",
        "In-Network Tier 5 Mail Order RX": "Deductible, then 50% coinsurance up to $300 maximum",
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == (
        "$30 / Deductible, then $100 / Deductible, then $150 / "
        "Deductible, then 50% coinsurance up to $200 maximum / "
        "Deductible, then 50% coinsurance up to $300 maximum"
    )
    assert result["Out-of-Network Mail Order RX"] == ""


def test_all_tiers_absent_health4_5_fixture():
    """health/4 and health/5 fixture pattern: all tier values None → combined is ''."""
    fields = {f"In-Network {s}": None for s in [
        "Generic Mail Order RX", "Brand Mail Order RX", "Tier 3 Mail Order RX",
        "Tier 4 Mail Order RX", "Tier 5 Mail Order RX",
    ]}
    fields.update({f"Out-of-Network {s}": None for s in [
        "Generic Mail Order RX", "Brand Mail Order RX", "Tier 3 Mail Order RX",
        "Tier 4 Mail Order RX", "Tier 5 Mail Order RX",
    ]})
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == ""
    assert result["Out-of-Network Mail Order RX"] == ""


def test_not_found_tiers_are_skipped():
    """NOT_FOUND values in individual tiers are excluded from the combined string.

    This avoids producing misleading strings like "$10 / NOT_FOUND / $30".
    """
    fields = {
        "In-Network Generic Mail Order RX": "$10",
        "In-Network Brand Mail Order RX": "NOT_FOUND",
        "In-Network Tier 3 Mail Order RX": "$50",
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == "$10 / $50"


def test_empty_string_tiers_are_skipped():
    """Empty-string tier values (post null→'' serialization) are skipped."""
    fields = {
        "In-Network Generic Mail Order RX": "$15",
        "In-Network Brand Mail Order RX": "",
        "In-Network Tier 3 Mail Order RX": "",
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == "$15"


# ---------------------------------------------------------------------------
# health_3tier — Designated Network prefix
# ---------------------------------------------------------------------------


def test_health_3tier_designated_network_combined():
    """Designated Network Mail Order RX is computed for health_3tier category."""
    fields = {
        "Designated Network Generic Mail Order RX": "$5",
        "Designated Network Brand Mail Order RX": "$25",
        "Designated Network Tier 3 Mail Order RX": None,
        "Designated Network Tier 4 Mail Order RX": None,
        "Designated Network Tier 5 Mail Order RX": None,
        "In-Network Generic Mail Order RX": "$10",
        "In-Network Brand Mail Order RX": "$30",
        "In-Network Tier 3 Mail Order RX": None,
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = compute_mail_order_fields(fields, HEALTH_3TIER_FIELDS)
    assert result["Designated Network Mail Order RX"] == "$5 / $25"
    assert result["In-Network Mail Order RX"] == "$10 / $30"
    assert result["Out-of-Network Mail Order RX"] == ""


# ---------------------------------------------------------------------------
# vlm_field_names — computed fields excluded from VLM prompt
# ---------------------------------------------------------------------------


def test_vlm_field_names_excludes_computed_health():
    filtered = vlm_field_names(HEALTH_FIELDS)
    for field in COMPUTED_FIELD_NAMES:
        assert field not in filtered, f"Computed field '{field}' should not be in VLM prompt"
    # Individual tier fields must still be present
    assert "In-Network Generic Mail Order RX" in filtered
    assert "In-Network Brand Mail Order RX" in filtered
    assert "In-Network Tier 5 Mail Order RX" in filtered


def test_vlm_field_names_excludes_computed_health_3tier():
    filtered = vlm_field_names(HEALTH_3TIER_FIELDS)
    assert "In-Network Mail Order RX" not in filtered
    assert "Out-of-Network Mail Order RX" not in filtered
    assert "Designated Network Mail Order RX" not in filtered
    assert "Designated Network Generic Mail Order RX" in filtered


def test_vlm_field_names_non_mail_order_category_unchanged():
    """Categories without mail order fields are returned as-is."""
    dental_fields = CATEGORY_FIELDS["dental"]
    assert vlm_field_names(dental_fields) == dental_fields


# ---------------------------------------------------------------------------
# apply_post_processing — integration
# ---------------------------------------------------------------------------


def test_apply_post_processing_adds_combined_to_health():
    fields: dict = {
        "Carrier Name": "Acme",
        "In-Network Generic Mail Order RX": "$10",
        "In-Network Brand Mail Order RX": "$40",
        "In-Network Tier 3 Mail Order RX": None,
        "In-Network Tier 4 Mail Order RX": None,
        "In-Network Tier 5 Mail Order RX": None,
        "Out-of-Network Generic Mail Order RX": None,
        "Out-of-Network Brand Mail Order RX": None,
        "Out-of-Network Tier 3 Mail Order RX": None,
        "Out-of-Network Tier 4 Mail Order RX": None,
        "Out-of-Network Tier 5 Mail Order RX": None,
    }
    result = apply_post_processing(fields, HEALTH_FIELDS)
    assert result["In-Network Mail Order RX"] == "$10 / $40"
    assert result["Out-of-Network Mail Order RX"] == ""
    assert result["Carrier Name"] == "Acme"  # unchanged


def test_apply_post_processing_does_not_mutate_input():
    fields: dict = {"Carrier Name": "Acme"}
    apply_post_processing(fields, HEALTH_FIELDS)
    assert "In-Network Mail Order RX" not in fields


def test_apply_post_processing_dental_passthrough():
    """Dental category has no mail order fields — fields returned unchanged."""
    dental_fields = CATEGORY_FIELDS["dental"]
    fields = {"Carrier Name": "Acme", "Plan Name": "Gold"}
    result = apply_post_processing(fields, dental_fields)
    assert result == {"Carrier Name": "Acme", "Plan Name": "Gold"}
