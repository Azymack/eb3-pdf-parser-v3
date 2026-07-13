"""Unit tests for the structured RX extractor (no server / VLM required).

    python -m pytest tests/test_rx_extractor.py -v
"""
import pytest

from app.rx_extractor import (
    RX_EXTRACTOR_CATEGORIES,
    _build_rx_schema,
    assemble_rx_fields,
    rx_owned_fields,
)
from app.page_router import select_rx_pages
from app.schemas import CATEGORY_FIELDS


def _row(label, tier, **kw):
    base = {
        "label": label,
        "standard_tier": tier,
        "in_network_retail": None,
        "in_network_mail_order": None,
        "preferred_pharmacy_retail": None,
        "preferred_pharmacy_mail_order": None,
        "out_of_network_retail": None,
        "out_of_network_mail_order": None,
    }
    base.update(kw)
    return base


def _data(rows, oon="not_shown", **kw):
    base = {
        "drug_rows": rows,
        "mail_order_service": True,
        "out_of_network_pharmacy": oon,
        "rx_deductible_in_network": None,
        "rx_deductible_out_of_network": None,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Field ownership / schema wiring
# ---------------------------------------------------------------------------

def test_owned_fields_are_subset_of_category_fields():
    for category in RX_EXTRACTOR_CATEGORIES:
        owned = set(rx_owned_fields(category))
        missing = owned - set(CATEGORY_FIELDS[category])
        assert not missing, f"{category}: rx fields not in category_keys: {missing}"


def test_schema_builds_for_both_categories():
    for category in RX_EXTRACTOR_CATEGORIES:
        schema = _build_rx_schema(category)
        row_props = schema["properties"]["drug_rows"]["items"]["properties"]
        assert "label" in row_props and "standard_tier" in row_props
    h3 = _build_rx_schema("health_3tier")
    assert "designated_network_retail" in h3["properties"]["drug_rows"]["items"]["properties"]
    assert "rx_deductible_designated_network" in h3["properties"]


# ---------------------------------------------------------------------------
# Tier slot mapping (semantic classification comes from the VLM)
# ---------------------------------------------------------------------------

def test_level_labels_map_by_standard_tier():
    """WPE 'Level 1-4' vocabulary — regexes could never map this."""
    fields = assemble_rx_fields("health", _data([
        _row("Level 1: Preferred generic drugs", "generic",
             in_network_retail="$5/prescription"),
        _row("Level 2: Preferred brand drugs", "preferred_brand",
             in_network_retail="20% coinsurance ($50 max)"),
        _row("Level 3: Non-preferred brand name", "non_preferred_brand",
             in_network_retail="40% coinsurance ($150 max)"),
        _row("Level 4: Specialty drugs", "preferred_specialty",
             in_network_retail="$50 copay"),
    ]))
    assert fields["In-Network Generic RX"] == "$5"
    assert fields["In-Network Brand RX"] == "20% coinsurance ($50 max)"
    assert fields["In-Network Tier 3 RX"] == "40% coinsurance ($150 max)"
    assert fields["In-Network Tier 4 RX"] == "$50"
    assert fields["In-Network Tier 5 RX"] == ""
    assert "Level 1" in fields["In-Network RX"]


def test_multiple_rows_same_slot_join_with_slash():
    fields = assemble_rx_fields("health", _data([
        _row("Preferred Generic", "generic", in_network_retail="$4"),
        _row("Non-preferred Generic", "generic", in_network_retail="$20"),
    ]))
    assert fields["In-Network Generic RX"] == "$4 / $20"


def test_duplicate_values_in_slot_are_collapsed():
    fields = assemble_rx_fields("health", _data([
        _row("Preferred Generic", "generic", in_network_retail="$20"),
        _row("Non-preferred Generic", "generic", in_network_retail="$20"),
    ]))
    assert fields["In-Network Generic RX"] == "$20"


def test_other_rows_are_skipped():
    fields = assemble_rx_fields("health", _data([
        _row("Preventive drugs", "other", in_network_retail="No charge"),
        _row("Generic", "generic", in_network_retail="$10"),
    ]))
    assert fields["In-Network Generic RX"] == "$10"
    assert "Preventive" not in fields["In-Network RX"]


# ---------------------------------------------------------------------------
# Mail order assembly
# ---------------------------------------------------------------------------

def test_mail_order_joined_cost_only_in_tier_order():
    fields = assemble_rx_fields("health", _data([
        _row("Specialty", "preferred_specialty", in_network_retail="50%"),
        _row("Generic", "generic", in_network_retail="$10",
             in_network_mail_order="$20"),
        _row("Brand", "preferred_brand", in_network_retail="$45",
             in_network_mail_order="$90"),
    ]))
    # tier order (generic, brand, ...) regardless of document order
    assert fields["In-Network Mail Order RX"] == "$20 / $90"


def test_mail_order_not_copied_from_retail_only_rows():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10"),
    ]))
    assert fields["In-Network Mail Order RX"] == ""


def test_oon_mail_order_not_covered_when_all_rows_say_so():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10",
             in_network_mail_order="$25",
             out_of_network_retail="50% coinsurance",
             out_of_network_mail_order="Not covered"),
    ], oon="covered"))
    assert fields["Out-of-Network Mail Order RX"] == "Not covered"
    assert fields["Out-of-Network Generic RX"] == "50% coinsurance"


# ---------------------------------------------------------------------------
# Out-of-network "Not covered" propagation
# ---------------------------------------------------------------------------

def test_oon_not_covered_enum_propagates():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10",
             out_of_network_retail="Not covered"),
    ], oon="not_covered"))
    assert fields["Out-of-Network RX"] == "Not covered"
    assert fields["Out-of-Network Generic RX"] == "Not covered"
    assert fields["Out-of-Network Brand RX"] == "Not covered"
    assert fields["Out-of-Network Mail Order RX"] == "Not covered"
    assert fields["Out-of-Network RX Deductible"] == ""


def test_oon_not_covered_from_rows_without_enum():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10",
             out_of_network_retail="Not covered"),
        _row("Brand", "preferred_brand", in_network_retail="$45",
             out_of_network_retail="Not covered"),
    ], oon="covered"))
    assert fields["Out-of-Network RX"] == "Not covered"
    assert fields["Out-of-Network Generic RX"] == "Not covered"


def test_oon_emergency_only_leaves_fields_empty():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10"),
    ], oon="emergency_or_reimbursement_only"))
    assert fields["Out-of-Network RX"] == ""
    assert fields["Out-of-Network Generic RX"] == ""


def test_oon_covered_costs_pass_through():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10",
             out_of_network_retail="50% coinsurance, deductible does not apply"),
    ], oon="covered"))
    assert fields["Out-of-Network Generic RX"] == (
        "50% coinsurance, deductible does not apply"
    )


# ---------------------------------------------------------------------------
# Preferred pharmacy column (health only)
# ---------------------------------------------------------------------------

def test_preferred_pharmacy_merges_into_in_network_tier():
    fields = assemble_rx_fields("health", _data([
        _row("Tier 1 (Typically Generic)", "generic",
             in_network_retail="$20", in_network_mail_order=None,
             preferred_pharmacy_retail="$10",
             preferred_pharmacy_mail_order="$25"),
    ]))
    assert fields["In-Network Generic RX"] == "$10 / $20"
    assert fields["Preferred Network RX"] == "Tier 1 (Typically Generic): $10"
    # home-delivery price from the preferred column feeds In-Network Mail Order
    assert fields["In-Network Mail Order RX"] == "$25"


def test_preferred_pharmacy_equal_value_not_duplicated():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic",
             in_network_retail="$10", preferred_pharmacy_retail="$10"),
    ]))
    assert fields["In-Network Generic RX"] == "$10"


# ---------------------------------------------------------------------------
# Deductible normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise", ["$0", "0", "None", "No deductible",
                                   "does not apply", "n/a", "null"])
def test_deductible_noise_cleared(noise):
    fields = assemble_rx_fields("health", _data(
        [], rx_deductible_in_network=noise,
    ))
    assert fields["In-Network RX Deductible"] == ""


def test_deductible_real_value_kept():
    fields = assemble_rx_fields("health", _data(
        [], rx_deductible_in_network="$250/person or $500/family",
    ))
    assert fields["In-Network RX Deductible"] == "$250/person or $500/family"


# ---------------------------------------------------------------------------
# health_3tier
# ---------------------------------------------------------------------------

def test_3tier_designated_network_fields():
    row = {
        "label": "Generic", "standard_tier": "generic",
        "designated_network_retail": "$5", "designated_network_mail_order": "$10",
        "in_network_retail": "$15", "in_network_mail_order": "$30",
        "out_of_network_retail": None, "out_of_network_mail_order": None,
    }
    fields = assemble_rx_fields("health_3tier", {
        "drug_rows": [row],
        "mail_order_service": True,
        "out_of_network_pharmacy": "not_shown",
        "rx_deductible_designated_network": None,
        "rx_deductible_in_network": None,
        "rx_deductible_out_of_network": None,
    })
    assert fields["Designated Network Generic RX"] == "$5"
    assert fields["Designated Network Mail Order RX"] == "$10"
    assert fields["In-Network Generic RX"] == "$15"
    assert fields["In-Network Mail Order RX"] == "$30"


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_implausible_computed_mail_coinsurance_dropped():
    """45% x 2.5 = 112.5% is a computed multiplier, never a printed price."""
    fields = assemble_rx_fields("health", _data([
        _row("Tier 2", "preferred_brand", in_network_retail="45% Coinsurance",
             in_network_mail_order="112.5% Coinsurance"),
        _row("Generic", "generic", in_network_retail="$3",
             in_network_mail_order="$7.50"),
    ]))
    assert fields["In-Network Mail Order RX"] == "$7.50"


def test_narrative_oon_cell_dropped():
    fields = assemble_rx_fields("health", _data([
        _row("Level 1", "generic", in_network_retail="$5",
             out_of_network_retail=(
                 "Prescriptions may be filled at an out-of-network pharmacy "
                 "in emergency situations only. You should pay for the "
                 "prescription in full and submit a reimbursement form."
             )),
    ], oon="emergency_or_reimbursement_only"))
    assert fields["Out-of-Network Generic RX"] == ""


def test_oon_enum_not_covered_does_not_override_printed_costs():
    fields = assemble_rx_fields("health", _data([
        _row("Generic", "generic", in_network_retail="$10",
             out_of_network_retail="50% coinsurance"),
    ], oon="not_covered"))
    assert fields["Out-of-Network Generic RX"] == "50% coinsurance"


def test_medical_deductible_echo_suppressed():
    from app.rx_extractor import suppress_medical_deductible_echo
    rx = {"In-Network RX Deductible": "$4,750 Individual / $9,500 Family",
          "Out-of-Network RX Deductible": ""}
    medical = {"In-Network Single Deductible": "$4,750",
               "In-Network Family Deductible": "$9,500"}
    suppress_medical_deductible_echo(rx, medical)
    assert rx["In-Network RX Deductible"] == ""


def test_true_rx_deductible_not_suppressed():
    from app.rx_extractor import suppress_medical_deductible_echo
    rx = {"In-Network RX Deductible": "$250/person or $500/family"}
    medical = {"In-Network Single Deductible": "$1,000",
               "In-Network Family Deductible": "$2,000"}
    suppress_medical_deductible_echo(rx, medical)
    assert rx["In-Network RX Deductible"] == "$250/person or $500/family"


def test_deductible_explanation_mentioning_rx_amount_does_not_suppress():
    """1780702082: 'Deductible Explanation' quotes the $200 RX deductible —
    that must not count as a medical amount."""
    from app.rx_extractor import suppress_medical_deductible_echo
    rx = {"In-Network RX Deductible": "$200"}
    medical = {
        "In-Network Single Deductible": "$6,350",
        "In-Network Family Deductible": "$12,700",
        "Deductible Explanation": (
            "There are separate deductibles for prescription drugs ($200) "
            "and infertility treatment ($500)."
        ),
    }
    suppress_medical_deductible_echo(rx, medical)
    assert rx["In-Network RX Deductible"] == "$200"


def test_deductible_description_without_amount_cleared():
    fields = assemble_rx_fields("health", _data(
        [], rx_deductible_in_network=(
            "Subject to combined medical and prescription drug deductible "
            "(waived for preferred and non-preferred brand insulin)"
        ),
    ))
    assert fields["In-Network RX Deductible"] == ""


# ---------------------------------------------------------------------------
# RX page routing
# ---------------------------------------------------------------------------

def test_select_rx_pages_prefers_pharmacy_table_page():
    pages = [
        {"page_number": 1, "markdown": "Deductible overview and copay info"},
        {"page_number": 2, "markdown": (
            "Pharmacy Benefit Retail Medications Tier 1: $15 copay "
            "Tier 2: $30 copay generic brand specialty prescription "
            "Mail Order Medications Tier 1: $30 copay"
        )},
        {"page_number": 3, "markdown": "Exclusions and definitions"},
    ]
    assert 2 in select_rx_pages(pages)


def test_select_rx_pages_empty_when_nothing_matches():
    pages = [{"page_number": 1, "markdown": "dental cleanings and x-rays"}]
    assert select_rx_pages(pages) == []
