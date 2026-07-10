"""Tests for post_process.py."""
from app.post_process import (
    COMPUTED_FIELD_NAMES,
    _build_mail_order_from_consolidated,
    _extract_home_delivery,
    _extract_mail_from_tier_value,
    _extract_retail_only,
    _extract_tier_values,
    _extract_tier_values_with_mail,
    _label_to_tier_index,
    _normalize_tier_label,
    _normalize_tier_retail_value,
    _split_retail_mail,
    _split_unlabeled_retail_mail,
    _strip_tier_labels,
    apply_post_processing,
    vlm_field_names,
)
from app.schemas import CATEGORY_FIELDS

_COMPUTED_TIER_FIELDS = {
    "Designated Network Generic RX", "Designated Network Brand RX",
    "Designated Network Tier 3 RX", "Designated Network Tier 4 RX",
    "Designated Network Tier 5 RX",
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
    """Whole-field Not covered on OON pharmacy is preserved and propagated to tiers."""
    fields = {
        "Network Type": "PPO",
        "Carrier Name": "Acme",
        "In-Network RX": "Tier 1: $10",
        "Out-of-Network RX": "Not covered",
        "Out-of-Network Mail Order RX": "Not Covered",
        "Plan Year": None,
    }
    result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
    assert result["In-Network RX"] == "Tier 1: $10"
    assert result["Out-of-Network RX"] == "Not covered"
    assert result["Out-of-Network Mail Order RX"] == "Not covered"
    assert result["Out-of-Network Generic RX"] == "Not covered"
    assert result["Out-of-Network Brand RX"] == "Not covered"
    assert result["Carrier Name"] == "Acme"


def test_hmo_oon_pharmacy_not_covered_preserved():
    """HMO with explicit OON pharmacy Not covered shows Not covered, not blank."""
    fields = {
        "Network Type": "HMO",
        "In-Network RX": "Generic: $10 / Brand: $40",
        "Out-of-Network RX": "Not covered",
    }
    result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
    assert result["Out-of-Network RX"] == "Not covered"
    assert result["Out-of-Network Generic RX"] == "Not covered"
    assert result["Out-of-Network Brand RX"] == "Not covered"
    assert result["Out-of-Network Mail Order RX"] == "Not covered"


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
                "In-Network Tier 3 Mail Order RX"):
        assert old not in health_fields, f"Old field present: {old!r}"


def test_health_3tier_has_new_rx_fields():
    fields = CATEGORY_FIELDS["health_3tier"]
    assert "Designated Network RX" in fields
    assert "In-Network RX" in fields
    assert "Out-of-Network RX" in fields
    assert "Designated Network Mail Order RX" in fields
    # Designated per-tier fields are computed from Designated Network RX
    for tier in ("Generic RX", "Brand RX", "Tier 3 RX", "Tier 4 RX", "Tier 5 RX"):
        assert f"Designated Network {tier}" in fields
    # Mail-order tier variants must NOT exist (never added back)
    for old in ("Designated Network Generic Mail Order RX",
                "In-Network Generic Mail Order RX"):
        assert old not in fields, f"Old field present: {old!r}"


def test_non_rx_categories_unaffected():
    dental = CATEGORY_FIELDS["dental"]
    assert "In-Network Cleanings" in dental
    assert "In-Network RX" not in dental


# ── Explicit (retail)/(mail order) qualifier handling ────────────────────────

class TestSplitRetailMail:
    def test_uhc_word_channel_per_tier_cell(self):
        cell = "Retail $10 / Mail-Order $20 / Specialty Drugs $10"
        r, m = _split_retail_mail(cell)
        assert r == "$10"
        assert m == "$20"
        assert _extract_retail_only(cell) == "$10"
        assert _extract_home_delivery(cell) == "$20"

    def test_uhc_colon_channel_coinsurance_cell(self):
        cell = (
            "Retail: 50% coinsurance with a $150 copay maximum, deductible does not apply / "
            "Mail-Order: 50% coinsurance with a $300 copay maximum, deductible does not apply / "
            "Specialty Drugs: 50% coinsurance with a $150 copay maximum, deductible does not apply"
        )
        r, m = _split_retail_mail(cell)
        assert r == "50% coinsurance with a $150 copay maximum, deductible does not apply"
        assert m == "50% coinsurance with a $300 copay maximum, deductible does not apply"

    def test_uhc_comma_channel_cell(self):
        cell = "$25 copay, Mail-Order: $50 copay, Specialty Drugs: $25"
        r, m = _split_retail_mail(cell)
        assert r == "$25 copay"
        assert m == "$50 copay"
        assert _extract_retail_only(cell) == "$25 copay"
        assert _extract_home_delivery(cell) == "$50 copay"

    def test_bcbs_supply_generic_retail_and_mail(self):
        cell = (
            "$10 / retail supply or $20 / mail service supply for low-cost generic drugs; "
            "$45 / retail supply or $90 / mail service supply for other generic drugs"
        )
        r, m = _split_retail_mail(cell)
        assert r == "$10 / $45"
        assert m == "$20 / $90"
        assert _extract_retail_only(cell) == "$10 / $45"
        assert _extract_home_delivery(cell) == "$20 / $90"

    def test_bcbs_supply_brand_single_clause(self):
        cell = "$150 / retail supply or $300 / mail service supply"
        r, m = _split_retail_mail(cell)
        assert r == "$150"
        assert m == "$300"

    def test_bcbs_supply_oon_mail_tail_stripped(self):
        cell = "$20 / retail supply for low-cost generic drugs or $90 / retail supply for other generic drugs and all charges for mail service"
        r, m = _split_retail_mail(cell)
        assert r == "$20 / $90"
        assert m is None

    def test_bcbs_supply_not_covered_mail_omitted(self):
        cell = "50% coinsurance / retail supply for preferred brand specialty drugs; not covered / mail service supply"
        r, m = _split_retail_mail(cell)
        assert r == "50% coinsurance"
        assert m is None

    def test_kaiser_pattern_with_prescription_suffix(self):
        r, m = _split_retail_mail("$5 / prescription (retail), $10 / prescription (mail order)")
        assert r == "$5"
        assert m == "$10"

    def test_simple_pattern(self):
        r, m = _split_retail_mail("$15 (retail), $30 (mail order)")
        assert r == "$15"
        assert m == "$30"

    def test_unqualified_value_unchanged(self):
        r, m = _split_retail_mail("10% coinsurance up to $250 / prescription")
        assert r == "10% coinsurance up to $250 / prescription"
        assert m is None

    def test_home_delivery_synonym(self):
        r, m = _split_retail_mail("$15 (retail), $30 (home delivery)")
        assert r == "$15"
        assert m == "$30"

    def test_retail_only_qualifier_stripped_by_normalize(self):
        assert _extract_retail_only("$20 copay (retail only)") == "$20 copay"
        r, m = _split_retail_mail("$20 copay (retail)")
        assert r == "$20 copay (retail)"
        assert m is None


class TestNewlineSeparatedTiers:
    """VLM sometimes separates tiers with newlines; ' / ' inside a line is
    then part of the value (e.g. '$250 / prescription'), not a separator."""

    def test_kaiser_newline_output(self):
        s = ("Generic (Tier 1): $5 / prescription (retail)\n"
             "Preferred Brand (Tier 2): $15 / prescription (retail)\n"
             "Non-preferred Brand (Tier 2): $15 / prescription (retail)\n"
             "Specialty Drugs (Tier 4): 10% coinsurance up to $250 / prescription")
        tv = _extract_tier_values(s)
        assert tv[0] == "$5 / prescription (retail)"
        assert tv[1] == "$15 / prescription (retail)"
        assert tv[2] == "$15 / prescription (retail)"  # Non-preferred Brand → Tier 3
        assert tv[3] == "10% coinsurance up to $250 / prescription"

    def test_with_mail_extraction_from_newline_output(self):
        s = ("Generic (Tier 1): $5 / prescription (retail), $10 / prescription (mail order)\n"
             "Preferred Brand (Tier 2): $15 / prescription (retail), $30 / prescription (mail order)\n"
             "Specialty Drugs (Tier 4): 10% coinsurance up to $250 / prescription")
        retail, mail = _extract_tier_values_with_mail(s)
        assert retail[0] == "$5"
        assert retail[1] == "$15"
        assert retail[3] == "10% coinsurance up to $250 / prescription"
        assert mail == {0: "$10", 1: "$30"}   # specialty has no mail order


class TestExplicitMailOverridesVlmAttribution:
    """When the doc marks values '(retail)/(mail order)', post-processing rebuilds
    In-Network Mail Order RX from the markers, overriding VLM attribution."""

    def test_mail_order_rebuilt_from_qualifiers(self):
        fields = {
            "Network Type": "HMO",
            "In-Network RX": (
                "Generic (Tier 1): $5 / prescription (retail), $10 / prescription (mail order)\n"
                "Preferred Brand (Tier 2): $15 / prescription (retail), $30 / prescription (mail order)\n"
                "Non-preferred Brand (Tier 2): $15 / prescription (retail), $30 / prescription (mail order)\n"
                "Specialty Drugs (Tier 4): 10% coinsurance up to $250 / prescription"
            ),
            # VLM wrongly included specialty's retail-only cost:
            "In-Network Mail Order RX": "$10 / $30 / $30 / 10% coinsurance up to $250",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$5"
        assert result["In-Network Brand RX"] == "$15"
        assert result["In-Network Tier 3 RX"] == "$15"
        assert result["In-Network Tier 4 RX"] == "10% coinsurance up to $250 / prescription"
        assert result["In-Network Mail Order RX"] == "$10 / $30 / $30"  # specialty dropped

    def test_no_qualifiers_leaves_vlm_mail_order_alone(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Generic: $10 / Brand: $40",
            "In-Network Mail Order RX": "$25 / $80",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Mail Order RX"] == "$25 / $80"


class TestBcbsMaSupplyFormat:
    """BCBS MA SBC — '/ retail supply' and '/ mail service supply' per drug row."""

    _INN_RX = (
        "Low-Cost Generic: $10 / retail supply or $20 / mail service supply for low-cost generic drugs; "
        "$45 / retail supply or $90 / mail service supply for other generic drugs / "
        "Preferred Brand: $150 / retail supply or $300 / mail service supply / "
        "Non-Preferred Brand: $225 / retail supply or $675 / mail service supply / "
        "Specialty Drugs: $10 / retail supply for low-cost generic drugs; "
        "$45 / retail supply for other generic drugs; "
        "50% coinsurance / retail supply for preferred brand specialty drugs; "
        "50% coinsurance / retail supply for non-preferred specialty drugs; "
        "not covered / mail service supply"
    )

    _COLLAPSED_INN_RX = (
        "Low-Cost Generic: $10 / Generic: $45 / Preferred Brand: $150 / "
        "Non-Preferred Brand: $225 / Specialty Drugs: Low-Cost Generic: $10 / "
        "Generic: $45 / Preferred: 50% Coinsurance / Non-Preferred: Not Covered"
    )

    def test_full_supply_text_post_process(self):
        result = apply_post_processing(
            {"Network Type": "PPO", "In-Network RX": self._INN_RX},
            CATEGORY_FIELDS["health"],
        )
        assert result["In-Network Generic RX"] == "$10 / $45"
        assert result["In-Network Brand RX"] == "$150"
        assert result["In-Network Tier 3 RX"] == "$225"
        assert result["In-Network Tier 4 RX"] == (
            "$10 / $45 / 50% coinsurance / 50% coinsurance"
        )
        assert result["In-Network Mail Order RX"] == "$20 / $90 / $300 / $675"

    def test_collapsed_vlm_output_post_process(self):
        result = apply_post_processing(
            {
                "Network Type": "PPO",
                "In-Network RX": self._COLLAPSED_INN_RX,
                "In-Network Mail Order RX": "$20 / $90 / $300 / $675",
            },
            CATEGORY_FIELDS["health"],
        )
        assert result["In-Network Generic RX"] == "$10 / $45"
        assert result["In-Network Brand RX"] == "$150"
        assert result["In-Network Tier 3 RX"] == "$225"
        assert result["In-Network Tier 4 RX"] == (
            "$10 / $45 / 50% Coinsurance / Not Covered"
        )
        assert result["In-Network Mail Order RX"] == "$20 / $90 / $300 / $675"


# ── Label-based tier mapping unit tests ──────────────────────────────────────

class TestLabelToTierIndex:
    """_label_to_tier_index must route labels to the correct 0-based tier slot."""

    def test_generic_aliases_to_tier1(self):
        for label in ("Generic", "Generic Drugs", "Tier 1", "Tier 1 - Typically Generic",
                      "Generic (Tier 1)", "Preferred Generic", "Tier 1a"):
            assert _label_to_tier_index(label) == 0, f"Expected Tier1 for {label!r}"

    def test_brand_aliases_to_tier2(self):
        for label in ("Brand", "Brand Name", "Brand Name Drugs", "Preferred Brand",
                      "Tier 2", "Tier 2 - Typically Preferred Brand",
                      "Brand Drugs (Preferred)"):
            assert _label_to_tier_index(label) == 1, f"Expected Tier2 for {label!r}"

    def test_non_preferred_brand_always_tier3(self):
        """Non-Preferred Brand ALWAYS maps to Tier 3 regardless of doc numbering."""
        for label in ("Non-Preferred Brand", "Non-Preferred Brand Drugs",
                      "Non Preferred Brand", "Tier 2 (Non-Preferred Brand)",
                      "Tier 3 (Non-Preferred Brand)", "Non-Preferred Brand Drugs (Tier 2)"):
            assert _label_to_tier_index(label) == 2, f"Expected Tier3 for {label!r}"

    def test_tier3_aliases(self):
        for label in ("Tier 3", "Tier 3 - Typically Non-Preferred Brand"):
            assert _label_to_tier_index(label) == 2, f"Expected Tier3 for {label!r}"

    def test_specialty_aliases_to_tier4(self):
        for label in ("Specialty", "Specialty (Preferred)",
                      "Tier 4", "Preferred Specialty",
                      "Tier 4 - Typically Preferred Specialty",
                      "Typically Preferred Specialty",
                      "Specialty Drugs (Tier 4)"):
            assert _label_to_tier_index(label) == 3, f"Expected Tier4 for {label!r}"

    def test_specialty_drugs_row_to_tier5(self):
        ctx = _normalize_tier_label(
            "Tier 1: $30 / Tier 2: $65 / Tier 3: $100 / Tier 4: $240 / Specialty Drugs"
        )
        assert _label_to_tier_index("Specialty Drugs", ctx) == 4
        assert _label_to_tier_index("Specialty drugs", ctx) == 4
        assert _label_to_tier_index("Specialty Drugs") == 3  # no numbered-tier context

    def test_non_preferred_specialty_to_tier5(self):
        for label in ("Non-Preferred Specialty", "Specialty (Non-Preferred)",
                      "Tier 5", "Non Preferred Specialty"):
            assert _label_to_tier_index(label) == 4, f"Expected Tier5 for {label!r}"

    def test_non_preferred_generic_to_generic_rx(self):
        """Non-Preferred Generic merges into Generic RX (slot 0), not Brand."""
        assert _label_to_tier_index("Non-Preferred Generic") == 0
        assert _label_to_tier_index("Non-Preferred Generic Drugs") == 0
        assert _label_to_tier_index("Generic Drugs (Non-Preferred)") == 0

    def test_brand_non_preferred_suffix_to_tier3(self):
        assert _label_to_tier_index("Brand Drugs (Non-Preferred)") == 2
        assert _label_to_tier_index("Brand drugs (non-preferred)") == 2
        assert _label_to_tier_index("Brand drugs (non-preffered)") == 2
        assert _label_to_tier_index("Brand drugs (nonpreferred)") == 2

    def test_speciality_typo_to_tier5(self):
        assert _label_to_tier_index("Speciality drugs (non-preferred)") == 4
        assert _label_to_tier_index("Speciality drugs (non-preffered)") == 4

    def test_preferred_brand_not_confused_with_non_preferred(self):
        """'Preferred Brand' must NOT match the Non-Preferred Brand check."""
        assert _label_to_tier_index("Preferred Brand") == 1   # Tier 2, not Tier 3

    def test_unknown_label_returns_none(self):
        for label in ("Formulary", "Step Therapy", "PA Required", "Preventive"):
            assert _label_to_tier_index(label) is None, f"Expected None for {label!r}"


class TestExtractTierValues:
    """_extract_tier_values must produce correct {index: value} dicts."""

    def test_standard_4tier(self):
        s = "Generic: $10 / Brand: $40 / Non-Preferred Brand: $75 / Specialty: 30%"
        tv = _extract_tier_values(s)
        assert tv == {0: "$10", 1: "$40", 2: "$75", 3: "30%"}

    def test_non_preferred_brand_skips_brand_slot(self):
        """When doc has Generic + Non-Preferred Brand (no Preferred Brand row),
        Brand slot must be empty — Non-Preferred Brand goes to Tier 3, not Tier 2."""
        s = "Generic: $10 / Non-Preferred Brand: $75 / Specialty: 30%"
        tv = _extract_tier_values(s)
        assert tv.get(0) == "$10"   # Generic → Tier 1
        assert 1 not in tv          # Brand slot empty — no Preferred Brand row
        assert tv.get(2) == "$75"   # Non-Preferred Brand → Tier 3
        assert tv.get(3) == "30%"   # Specialty → Tier 4

    def test_doc_mislabels_non_preferred_brand_as_tier2(self):
        """Carrier labels Non-Preferred Brand as 'Tier 2' in the document.
        VLM returns 'Tier 2 (Non-Preferred Brand)' — must still route to Tier 3.
        The trailing plain 'Tier 3' label also maps to index 2 by our rules but
        loses to the first-match-wins rule, so it is dropped rather than placed
        at an unknown index.  A VLM that returns 'Tier 3 (Specialty)' would be
        handled correctly by the specialty check."""
        s = "Tier 1: $10 / Tier 2 (Non-Preferred Brand): $75 / Tier 3: 30%"
        tv = _extract_tier_values(s)
        assert tv.get(0) == "$10"   # Tier 1 → Generic
        assert 1 not in tv          # Brand slot empty — no Preferred Brand row
        assert tv.get(2) == "$75"   # Non-Preferred Brand → Tier 3
        assert 3 not in tv          # plain "Tier 3" conflicts with index 2 (taken); dropped

    def test_preferred_and_non_preferred_brand(self):
        """Both Brand rows present: Preferred Brand → Tier 2, Non-Preferred Brand → Tier 3."""
        s = "Generic: $10 / Preferred Brand: $40 / Non-Preferred Brand: $75 / Specialty: 30%"
        tv = _extract_tier_values(s)
        assert tv == {0: "$10", 1: "$40", 2: "$75", 3: "30%"}

    def test_5tier_plan(self):
        s = ("Tier 1: $10 / Tier 2: $40 / Tier 3: $75 / "
             "Tier 4: $120 / Non-Preferred Specialty: 50%")
        tv = _extract_tier_values(s)
        assert tv == {0: "$10", 1: "$40", 2: "$75", 3: "$120", 4: "50%"}

    def test_empty_string_returns_empty(self):
        assert _extract_tier_values("") == {}

    def test_first_match_wins_on_duplicate_labels(self):
        """VLM sometimes emits the same tier twice; first occurrence wins."""
        s = "Generic: $10 / Generic: $15 / Brand: $40"
        tv = _extract_tier_values(s)
        assert tv[0] == "$10"   # first Generic wins
        assert tv[1] == "$40"


class TestApplyPostProcessingLabelBased:
    """Integration: apply_post_processing with label-based tier mapping."""

    def test_specialty_drugs_row_below_tier4_routes_to_tier5(self):
        """Wellmark SBC: Specialty Drugs row with Generic/Preferred/Non-Preferred sub-rows."""
        fields = {
            "Network Type": "PPO",
            "In-Network RX": (
                "Tier 1: $30 / Tier 2: $65 / Tier 3: $100 / Tier 4: $240 / "
                "Specialty Drugs: Generic: $190 / Preferred: $275 / Non-Preferred: $325"
            ),
            "Out-of-Network RX": (
                "Tier 1: Not covered / Tier 2: Not covered / Tier 3: Not covered / "
                "Tier 4: Not covered / Specialty Drugs: Not covered"
            ),
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$30"
        assert result["In-Network Brand RX"] == "$65"
        assert result["In-Network Tier 3 RX"] == "$100"
        assert result["In-Network Tier 4 RX"] == "$240"
        assert result["In-Network Tier 5 RX"] == "$190 / $275 / $325"
        assert result["Out-of-Network Tier 5 RX"] == "Not covered"

    def test_embedded_specialty_drugs_in_tier_cells_routes_to_tier5(self):
        """UHC SBC: Specialty Drugs cost parenthetical inside each tier cell."""
        fields = {
            "Network Type": "HMO",
            "In-Network RX": (
                "Tier 1 (Generic): $10 / "
                "Tier 2 (Midrange-Cost): $40 (Specialty Drugs: $40 copay) / "
                "Tier 3 (Midrange-Cost): $75 (Specialty Drugs: $100 copay) / "
                "Tier 4 (Additional High-Cost): $125 (Specialty Drugs: $150 copay)"
            ),
            "Out-of-Network RX": "",
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == "$40"
        assert result["In-Network Tier 3 RX"] == "$75"
        assert result["In-Network Tier 4 RX"] == "$125"
        assert result["In-Network Tier 5 RX"] == "$40 / $100 / $150"

    def test_uhc_specialty_drugs_aggregate_row(self):
        """UHC SBC: separate Specialty Drugs aggregate row after Tier 4."""
        fields = {
            "Network Type": "EPO",
            "In-Network RX": (
                "Tier 1: $10 / Tier 2: $35 / Tier 3: $75 / Tier 4: $250 / "
                "Specialty Drugs: $10 / $150 / $350 / $500"
            ),
            "Out-of-Network RX": "",
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Tier 5 RX"] == "$10 / $150 / $350 / $500"

    def test_uhc_retail_mail_specialty_inline_per_tier_cell(self):
        """UHC SBC: Retail / Mail-Order / Specialty Drugs word labels inside each tier cell."""
        fields = {
            "Network Type": "EPO",
            "In-Network RX": (
                "Tier 1 (Generic): Retail $10 / Mail-Order $25 / Specialty Drugs $10 / "
                "Tier 2 (Mid-Range): Retail $35 / Mail-Order $87.50 / Specialty Drugs $150 / "
                "Tier 3 (Mid-Range): Retail $70 / Mail-Order $175 / Specialty Drugs $350 / "
                "Tier 4 (Highest): Retail $150 / Mail-Order $375 / Specialty Drugs $500"
            ),
            "Out-of-Network RX": "",
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == "$35"
        assert result["In-Network Tier 3 RX"] == "$70"
        assert result["In-Network Tier 4 RX"] == "$150"
        assert result["In-Network Tier 5 RX"] == "$10 / $150 / $350 / $500"
        assert result["In-Network Mail Order RX"] == "$25 / $87.50 / $175 / $375"

    def test_uhc_1772736193_retail_mail_specialty(self):
        """UHC HMO DW46 — Tier 1-4 with Retail/Mail/Specialty per cell (1772736193)."""
        fields = {
            "Network Type": "HMO",
            "In-Network RX": (
                "Tier 1 (Generic): Retail $10 copay / Mail-Order $20 copay / Specialty Drugs $10 copay / "
                "Tier 2 (Midrange-Cost): Retail $40 copay / Mail-Order $80 copay / Specialty Drugs $40 copay / "
                "Tier 3 (Midrange-Cost): Retail $75 copay / Mail-Order $150 copay / Specialty Drugs $100 copay / "
                "Tier 4 (Additional High-Cost): Retail $125 copay / Mail-Order $250 copay / Specialty Drugs $150 copay"
            ),
            "In-Network Mail Order RX": "$20 copay / $80 copay / $150 copay / $250 copay",
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == "$40"
        assert result["In-Network Tier 3 RX"] == "$75"
        assert result["In-Network Tier 4 RX"] == "$125"
        assert result["In-Network Tier 5 RX"] == "$10 / $40 / $100 / $150"
        assert result["In-Network Mail Order RX"] == "$20 / $80 / $150 / $250"

    def test_uhc_boilerplate_only_tier_value_stripped(self):
        """VLM mistake: 'Deductible does not apply' alone must not become Generic RX."""
        fields = {
            "Network Type": "HMO",
            "In-Network RX": (
                "Tier 1 (Generic): Deductible does not apply / "
                "Tier 2 (Midrange-Cost): Retail: $40 copay / "
                "Tier 3 (Midrange-Cost): Retail: $75 copay / "
                "Tier 4 (Additional High-Cost): Retail: $125 copay"
            ),
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == ""
        assert result["In-Network Brand RX"] == "$40"
        assert result["In-Network Tier 3 RX"] == "$75"

    def test_uhc_live_vlm_string_1772736193(self):
        """UHC HMO — live VLM with boilerplate prefix and Specialty Drugs** footnotes."""
        inn = (
            "Tier 1 (Generic): Deductible does not apply. Retail: $10 copay / Mail-Order: $20 copay / "
            "Specialty Drugs**: $10 copay / Tier 2 (Midrange-Cost): Retail: $40 copay / Mail-Order: $80 copay / "
            "Specialty Drugs**: $40 copay / Tier 3 (Midrange-Cost): Retail: $75 copay / Mail-Order: $150 copay / "
            "Specialty Drugs**: $100 copay / Tier 4 (Additional High-Cost): Retail: $125 copay / "
            "Mail-Order: $250 copay / Specialty Drugs**: $150 copay"
        )
        result = apply_post_processing(
            {
                "Network Type": "HMO",
                "In-Network RX": inn,
                "In-Network Mail Order RX": "$20 / $80 / $150 / $250",
            },
            CATEGORY_FIELDS["health"],
        )
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == "$40"
        assert result["In-Network Tier 3 RX"] == "$75"
        assert result["In-Network Tier 4 RX"] == "$125"
        assert result["In-Network Tier 5 RX"] == "$10 / $40 / $100 / $150"

    def test_bcbs_generic_drugs_label_format(self):
        """BCBS VLM may label rows 'Generic drugs (Preferred)' — must not merge all rows."""
        inn = (
            "Generic drugs (Preferred): Retail: Preferred - No Charge Participating - $10/prescription / "
            "Generic drugs (Non-Preferred): Retail: Preferred - $10/prescription Participating - $20/prescription / "
            "Brand drugs (Preferred): Retail: Preferred - $50/prescription Participating - $70/prescription / "
            "Brand drugs (Non-Preferred): Retail: Preferred - $100/prescription Participating - $120/prescription / "
            "Specialty drugs (Preferred): $250/prescription / Specialty drugs (Non-Preferred): $350/prescription"
        )
        result = apply_post_processing(
            {"Network Type": "PPO", "In-Network RX": inn},
            CATEGORY_FIELDS["health"],
        )
        assert result["In-Network Generic RX"] == "No Charge / $10 / $20"
        assert result["In-Network Brand RX"] == "$50 / $70"
        assert result["In-Network Tier 3 RX"] == "$100 / $120"

    def test_uhc_colon_labeled_tier_cells_1772746611(self):
        """UHC NJ SBC — Retail:/Mail-Order:/Specialty Drugs: labels with coinsurance."""
        fields = {
            "Network Type": "EPO",
            "In-Network RX": (
                "Tier 1: Retail: $25 copay, deductible does not apply / "
                "Mail-Order: $50 copay, deductible does not apply / "
                "Specialty Drugs: $25 copay, deductible does not apply / "
                "Tier 2: Retail: 50% coinsurance with a $150 copay maximum, deductible does not apply / "
                "Mail-Order: 50% coinsurance with a $300 copay maximum, deductible does not apply / "
                "Specialty Drugs: 50% coinsurance with a $150 copay maximum, deductible does not apply / "
                "Tier 3: Retail: 50% coinsurance, deductible does not apply / "
                "Mail-Order: 50% coinsurance, deductible does not apply / "
                "Specialty Drugs: 50% coinsurance with a $150 copay maximum, deductible does not apply / "
                "Tier 4: Not Applicable"
            ),
            "In-Network Mail Order RX": "$50 / $300 / $150",
            "Out-of-Network RX": "",
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$25 copay, deductible does not apply"
        assert result["In-Network Brand RX"] == (
            "50% coinsurance with a $150 copay maximum, deductible does not apply"
        )
        assert result["In-Network Tier 3 RX"] == "50% coinsurance, deductible does not apply"
        assert result["In-Network Tier 5 RX"] == (
            "$25 copay, deductible does not apply / "
            "50% coinsurance with a $150 copay maximum, deductible does not apply / "
            "50% coinsurance with a $150 copay maximum, deductible does not apply"
        )
        assert result["In-Network Mail Order RX"] == (
            "$50 copay, deductible does not apply / "
            "50% coinsurance with a $300 copay maximum, deductible does not apply / "
            "50% coinsurance, deductible does not apply"
        )

    def test_uhc_comma_separated_channels_1772746611(self):
        """VLM comma format: '$25 copay, Mail-Order: $50 copay, Specialty Drugs: $25'."""
        fields = {
            "Network Type": "EPO",
            "In-Network RX": (
                "Tier 1: $25 copay, Mail-Order: $50 copay, Specialty Drugs: $25 / "
                "Tier 2: 50% coinsurance with a $150 copay maximum, "
                "Mail-Order: 50% coinsurance with a $300 copay maximum, "
                "Specialty Drugs: 50% coinsurance with a $150 copay maximum / "
                "Tier 3: 50% coinsurance, Mail-Order: 50% coinsurance, "
                "Specialty Drugs: 50% coinsurance with a $150 copay maximum / "
                "Tier 4: Not Applicable"
            ),
            "In-Network Mail Order RX": (
                "$50 copay, Specialty Drugs: $25 / "
                "50% coinsurance with a $300 copay maximum, Specialty Drugs: 50% coinsurance with a $150 copay maximum / "
                "50% coinsurance, Specialty Drugs: 50% coinsurance with a $150 copay maximum"
            ),
            "Out-of-Network RX": "",
            "Preferred Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$25"
        assert result["In-Network Brand RX"] == "50% coinsurance with a $150 copay maximum"
        assert result["In-Network Tier 3 RX"] == "50% coinsurance"
        assert result["In-Network Tier 4 RX"] == "Not Applicable"
        assert result["In-Network Tier 5 RX"] == (
            "$25 / 50% coinsurance with a $150 copay maximum / "
            "50% coinsurance with a $150 copay maximum"
        )
        assert result["In-Network Mail Order RX"] == (
            "$50 / 50% coinsurance with a $300 copay maximum / 50% coinsurance"
        )

    def test_hmo_whole_field_not_covered_27833il0140061(self):
        """27833IL0140061-00 — VLM returns whole-field Not covered on HMO OON pharmacy."""
        fields = {
            "Network Type": "HMO",
            "In-Network RX": (
                "Tier 1a - Preferred: No charge / Tier 1b - Generic Retail: No charge / "
                "Tier 2 - Retail: No charge / Tier 3 - Retail: No charge / "
                "Tier 4 - Retail: No charge"
            ),
            "Out-of-Network RX": "Not covered",
            "Out-of-Network Mail Order RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["Out-of-Network RX"] == "Not covered"
        assert result["Out-of-Network Generic RX"] == "Not covered"
        assert result["Out-of-Network Brand RX"] == "Not covered"
        assert result["Out-of-Network Tier 3 RX"] == "Not covered"
        assert result["Out-of-Network Tier 4 RX"] == "Not covered"
        assert result["Out-of-Network Tier 5 RX"] == "Not covered"
        assert result["Out-of-Network Mail Order RX"] == "Not covered"

    def test_hmo_whole_field_not_covered_41047oh0030057(self):
        """41047OH0030057-00 — same HMO whole-field Not covered layout."""
        fields = {
            "Network Type": "HMO",
            "In-Network RX": "Tier 1a - Preferred Generic Retail: $3 Copay / Tier 2 - Retail: 45% Coinsurance",
            "Out-of-Network RX": "Not covered",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["Out-of-Network RX"] == "Not covered"
        assert result["Out-of-Network Generic RX"] == "Not covered"
        assert result["Out-of-Network Mail Order RX"] == "Not covered"

    def test_ppo_labeled_all_rows_not_covered_1783617094(self):
        """1783617094 — every OON drug row Not covered; all tiers + mail show Not covered."""
        fields = {
            "Network Type": "PPO",
            "In-Network RX": (
                "Generic: No charge after deductible / Preferred Brand: No charge after deductible / "
                "Non-preferred Brand: No charge after deductible / "
                "Specialty Drugs: No charge after deductible"
            ),
            "Out-of-Network RX": (
                "Generic: Not covered / Preferred Brand: Not covered / "
                "Non-preferred Brand: Not covered / Specialty Drugs: Not covered"
            ),
            "Out-of-Network Mail Order RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["Out-of-Network RX"] == (
            "Generic: Not covered / Preferred Brand: Not covered / "
            "Non-preferred Brand: Not covered / Specialty Drugs: Not covered"
        )
        assert result["Out-of-Network Generic RX"] == "Not covered"
        assert result["Out-of-Network Brand RX"] == "Not covered"
        assert result["Out-of-Network Tier 3 RX"] == "Not covered"
        assert result["Out-of-Network Tier 4 RX"] == "Not covered"
        assert result["Out-of-Network Tier 5 RX"] == "Not covered"
        assert result["Out-of-Network Mail Order RX"] == "Not covered"

    def test_aetna_flat_oon_broadcast_1781127953(self):
        """Aetna POS — VLM collapses repeated OON column to one unlabeled coinsurance."""
        inn = (
            "Tier 1A (Preferred Generic): $3 (retail) / Tier 1 (Preferred Generic): $10 (retail) / "
            "Tier 2 (Preferred Brand): $45 (retail) / Tier 3 (Non-preferred Generic/Brand): $75 (retail) / "
            "Specialty Drugs: Preferred: 20% coinsurance for up to a 30 day supply; "
            "Non-preferred: 40% coinsurance for up to a 30 day supply"
        )
        oon_labeled = (
            "Generic: 50% coinsurance deductible does not apply / "
            "Brand: 50% coinsurance deductible does not apply / "
            "Tier 3: 50% coinsurance deductible does not apply"
        )
        oon_flat = "50% coinsurance (retail), deductible does not apply"
        base = {"Network Type": "POS", "In-Network RX": inn}

        labeled = apply_post_processing({**base, "Out-of-Network RX": oon_labeled}, CATEGORY_FIELDS["health"])
        assert labeled["Out-of-Network Generic RX"] == "50% coinsurance deductible does not apply"
        assert labeled["Out-of-Network Brand RX"] == "50% coinsurance deductible does not apply"
        assert labeled["Out-of-Network Tier 3 RX"] == "50% coinsurance deductible does not apply"

        flat = apply_post_processing({**base, "Out-of-Network RX": oon_flat}, CATEGORY_FIELDS["health"])
        assert flat["Out-of-Network Generic RX"] == "50% coinsurance"
        assert flat["Out-of-Network Brand RX"] == "50% coinsurance"
        assert flat["Out-of-Network Tier 3 RX"] == "50% coinsurance"
        assert flat["Out-of-Network Tier 4 RX"] == "Not covered"

    def test_premera_comma_separated_rows_1773085355(self):
        """Premera AWB — simple 4-row table comma-separated on one line."""
        inn = (
            "Generic drugs: $20 copay/prescription (retail), Preferred brand drugs: "
            "$35 copay/prescription (retail), Non-preferred brand drugs: "
            "$55 copay/prescription (retail), Specialty drugs: $150 copay/prescription"
        )
        oon = (
            "Generic drugs: $20 copay/prescription + 40% coinsurance (retail), "
            "Preferred brand drugs: $35 copay/prescription + 40% coinsurance (retail), "
            "Non-preferred brand drugs: $55 copay/prescription + 40% coinsurance (retail), "
            "Specialty drugs: Not covered"
        )
        result = apply_post_processing(
            {
                "Network Type": "PPO",
                "In-Network RX": inn,
                "In-Network Mail Order RX": "$60 / $105 / $165",
                "Out-of-Network RX": oon,
            },
            CATEGORY_FIELDS["health"],
        )
        assert result["In-Network Generic RX"] == "$20"
        assert result["In-Network Brand RX"] == "$35"
        assert result["In-Network Tier 3 RX"] == "$55"
        assert result["In-Network Tier 4 RX"] == "$150"
        assert result["In-Network Mail Order RX"] == "$60 / $105 / $165"
        assert result["Out-of-Network Generic RX"] == "$20 copay/prescription + 40% coinsurance"
        assert result["Out-of-Network Brand RX"] == "$35 copay/prescription + 40% coinsurance"
        assert result["Out-of-Network Tier 3 RX"] == "$55 copay/prescription + 40% coinsurance"
        assert result["Out-of-Network Tier 4 RX"] == "Not covered"

    def test_bcbs_six_row_preferred_non_preferred_1772810659(self):
        """BCBS OK MOBAP0105 — 6-row pref/non-pref, dual retail columns per row."""
        inn = (
            "Generic (Preferred): No Charge / Generic (Non-Preferred): $10 / $20 / "
            "Brand (Preferred): $50 / $70 / Brand (Non-Preferred): $100 / $120 / "
            "Specialty (Preferred): $250 / Specialty (Non-Preferred): $350"
        )
        oon = (
            "Generic (Preferred): $10 / Generic (Non-Preferred): $20 / "
            "Brand (Preferred): $70 / Brand (Non-Preferred): $120 / "
            "Specialty (Preferred): $250 / Specialty (Non-Preferred): $350"
        )
        fields = {
            "Network Type": "PPO",
            "In-Network RX": inn,
            "Preferred Network RX": inn,
            "Out-of-Network RX": oon,
            "In-Network Mail Order RX": "$0 / $10 / $20 / $70 / $120 / $250 / $350",
            "Out-of-Network Mail Order RX": "$10 / $20 / $70 / $120 / $250 / $350",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "No Charge / $10 / $20"
        assert result["In-Network Brand RX"] == "$50 / $70"
        assert result["In-Network Tier 3 RX"] == "$100 / $120"
        assert result["In-Network Tier 4 RX"] == "$250"
        assert result["In-Network Tier 5 RX"] == "$350"
        assert result["Out-of-Network Generic RX"] == "$10 / $20"
        assert result["Out-of-Network Brand RX"] == "$70"
        assert result["Out-of-Network Tier 3 RX"] == "$120"
        assert result["In-Network Mail Order RX"] == ""
        assert result["Out-of-Network Mail Order RX"] == ""

    def test_bcbs_preferred_participating_format_1772810659(self):
        """BCBS OK — PDF uses Preferred - / Participating - dual retail columns."""
        inn = (
            "Generic (Preferred): Preferred - No Charge Participating - $10/prescription / "
            "Generic (Non-Preferred): Preferred - $10/prescription Participating - $20/prescription / "
            "Brand (Preferred): Preferred - $50/prescription Participating - $70/prescription / "
            "Brand (Non-Preferred): Preferred - $100/prescription Participating - $120/prescription / "
            "Specialty (Preferred): $250/prescription / Specialty (Non-Preferred): $350/prescription"
        )
        oon = (
            "Generic (Preferred): $10/prescription / Generic (Non-Preferred): $20/prescription / "
            "Brand (Preferred): $70/prescription / Brand (Non-Preferred): $120/prescription / "
            "Specialty (Preferred): $250/prescription / Specialty (Non-Preferred): $350/prescription"
        )
        fields = {
            "Network Type": "PPO",
            "In-Network RX": inn,
            "Preferred Network RX": "",
            "Out-of-Network RX": oon,
            "In-Network Mail Order RX": "No Charge / $30 / $150 / $300",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "No Charge / $10 / $20"
        assert result["In-Network Brand RX"] == "$50 / $70"
        assert result["In-Network Tier 3 RX"] == "$100 / $120"
        assert result["Out-of-Network Tier 4 RX"] == "$250"
        assert result["Out-of-Network Tier 5 RX"] == "$350"
        assert result["In-Network Mail Order RX"] == "$30 / $150 / $300"

    def test_oon_tier3_brand_continuation_split_1772810659(self):
        """OON: VLM appends non-pref brand cost to Brand (Preferred) without label."""
        inn = (
            "Generic (Preferred): Preferred - No Charge Participating - $10/prescription / "
            "Generic (Non-Preferred): Preferred - $10/prescription Participating - $20/prescription / "
            "Brand (Preferred): Preferred - $50/prescription Participating - $70/prescription / "
            "Brand (Non-Preferred): Preferred - $100/prescription Participating - $120/prescription / "
            "Specialty (Preferred): $250/prescription / Specialty (Non-Preferred): $350/prescription"
        )
        oon = (
            "Generic (Preferred): $10/prescription / Generic (Non-Preferred): $20/prescription / "
            "Brand (Preferred): $70/prescription / $120/prescription / "
            "Specialty (Preferred): $250/prescription / Specialty (Non-Preferred): $350/prescription"
        )
        fields = {
            "Network Type": "PPO",
            "In-Network RX": inn,
            "Out-of-Network RX": oon,
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["Out-of-Network Brand RX"] == "$70"
        assert result["Out-of-Network Tier 3 RX"] == "$120"

    def test_oon_tier3_plain_brand_continuation_split(self):
        """OON: plain Brand label with $70 / $120 continuation when INN is 6-row."""
        inn = (
            "Generic (Preferred): No Charge / Generic (Non-Preferred): $10 / $20 / "
            "Brand (Preferred): $50 / $70 / Brand (Non-Preferred): $100 / $120"
        )
        oon = "Generic: $10/prescription / $20/prescription / Brand: $70/prescription / $120/prescription"
        result = apply_post_processing(
            {"Network Type": "PPO", "In-Network RX": inn, "Out-of-Network RX": oon},
            CATEGORY_FIELDS["health"],
        )
        assert result["Out-of-Network Brand RX"] == "$70"
        assert result["Out-of-Network Tier 3 RX"] == "$120"

    def test_non_preferred_brand_routes_to_tier3_not_brand(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Generic: $10 / Non-Preferred Brand: $75 / Specialty: 30%",
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == ""       # no Preferred Brand row
        assert result["In-Network Tier 3 RX"] == "$75"   # Non-Preferred Brand → Tier 3
        assert result["In-Network Tier 4 RX"] == "30%"   # Specialty → Tier 4

    def test_carrier_mislabeled_tier2_as_non_preferred_brand(self):
        """Carrier used 'Tier 2 (Non-Preferred Brand)' — must map to Tier 3."""
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Tier 1 (Generic): $10 / Tier 2 (Non-Preferred Brand): $60 / Tier 3 (Specialty): 25%",
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Generic RX"] == "$10"
        assert result["In-Network Brand RX"] == ""        # no Preferred Brand
        assert result["In-Network Tier 3 RX"] == "$60"    # Non-Preferred Brand → Tier 3
        assert result["In-Network Tier 4 RX"] == "25%"    # Specialty → Tier 4


# ── Mail Order RX label stripping ────────────────────────────────────────────

class TestStripTierLabels:
    """_strip_tier_labels must return cost values only, no tier labels."""

    def test_strips_standard_labels(self):
        assert _strip_tier_labels("Tier 1: $20 / Tier 2: $80 / Tier 3: $130") == "$20 / $80 / $130"

    def test_strips_descriptive_labels(self):
        assert _strip_tier_labels(
            "Tier 1 (Generic): $20 / Tier 2 (Preferred Brand): $80"
        ) == "$20 / $80"

    def test_already_cost_only_unchanged(self):
        assert _strip_tier_labels("$20 / $80 / $130") == "$20 / $80 / $130"

    def test_empty_string_unchanged(self):
        assert _strip_tier_labels("") == ""

    def test_single_tier(self):
        assert _strip_tier_labels("Specialty: 30% up to $150") == "30% up to $150"

    def test_semicolon_separator_normalised(self):
        assert _strip_tier_labels("Tier 1: $20; Tier 2: $80") == "$20 / $80"

    def test_mixed_labelled_and_plain(self):
        """If some parts have labels and some don't, handle both."""
        assert _strip_tier_labels("Tier 1: $20 / $80") == "$20 / $80"


class TestMailOrderStrippedByPostProcessing:
    """apply_post_processing must strip labels from Mail Order RX fields (Pass 3)."""

    def test_mail_order_labels_stripped(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Tier 1 (Generic): $10 / Tier 2 (Brand): $40",
            "In-Network Mail Order RX": "Tier 1: $25 / Tier 2: $80",
            "Out-of-Network Mail Order RX": "Tier 1 (Generic): $30 / Tier 2 (Brand): $90",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Mail Order RX"] == "$25 / $80"
        assert result["Out-of-Network Mail Order RX"] == "$30 / $90"

    def test_retail_rx_labels_preserved(self):
        """Retail RX (In-Network RX) must keep its tier labels — only Mail Order is stripped."""
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Tier 1 (Generic): $10 / Tier 2 (Brand): $40",
            "In-Network Mail Order RX": "Tier 1: $25 / Tier 2: $80",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network RX"] == "Tier 1 (Generic): $10 / Tier 2 (Brand): $40"

    def test_empty_mail_order_unchanged(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Generic: $10",
            "In-Network Mail Order RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Mail Order RX"] == ""

    def test_already_cost_only_mail_order_unchanged(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": "Generic: $10 / Brand: $40",
            "In-Network Mail Order RX": "$25 / $80",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Mail Order RX"] == "$25 / $80"


class TestSplitPreferredParticipating:
    def test_bcbs_dual_column_cell(self):
        from app.post_process import _split_preferred_participating_value
        cell = (
            "Preferred: Preferred - No Charge Participating - $10/prescription / "
            "Non preferred: Preferred - $10/prescription Participating - $20/prescription"
        )
        assert _split_preferred_participating_value(cell) == (
            "No Charge / $10/prescription / $10/prescription / $20/prescription"
        )

class TestSixTierPreferredNonPreferredSplit:
    """BCBS TX 1769090950 — 6-row preferred/non-preferred pharmacy table."""

    _INN_RX = (
        "Generic drugs (preferred): Retail - Preferred Participating - 10% Coinsurance "
        "Participating - 20% coinsurance / "
        "Generic drugs (non-preferred): Retail - Preferred Participating - 10% Coinsurance "
        "Participating - 20% coinsurance / "
        "Brand drugs (preferred): Retail - Preferred Participating - 20% Coinsurance "
        "Participating - 30% coinsurance / "
        "Brand drugs (non-preferred): Retail - Preferred Participating - 30% Coinsurance "
        "Participating - 40% coinsurance / "
        "Specialty drugs (preferred): 40% coinsurance / "
        "Specialty drugs (non-preferred): 50% coinsurance"
    )

    def test_tier3_gets_non_preferred_brand(self):
        tv = _extract_tier_values(self._INN_RX)
        assert "30% Coinsurance Participating - 40% coinsurance" in tv[2]
        assert tv[1] == (
            "Retail - Preferred Participating - 20% Coinsurance Participating - 30% coinsurance"
        )

    def test_generic_merges_preferred_and_non_preferred(self):
        tv = _extract_tier_values(self._INN_RX)
        assert "10% Coinsurance" in tv[0]
        assert "20% coinsurance" in tv[0]
        # Identical preferred/non-preferred costs collapse to one value
        assert tv[0] == (
            "Retail - Preferred Participating - 10% Coinsurance Participating - 20% coinsurance"
        )

    def test_generic_merge_keeps_both_when_different(self):
        s = (
            "Generic drugs (preferred): $10 / "
            "Generic drugs (non-preferred): $20"
        )
        tv = _extract_tier_values(s)
        assert tv[0] == "$10 / $20"

    def test_full_post_process_bcbs_six_tier(self):
        fields = {
            "Network Type": "PPO",
            "In-Network RX": self._INN_RX,
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health"])
        assert result["In-Network Tier 3 RX"] == (
            "Retail - Preferred Participating - 30% Coinsurance Participating - 40% coinsurance"
        )
        assert result["In-Network Brand RX"] == (
            "Retail - Preferred Participating - 20% Coinsurance Participating - 30% coinsurance"
        )
        assert result["In-Network Tier 4 RX"] == "40% coinsurance"
        assert result["In-Network Tier 5 RX"] == "50% coinsurance"
        assert result["In-Network Generic RX"] == (
            "Retail - Preferred Participating - 10% Coinsurance Participating - 20% coinsurance"
        )

    def test_plain_labels_with_preferred_context(self):
        """When preferred rows exist, plain Brand/Specialty rows represent non-preferred."""
        s = (
            "Preferred Generic: $10 / Generic: $35 / "
            "Preferred Brand: 25% Coinsurance / Brand: 50% Coinsurance / "
            "Preferred Specialty: 20% Coinsurance / Specialty: 50% Coinsurance"
        )
        tv = _extract_tier_values(s)
        assert tv[0] == "$10 / $35"
        assert tv[1] == "25% Coinsurance"
        assert tv[2] == "50% Coinsurance"
        assert tv[3] == "20% Coinsurance"
        assert tv[4] == "50% Coinsurance"


class TestChannelSplitting:
    """Unified retail / mail-order channel splitting for all network columns."""

    def test_anthem_retail_and_home_delivery(self):
        cell = "$5 copay (retail) and $10 copay (home delivery)"
        assert _normalize_tier_retail_value(cell) == "$5"
        assert _extract_mail_from_tier_value(cell) == "$10"
        assert _build_mail_order_from_consolidated(
            f"Generic: {cell} / Brand: $50 copay (retail) and $125 copay (home delivery)"
        ) == "$10 / $125"

    def test_retail_and_home_delivery_same_rate(self):
        cell = "30% coinsurance (retail and home delivery)"
        assert _normalize_tier_retail_value(cell) == "30% coinsurance"
        assert _extract_mail_from_tier_value(cell) == "30% coinsurance"

    def test_newline_retail_only_cleaned_in_per_tier(self):
        s = ("Generic (Tier 1): $5 / prescription (retail)\n"
             "Preferred Brand (Tier 2): $15 / prescription (retail)\n"
             "Specialty Drugs (Tier 4): 10% coinsurance up to $250 / prescription")
        retail, mail = _extract_tier_values_with_mail(s)
        assert retail[0] == "$5"
        assert retail[1] == "$15"
        assert mail == {}

    def test_unlabeled_retail_mail_pair_on_designated(self):
        assert _split_unlabeled_retail_mail("$5 / $10") == ("$5", "$10")
        assert _split_unlabeled_retail_mail("$50 / $125") == ("$50", "$125")
        # Cross-network retail merge must NOT be split as retail/mail
        assert _normalize_tier_retail_value("$80 / $90") == "$80 / $90"
        assert _normalize_tier_retail_value("$5 / $10", split_retail_mail_pairs=True) == "$5"

    def test_vlm_unlabeled_pairs_live_output_pattern(self):
        """Regression: VLM puts retail/mail as '$5 / $10' in Designated Network RX."""
        fields = {
            "Network Type": "HMO",
            "Designated Network RX": (
                "Generic: $5 / $10 / Brand: $20 / $50 / "
                "Tier 3: $50 / $125 / "
                "Tier 4: 30% coinsurance up to $250/prescription"
            ),
            "In-Network RX": (
                "Generic: $15/prescription, deductible does not apply / "
                "Brand: $30/prescription, deductible does not apply / "
                "Tier 3: $60/prescription, deductible does not apply / "
                "Tier 4: 40% coinsurance up to $250/prescription"
            ),
            "Out-of-Network RX": "",
            # VLM wrongly copies designated mail into In-Network mail order
            "In-Network Mail Order RX": "$10 / $50 / $125",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Generic RX"] == "$5"
        assert result["Designated Network Brand RX"] == "$20"
        assert result["Designated Network Tier 3 RX"] == "$50"
        assert result["Designated Network Tier 4 RX"] == (
            "30% coinsurance up to $250/prescription"
        )
        assert result["In-Network Generic RX"] == "$15"
        assert result["Designated Network Mail Order RX"] == "$10 / $50 / $125"
        assert result["In-Network Mail Order RX"] == ""

    def test_designated_column_derives_mail_order_pass4(self):
        fields = {
            "Network Type": "HMO",
            "Designated Network RX": (
                "Generic: $5 copay (retail) and $10 copay (home delivery) / "
                "Brand: $50 copay (retail) and $125 copay (home delivery) / "
                "Tier 3: $50 copay (retail) and $125 copay (home delivery) / "
                "Tier 4: 30% coinsurance (retail and home delivery)"
            ),
            "In-Network RX": (
                "Generic: $15/prescription, deductible does not apply / "
                "Brand: $30/prescription, deductible does not apply"
            ),
            "Out-of-Network RX": "",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Generic RX"] == "$5"
        assert result["Designated Network Mail Order RX"] == (
            "$10 / $125 / $125 / 30% coinsurance"
        )
        assert result["In-Network Generic RX"] == "$15"


class TestHealth3TierDesignatedTierFields:
    """health_3tier: each network column feeds its own per-tier fields with stripped costs."""

    def test_anthem_platinum_select_hmo_pattern(self):
        """Regression for 1768608307 — Designated vs In-Network columns differ per tier."""
        fields = {
            "Network Type": "HMO",
            "Designated Network RX": (
                "Generic: $5/prescription, deductible does not apply / "
                "Brand: $50/prescription, deductible does not apply / "
                "Tier 3: $50/prescription, deductible does not apply / "
                "Tier 4: 30% coinsurance up to $250/prescription"
            ),
            "In-Network RX": (
                "Generic: $15/prescription, deductible does not apply / "
                "Brand: $30/prescription, deductible does not apply / "
                "Tier 3: $60/prescription, deductible does not apply / "
                "Tier 4: 40% coinsurance up to $250/prescription"
            ),
            "Out-of-Network RX": "",
            "Designated Network Mail Order RX": (
                "$10 / $50 / $125 / 30% coinsurance up to $250/prescription"
            ),
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Generic RX"] == "$5"
        assert result["In-Network Generic RX"] == "$15"
        assert result["Designated Network Brand RX"] == "$50"
        assert result["In-Network Brand RX"] == "$30"
        assert result["Designated Network Tier 3 RX"] == "$50"
        assert result["In-Network Tier 3 RX"] == "$60"
        assert result["Designated Network Tier 4 RX"] == "30% coinsurance up to $250/prescription"
        assert result["In-Network Tier 4 RX"] == "40% coinsurance up to $250/prescription"
        assert result["Designated Network Mail Order RX"] == (
            "$10 / $50 / $125 / 30% coinsurance up to $250/prescription"
        )
        assert result["Out-of-Network Generic RX"] == ""

    def test_anthem_gold_select_1780511663(self):
        """1780511663 — Anthem Typically (Tier N) rows; Brand RX + per-network mail."""
        des = (
            "Typically Generic (Tier 1): $10 (retail) / $20 (home delivery) / "
            "Typically Preferred Brand & Non-Preferred Generic Drugs (Tier 2): $50 (retail) / $60 (home delivery) / "
            "Typically Non-Preferred Brand and Generic drugs (Tier 3): $90 (retail) / $225 (home delivery) / "
            "Typically Preferred Specialty (brand and generic) (Tier 4): 30% coinsurance up to $250 (retail and home delivery)"
        )
        inn = (
            "Typically Generic (Tier 1): $20 (retail) / "
            "Typically Preferred Brand & Non-Preferred Generic Drugs (Tier 2): $60 (retail) / "
            "Typically Non-Preferred Brand and Generic drugs (Tier 3): $100 (retail) / "
            "Typically Preferred Specialty (brand and generic) (Tier 4): 40% coinsurance up to $250 (retail)"
        )
        fields = {
            "Network Type": "PPO",
            "Designated Network RX": des,
            "In-Network RX": inn,
            "Out-of-Network RX": "Not covered",
            "Designated Network Mail Order RX": "$20 / $60 / $225 / 30% coinsurance up to $250",
            "In-Network Mail Order RX": "$20 / $60 / $100 / 40% coinsurance up to $250",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Generic RX"] == "$10"
        assert result["Designated Network Brand RX"] == "$50"
        assert result["Designated Network Tier 3 RX"] == "$90"
        assert result["Designated Network Tier 4 RX"] == "30% coinsurance up to $250"
        assert result["In-Network Generic RX"] == "$20"
        assert result["In-Network Brand RX"] == "$60"
        assert result["In-Network Tier 3 RX"] == "$100"
        assert result["In-Network Tier 4 RX"] == "40% coinsurance up to $250"
        assert result["Designated Network Mail Order RX"] == (
            "$20 / $60 / $225 / 30% coinsurance up to $250"
        )
        assert result["In-Network Mail Order RX"] == (
            "$20 / $60 / $100 / 40% coinsurance up to $250"
        )
        assert result["Out-of-Network Generic RX"] == "Not covered"
        assert result["Out-of-Network Mail Order RX"] == "Not covered"

    def test_anthem_gold_select_1780511663_tier_prefix_labels(self):
        """1780511663 live VLM — labels as 'Tier N (Typically X)' instead of trailing (Tier N)."""
        des = (
            "Tier 1 (Typically Generic): $10 (retail) / $20 (home delivery) / "
            "Tier 2 (Typically Preferred Brand & Non-Preferred Generic Drugs): $50 (retail) / $60 (home delivery) / "
            "Tier 3 (Typically Non-Preferred Brand and Generic drugs): $90 (retail) / $225 (home delivery) / "
            "Tier 4 (Typically Preferred Specialty): 30% coinsurance up to $250 (retail and home delivery)"
        )
        inn = (
            "Tier 1 (Typically Generic): $20 (retail) / "
            "Tier 2 (Typically Preferred Brand & Non-Preferred Generic Drugs): $60 (retail) / "
            "Tier 3 (Typically Non-Preferred Brand and Generic drugs): $100 (retail) / "
            "Tier 4 (Typically Preferred Specialty): 40% coinsurance up to $250 (retail)"
        )
        fields = {
            "Network Type": "PPO",
            "Designated Network RX": des,
            "In-Network RX": inn,
            "Out-of-Network RX": "Not covered",
            "Designated Network Mail Order RX": "$20 / $60 / $225 / 30% coinsurance up to $250",
            "In-Network Mail Order RX": "$20 / $60 / $100 / 40% coinsurance up to $250",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Brand RX"] == "$50"
        assert result["Designated Network Tier 3 RX"] == "$90"
        assert result["In-Network Brand RX"] == "$60"
        assert result["In-Network Tier 3 RX"] == "$100"
        assert result["In-Network Mail Order RX"] == (
            "$20 / $60 / $100 / 40% coinsurance up to $250"
        )

    def test_designated_does_not_merge_into_inn_per_tier(self):
        """When designated per-tier fields exist, In-Network per-tier uses In-Network RX only."""
        fields = {
            "Network Type": "PPO",
            "Preferred Network RX": "Generic: $10 / Brand: $40",
            "Designated Network RX": "Generic: $5 / Brand: $50",
            "In-Network RX": "Generic: $15 / Brand: $30",
        }
        result = apply_post_processing(fields, CATEGORY_FIELDS["health_3tier"])
        assert result["Designated Network Generic RX"] == "$5"
        assert result["In-Network Generic RX"] == "$15"   # not "$5 / $15"
        assert result["Designated Network Brand RX"] == "$50"
        assert result["In-Network Brand RX"] == "$30"


class TestStripRxSuffix:
    def test_flat_copay_per_prescription_shorthand(self):
        from app.post_process import _strip_rx_suffix
        assert _strip_rx_suffix("$5/prescription, deductible does not apply") == "$5"
        assert _strip_rx_suffix("30% coinsurance up to $250/prescription") == (
            "30% coinsurance up to $250/prescription"
        )


class TestThreeColumnRxContinuation:
    """_extract_tier_values continuation logic for 3-column pharmacy tables.

    A 3-column table (Preferred Network | In-Network | Out-of-Network) is
    ingested by the VLM as 'Label: pref_cost / inn_cost' per tier, i.e. one
    labeled part followed by one unlabeled part.  The continuation logic must
    append the unlabeled cost to the previous tier's value.
    """

    def test_basic_two_tier_continuation(self):
        """Each tier gets two costs merged by ' / '."""
        tv = _extract_tier_values(
            "Tier 1 (Generic): $20 / $20 / Tier 2 (Preferred Brand): $80 / $90"
        )
        assert tv[0] == "$20 / $20"
        assert tv[1] == "$80 / $90"

    def test_four_tier_anthem_style(self):
        """Full Anthem Bronze PPO pattern: 4 tiers, each with two in-network costs."""
        tv = _extract_tier_values(
            "Tier 1 (Generic): $20 / $20 / "
            "Tier 2 (Preferred Brand): $80 / $90 / "
            "Tier 3 (Non-Preferred Brand): $120 / $130 / "
            "Tier 4 (Specialty): 30% up to $400 / 40% up to $500"
        )
        assert tv[0] == "$20 / $20"       # Generic
        assert tv[1] == "$80 / $90"       # Preferred Brand → Tier 2
        assert tv[2] == "$120 / $130"     # Non-Preferred Brand → always Tier 3
        assert tv[3] == "30% up to $400 / 40% up to $500"  # Specialty → Tier 4

    def test_continuation_stops_at_next_label(self):
        """Unlabeled part is only appended to the immediately preceding mapped tier."""
        tv = _extract_tier_values(
            "Generic: $20 / $20 / Brand: $80 / $90 / Non-Preferred Brand: $120"
        )
        assert tv[0] == "$20 / $20"
        assert tv[1] == "$80 / $90"
        assert tv[2] == "$120"            # Non-Preferred Brand — no trailing unlabeled part

    def test_continuation_does_not_add_to_duplicate_label(self):
        """When a tier slot is already occupied the duplicate is dropped and
        continuation tracking resets, so the subsequent unlabeled part is ignored."""
        tv = _extract_tier_values(
            "Generic: $10 / Generic: $15 / $20"
        )
        assert tv[0] == "$10"    # first Generic wins
        assert len(tv) == 1      # duplicate dropped; $20 has no target

    def test_unrecognized_label_resets_continuation(self):
        """An unrecognized label clears the continuation anchor.
        The unlabeled part after it must NOT be appended to the previous tier."""
        tv = _extract_tier_values(
            "Generic: $10 / Unknown Drug Class: $50 / $99 / Brand: $40"
        )
        assert tv[0] == "$10"
        assert tv[1] == "$40"
        assert len(tv) == 2      # $50 and $99 both discarded

    def test_florida_blue_subrow_mode(self):
        """Florida Blue pattern: tiers with program sub-rows (Preventive /
        Condition Care Rx / All Other X).  Preventive is skipped; the other
        sub-row costs are appended to the tier, sub-labels dropped."""
        tv = _extract_tier_values(
            "Generic: Preventive: No Charge / Condition Care Rx: $4 / "
            "All Other Generic: $20 / "
            "Preferred Brand: Condition Care Rx: $50 / "
            "All Other Preferred Brand: Deductible + $100 / "
            "Non-preferred Brand: Deductible + $300 / "
            "Specialty Drugs: Deductible + $500"
        )
        assert tv[0] == "$4 / $20"
        assert tv[1] == "$50 / Deductible + $100"
        assert tv[2] == "Deductible + $300"
        assert tv[3] == "Deductible + $500"

    def test_tier_number_with_unrecognized_descriptor_falls_back_to_number(self):
        """'Tier 3 (Non-preferred)' — descriptor alone maps nothing, so the
        leading tier number decides the slot."""
        assert _label_to_tier_index("Tier 3 (Non-preferred)") == 2
        assert _label_to_tier_index("Tier 4 (Non-preferred)") == 3

    def test_all_other_prefix_stripped(self):
        assert _label_to_tier_index("All Other Generic") == 0
        assert _label_to_tier_index("All Other Preferred Brand") == 1

    def test_preventive_only_tier_dropped(self):
        """A tier whose only sub-row is Preventive ends up absent, not empty."""
        tv = _extract_tier_values("Generic: Preventive: No Charge / Brand: $40")
        assert 0 not in tv
        assert tv[1] == "$40"

    def test_no_continuation_in_normal_two_column_table(self):
        """Standard 2-column table: each tier has exactly one cost — no trailing
        unlabeled parts, so continuation logic is inert."""
        tv = _extract_tier_values(
            "Tier 1 (Generic): $10 / Tier 2 (Brand): $40 / "
            "Tier 3 (Non-Preferred Brand): $65 / Tier 4 (Specialty): 50% up to $150"
        )
        assert tv[0] == "$10"
        assert tv[1] == "$40"
        assert tv[2] == "$65"
        assert tv[3] == "50% up to $150"
