"""Tests for post_process.py."""
from app.post_process import (
    COMPUTED_FIELD_NAMES,
    _extract_tier_values,
    _extract_tier_values_with_mail,
    _label_to_tier_index,
    _split_retail_mail,
    _strip_tier_labels,
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


# ── Explicit (retail)/(mail order) qualifier handling ────────────────────────

class TestSplitRetailMail:
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

    def test_retail_only_qualifier_unchanged(self):
        """A lone '(retail)' without a mail-order counterpart is not split."""
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
        for label in ("Specialty", "Specialty Drugs", "Specialty (Preferred)",
                      "Tier 4", "Preferred Specialty",
                      "Tier 4 - Typically Preferred Specialty",
                      "Typically Preferred Specialty"):
            assert _label_to_tier_index(label) == 3, f"Expected Tier4 for {label!r}"

    def test_non_preferred_specialty_to_tier5(self):
        for label in ("Non-Preferred Specialty", "Specialty (Non-Preferred)",
                      "Tier 5", "Non Preferred Specialty"):
            assert _label_to_tier_index(label) == 4, f"Expected Tier5 for {label!r}"

    def test_non_preferred_generic_to_tier2(self):
        """Non-Preferred Generic maps to Tier 2 (Brand slot) in split-generic plans."""
        assert _label_to_tier_index("Non-Preferred Generic") == 1
        assert _label_to_tier_index("Non-Preferred Generic Drugs") == 1

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
