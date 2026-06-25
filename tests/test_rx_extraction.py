"""Live investigation tests for the redesigned RX tier extraction.

Run against a real server + VLM:
    python -m pytest tests/test_rx_extraction.py -v -s --no-header

These tests verify:
  1. The new 'In-Network RX' field is populated and contains multiple tiers.
  2. Tier labels use the carrier's own words (not forced canonical names).
  3. Retail and mail order values are correctly separated.
  4. Edge cases: Tier 1a/1b (health/4), 5-tier HMO (health/3), no-mail-order plan (health/4, health/5).
  5. Old per-tier field names are NOT present in responses.

Because these hit real services they are skipped unless --live is passed:
    python -m pytest tests/test_rx_extraction.py -v -s --live

Alternatively set the env var RUN_LIVE_RX_TESTS=1.
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

DOCS_DIR = Path("tests/documents/health")
API_URL = "http://localhost:8002/extract_json_v2"
HEADERS = {"X-EB3-Token": "eb3-key-1"}

LIVE = os.getenv("RUN_LIVE_RX_TESTS") == "1"
live = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_RX_TESTS=1 to run live VLM tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(pdf_name: str, include_metadata: bool = False):
    import httpx
    pdf_path = DOCS_DIR / pdf_name
    params = {"include_metadata": "true"} if include_metadata else {}
    resp = httpx.post(
        API_URL,
        headers=HEADERS,
        files={"file": (pdf_name, pdf_path.read_bytes(), "application/pdf")},
        data={"category": "health"},
        params=params,
        timeout=180,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def _tier_count(rx_string: str) -> int:
    """Count the number of ' / ' separated tiers in an RX field value."""
    if not rx_string:
        return 0
    return len([p for p in rx_string.split(" / ") if p.strip()])


def _has_cost(rx_string: str) -> bool:
    """True if the string contains at least one recognisable cost value."""
    return bool(re.search(r"\$\d|%|\bno charge\b|\bdeductible\b|\bcopay\b|\bvaries\b",
                          rx_string, re.IGNORECASE))


OLD_FIELDS = {
    "In-Network Generic RX", "Out-of-Network Generic RX",
    "In-Network Brand RX", "Out-of-Network Brand RX",
    "In-Network Tier 3 RX", "Out-of-Network Tier 3 RX",
    "In-Network Tier 4 RX", "Out-of-Network Tier 4 RX",
    "In-Network Tier 5 RX", "Out-of-Network Tier 5 RX",
    "In-Network Generic Mail Order RX", "Out-of-Network Generic Mail Order RX",
    "In-Network Brand Mail Order RX", "Out-of-Network Brand Mail Order RX",
    "In-Network Tier 3 Mail Order RX", "Out-of-Network Tier 3 Mail Order RX",
    "In-Network Tier 4 Mail Order RX", "Out-of-Network Tier 4 Mail Order RX",
    "In-Network Tier 5 Mail Order RX", "Out-of-Network Tier 5 Mail Order RX",
}


# ---------------------------------------------------------------------------
# Structural tests (no VLM, use schema only)
# ---------------------------------------------------------------------------

def test_schema_has_new_rx_fields():
    from app.schemas import CATEGORY_FIELDS
    health = CATEGORY_FIELDS["health"]
    assert "In-Network RX" in health
    assert "Out-of-Network RX" in health
    assert "In-Network Mail Order RX" in health
    assert "Out-of-Network Mail Order RX" in health
    for old in OLD_FIELDS:
        assert old not in health, f"Schema still has old field: {old!r}"


def test_fixtures_have_new_rx_fields():
    """All health fixture files use the new schema."""
    for num in range(1, 6):
        path = DOCS_DIR / f"{num}.json"
        data = json.loads(path.read_text())
        assert "In-Network RX" in data, f"health/{num}.json missing In-Network RX"
        assert "In-Network Mail Order RX" in data, f"health/{num}.json missing Mail Order RX"
        for old in OLD_FIELDS:
            assert old not in data, f"health/{num}.json still has old field {old!r}"


def test_fixture_rx_tier_counts():
    """Each fixture's RX field contains a sensible number of tiers."""
    expected_min_tiers = {
        "1.json": 4,  # Kaiser: T1/T2/T3/T4
        "2.json": 3,  # Kaiser Platinum: T1 + T2 preferred + T2 non-preferred + T4 = 4 effective
        "3.json": 5,  # BlueChoice: Generic + PrefBrand + NonprefBrand + PrefSpec + NonprefSpec
        "4.json": 4,  # BCBS AZ: T1a + T1b + T2 + T3 + Specialty = 5 effective
        "5.json": 3,  # Geisinger: T1 + T2 + T3
    }
    for fname, min_tiers in expected_min_tiers.items():
        data = json.loads((DOCS_DIR / fname).read_text())
        rx = data["In-Network RX"]
        count = _tier_count(rx)
        assert count >= min_tiers, (
            f"health/{fname} In-Network RX has {count} tiers, expected >= {min_tiers}: {rx!r}"
        )


def test_fixture_health4_has_tier_1a_1b():
    """health/4 fixture must preserve BCBS Arizona's split Tier 1a / 1b."""
    data = json.loads((DOCS_DIR / "4.json").read_text())
    rx = data["In-Network RX"]
    assert "1a" in rx.lower() and "1b" in rx.lower(), (
        f"health/4 RX does not capture Tier 1a/1b: {rx!r}"
    )


def test_fixture_health3_has_5_tiers_in_mail_order():
    data = json.loads((DOCS_DIR / "3.json").read_text())
    mo = data["In-Network Mail Order RX"]
    assert _tier_count(mo) >= 5, (
        f"health/3 mail order should have 5 tiers: {mo!r}"
    )


def test_fixture_health4_mail_order_empty():
    """BCBS Arizona has no separate mail order pricing."""
    data = json.loads((DOCS_DIR / "4.json").read_text())
    assert data["In-Network Mail Order RX"] == ""
    assert data["Out-of-Network Mail Order RX"] == ""


def test_fixture_health5_mail_order_empty():
    """Geisinger HMO has no mail order."""
    data = json.loads((DOCS_DIR / "5.json").read_text())
    assert data["In-Network Mail Order RX"] == ""


def test_fixture_hmo_plans_have_empty_oon_rx():
    """HMO plans should have empty Out-of-Network RX."""
    for fname in ("1.json", "2.json", "3.json", "5.json"):
        data = json.loads((DOCS_DIR / fname).read_text())
        assert data["Out-of-Network RX"] == "", (
            f"health/{fname} is HMO but has OON RX: {data['Out-of-Network RX']!r}"
        )


def test_fixture_health4_oon_rx_omits_not_covered_specialty():
    """health/4 OON RX should not include Specialty (Not Covered OON)."""
    data = json.loads((DOCS_DIR / "4.json").read_text())
    oon = data["Out-of-Network RX"]
    assert oon, "health/4 should have OON RX (PPO plan)"
    assert "specialty" not in oon.lower(), (
        f"health/4 OON RX includes Specialty (should be omitted — Not Covered): {oon!r}"
    )


def test_fixture_health3_rx_deductible_captured():
    """BlueChoice has a $450 prescription drug deductible — should be in RX Deductible field."""
    data = json.loads((DOCS_DIR / "3.json").read_text())
    assert "450" in data.get("In-Network RX Deductible", ""), (
        f"health/3 RX Deductible missing $450: {data.get('In-Network RX Deductible')!r}"
    )


# ---------------------------------------------------------------------------
# Live VLM tests — require a running server + real services
# ---------------------------------------------------------------------------

@live
def test_live_health1_rx_tiers_populated():
    """Kaiser Gold HMO: 4-tier plan with clear mail order pricing."""
    body = _post("1.pdf")
    rx = body.get("In-Network RX", "")
    mo = body.get("In-Network Mail Order RX", "")
    assert _has_cost(rx), f"In-Network RX has no cost: {rx!r}"
    assert _tier_count(rx) >= 3, f"Expected >= 3 tiers: {rx!r}"
    assert _has_cost(mo), f"Mail Order RX has no cost: {mo!r}"
    assert _tier_count(mo) >= 2, f"Mail order should have >= 2 tiers: {mo!r}"
    # Mail order values must be distinct from retail
    assert rx != mo, "Retail and mail order RX must differ for this plan"
    for old in OLD_FIELDS:
        assert old not in body, f"Old field leaked into response: {old!r}"


@live
def test_live_health2_tier2_appears_twice():
    """Kaiser Platinum: non-preferred brand is also labelled Tier 2 in the document."""
    body = _post("2.pdf")
    rx = body.get("In-Network RX", "")
    # Tier 2 appears twice (preferred and non-preferred both Tier 2)
    assert rx.lower().count("tier 2") >= 2 or "non-preferred" in rx.lower(), (
        f"health/2: expected Tier 2 to appear twice or non-preferred label: {rx!r}"
    )
    assert _tier_count(rx) >= 3


@live
def test_live_health3_five_tier_hmo():
    """BlueChoice 5-tier HMO with separate retail and 90-day mail order."""
    body = _post("3.pdf")
    rx = body.get("In-Network RX", "")
    mo = body.get("In-Network Mail Order RX", "")
    assert _tier_count(rx) >= 5, f"Expected 5 tiers: {rx!r}"
    assert _tier_count(mo) >= 5, f"Expected 5 mail order tiers: {mo!r}"
    # Mail order (90-day) values should differ from retail (30-day)
    assert rx != mo
    assert body.get("Out-of-Network RX", "") == ""  # HMO


@live
def test_live_health4_tier1a_1b_and_no_mail_order():
    """BCBS Arizona PPO: split Tier 1a/1b, no dedicated mail order column."""
    body = _post("4.pdf")
    rx = body.get("In-Network RX", "")
    oon_rx = body.get("Out-of-Network RX", "")
    mo = body.get("In-Network Mail Order RX", "")
    assert "1a" in rx.lower() and "1b" in rx.lower(), (
        f"Tier 1a/1b missing from In-Network RX: {rx!r}"
    )
    assert oon_rx, "PPO plan should have OON RX"
    assert "specialty" not in oon_rx.lower(), "Specialty is Not Covered OON — should not appear"
    assert mo == "", f"No separate mail order pricing; expected empty: {mo!r}"


@live
def test_live_health5_three_tier_hmo_no_mail_order():
    """Geisinger HMO: 3-tier plan, no mail order."""
    body = _post("5.pdf")
    rx = body.get("In-Network RX", "")
    assert _tier_count(rx) >= 3, f"Expected >= 3 tiers: {rx!r}"
    assert body.get("In-Network Mail Order RX", "") == ""
    assert body.get("Out-of-Network RX", "") == ""  # HMO


@live
def test_live_no_old_fields_in_any_health_response():
    """None of the old per-tier field names should appear in any health response."""
    for num in range(1, 6):
        body = _post(f"{num}.pdf")
        for old in OLD_FIELDS:
            assert old not in body, (
                f"Old field {old!r} leaked into health/{num} response"
            )
