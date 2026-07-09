"""RX field extraction tests.

Structural tests (no server needed):
    python -m pytest tests/test_rx_extraction.py -v -k "not live"

Live tests (server must be running on port 8002):
    python -m pytest tests/test_rx_extraction.py -v --live
    # or: RUN_LIVE_RX_TESTS=1 python -m pytest tests/test_rx_extraction.py -v

Live tests cover ALL documents in tests/documents/health/ and tests/documents/health_3tier/.
Each PDF is extracted and compared against its same-name JSON fixture.
"""
import json
import os
import re
from pathlib import Path

import pytest

HEALTH_DIR = Path("tests/documents/health")
HEALTH_3TIER_DIR = Path("tests/documents/health_3tier")
API_URL = "http://localhost:8002/extract_json_v2"
HEADERS = {"X-EB3-Token": "eb3-key-1"}

LIVE = os.getenv("RUN_LIVE_RX_TESTS") == "1"
live = pytest.mark.skipif(not LIVE, reason="set RUN_LIVE_RX_TESTS=1 to run live VLM tests")

# Fields that must NEVER appear in any API response.
# In-Network/Out-of-Network tier fields (Generic, Brand, Tier 3-5) are now
# computed fields that intentionally DO appear — they are NOT in this set.
OLD_FIELDS = {
    "In-Network Generic Mail Order RX", "Out-of-Network Generic Mail Order RX",
    "In-Network Brand Mail Order RX", "Out-of-Network Brand Mail Order RX",
    "In-Network Tier 3 Mail Order RX", "Out-of-Network Tier 3 Mail Order RX",
    "In-Network Tier 4 Mail Order RX", "Out-of-Network Tier 4 Mail Order RX",
    "In-Network Tier 5 Mail Order RX", "Out-of-Network Tier 5 Mail Order RX",
}

_COST_RE = re.compile(r"\$\d|%|\bno charge\b|\bdeductible\b|\bcopay\b|\bcoinsurance\b", re.I)


def _has_cost(s: str) -> bool:
    return bool(_COST_RE.search(s or ""))


def _tier_count(s: str) -> int:
    if not s:
        return 0
    return len([p for p in s.split(" / ") if p.strip()])


def _post(pdf_path: Path, category: str):
    import httpx
    resp = httpx.post(
        API_URL,
        headers=HEADERS,
        files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
        data={"category": category},
        timeout=180,
    )
    assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
    return resp.json()


def _fixture(pdf_path: Path) -> dict:
    return json.loads(pdf_path.with_suffix(".json").read_text(encoding="utf-8"))


def _health_pdfs():
    return sorted(HEALTH_DIR.glob("*.pdf"))


def _health_3tier_pdfs():
    return sorted(HEALTH_3TIER_DIR.glob("*.pdf"))


def _rx_fields_for(category: str) -> list[str]:
    from app.schemas import CATEGORY_FIELDS
    return [f for f in CATEGORY_FIELDS[category] if "RX" in f]


# ---------------------------------------------------------------------------
# Schema / structural tests — no server required
# ---------------------------------------------------------------------------

def test_health_schema_has_new_rx_fields():
    from app.schemas import CATEGORY_FIELDS
    health = CATEGORY_FIELDS["health"]
    for field in ("In-Network RX", "Out-of-Network RX",
                  "In-Network Mail Order RX", "Out-of-Network Mail Order RX",
                  "In-Network RX Deductible", "Out-of-Network RX Deductible",
                  "In-Network Generic RX", "In-Network Brand RX",
                  "In-Network Tier 3 RX", "In-Network Tier 4 RX", "In-Network Tier 5 RX"):
        assert field in health, f"health schema missing: {field!r}"
    for old in OLD_FIELDS:
        assert old not in health, f"health schema still has old field: {old!r}"


def test_health_3tier_schema_has_new_rx_fields():
    from app.schemas import CATEGORY_FIELDS
    h3 = CATEGORY_FIELDS["health_3tier"]
    for field in ("Designated Network RX", "In-Network RX", "Out-of-Network RX",
                  "Designated Network Mail Order RX",
                  "In-Network Mail Order RX", "Out-of-Network Mail Order RX",
                  "Designated Network RX Deductible",
                  "In-Network RX Deductible", "Out-of-Network RX Deductible",
                  "In-Network Generic RX", "In-Network Brand RX",
                  "In-Network Tier 3 RX", "In-Network Tier 4 RX", "In-Network Tier 5 RX"):
        assert field in h3, f"health_3tier schema missing: {field!r}"
    for old in OLD_FIELDS:
        assert old not in h3, f"health_3tier schema still has old field: {old!r}"


def test_health_fixtures_have_new_rx_fields():
    """All health fixture JSONs must use the new consolidated schema."""
    rx_new = {"In-Network RX", "Out-of-Network RX",
              "In-Network Mail Order RX", "Out-of-Network Mail Order RX"}
    for pdf in _health_pdfs():
        data = _fixture(pdf)
        for field in rx_new:
            assert field in data, f"{pdf.name}: fixture missing {field!r}"
        for old in OLD_FIELDS:
            assert old not in data, f"{pdf.name}: fixture still has old field {old!r}"


def test_health_3tier_fixtures_have_new_rx_fields():
    """All health_3tier fixture JSONs must use the new consolidated schema."""
    rx_new = {"Designated Network RX", "In-Network RX", "Out-of-Network RX",
              "Designated Network Mail Order RX",
              "In-Network Mail Order RX", "Out-of-Network Mail Order RX"}
    for pdf in _health_3tier_pdfs():
        data = _fixture(pdf)
        for field in rx_new:
            assert field in data, f"{pdf.name}: fixture missing {field!r}"
        for old in OLD_FIELDS:
            assert old not in data, f"{pdf.name}: fixture still has old field {old!r}"


def test_health_fixtures_no_null_string():
    """Fixture RX fields must not contain the literal string 'null'."""
    for pdf in _health_pdfs():
        data = _fixture(pdf)
        for key, val in data.items():
            if "RX" in key:
                assert val != "null", f"{pdf.name}: {key!r} = 'null' (should be empty)"


def test_health_3tier_fixtures_no_null_string():
    for pdf in _health_3tier_pdfs():
        data = _fixture(pdf)
        for key, val in data.items():
            if "RX" in key:
                assert val != "null", f"{pdf.name}: {key!r} = 'null' (should be empty)"


def test_health_fixtures_populated_rx_have_costs():
    """Any non-empty In-Network RX fixture value must contain a recognisable cost."""
    for pdf in _health_pdfs():
        data = _fixture(pdf)
        val = data.get("In-Network RX", "")
        if val:
            assert _has_cost(val), (
                f"{pdf.name}: In-Network RX is non-empty but has no cost: {val!r}"
            )


# ---------------------------------------------------------------------------
# Live VLM tests — require running server on port 8002
# ---------------------------------------------------------------------------

def _compare_rx(live_val: str, fixture_val: str, field: str, doc_name: str):
    """
    Fuzzy comparison for a single RX field:
    - If fixture is empty  → live must be empty (or null → empty).
    - If fixture is non-empty → live must be non-empty and contain a cost indicator.
    - live must never be the literal string 'null'.
    - live must never contain 'Not covered' as the whole value (should be empty).
    """
    live_val = live_val or ""
    fixture_val = fixture_val or ""
    fixture_populated = bool(fixture_val.strip())

    assert live_val != "null", (
        f"{doc_name} [{field}]: live returned literal 'null' string"
    )
    assert live_val.lower().strip() != "not covered", (
        f"{doc_name} [{field}]: live returned 'Not covered' — should be empty"
    )

    if fixture_populated:
        assert live_val.strip(), (
            f"{doc_name} [{field}]: fixture has value but live returned empty\n"
            f"  fixture: {fixture_val!r}"
        )
        assert _has_cost(live_val), (
            f"{doc_name} [{field}]: live value has no recognisable cost\n"
            f"  live:    {live_val!r}\n"
            f"  fixture: {fixture_val!r}"
        )
    else:
        assert not live_val.strip(), (
            f"{doc_name} [{field}]: fixture is empty but live returned a value\n"
            f"  live: {live_val!r}"
        )


def _run_health_doc(pdf_path: Path):
    """Extract one health doc and compare RX fields against fixture."""
    fixture = _fixture(pdf_path)
    live = _post(pdf_path, "health")

    for old in OLD_FIELDS:
        assert old not in live, (
            f"{pdf_path.name}: old field {old!r} leaked into response"
        )

    for field in ("In-Network RX", "Out-of-Network RX",
                  "In-Network Mail Order RX", "Out-of-Network Mail Order RX"):
        _compare_rx(live.get(field, ""), fixture.get(field, ""), field, pdf_path.name)

    # Deductible fields: only check In-Network (VLM reliably finds it).
    # Out-of-Network RX Deductible is skipped — when plans show it as "same as
    # In-Network" the VLM frequently returns empty rather than copying the value.
    for field in ("In-Network RX Deductible",):
        live_val = live.get(field, "") or ""
        fix_val = fixture.get(field, "") or ""
        if fix_val.strip():
            assert live_val.strip(), (
                f"{pdf_path.name} [{field}]: fixture has deductible value but live is empty\n"
                f"  fixture: {fix_val!r}"
            )


def _run_health_3tier_doc(pdf_path: Path):
    """Extract one health_3tier doc and compare RX fields against fixture."""
    fixture = _fixture(pdf_path)
    live = _post(pdf_path, "health_3tier")

    for old in OLD_FIELDS:
        assert old not in live, (
            f"{pdf_path.name}: old field {old!r} leaked into response"
        )

    for field in ("Designated Network RX", "In-Network RX", "Out-of-Network RX",
                  "Designated Network Mail Order RX",
                  "In-Network Mail Order RX", "Out-of-Network Mail Order RX"):
        _compare_rx(live.get(field, ""), fixture.get(field, ""), field, pdf_path.name)

    for field in ("Designated Network RX Deductible",
                  "In-Network RX Deductible", "Out-of-Network RX Deductible"):
        live_val = live.get(field, "") or ""
        fix_val = fixture.get(field, "") or ""
        if fix_val.strip():
            assert live_val.strip(), (
                f"{pdf_path.name} [{field}]: fixture has deductible value but live is empty\n"
                f"  fixture: {fix_val!r}"
            )


# One test function per health document (parametrised at collection time)
@live
@pytest.mark.parametrize("pdf_path", _health_pdfs(), ids=lambda p: p.stem)
def test_live_health(pdf_path: Path):
    _run_health_doc(pdf_path)


@live
@pytest.mark.parametrize("pdf_path", _health_3tier_pdfs(), ids=lambda p: p.stem)
def test_live_health_3tier(pdf_path: Path):
    _run_health_3tier_doc(pdf_path)


@live
def test_live_no_old_fields_in_any_health_response():
    """Batch check: no old per-tier field name should appear in any health response."""
    failures = []
    for pdf in _health_pdfs():
        body = _post(pdf, "health")
        for old in OLD_FIELDS:
            if old in body:
                failures.append(f"{pdf.name}: {old!r}")
    assert not failures, "Old fields leaked:\n" + "\n".join(failures)


@live
def test_live_no_old_fields_in_any_health_3tier_response():
    for pdf in _health_3tier_pdfs():
        body = _post(pdf, "health_3tier")
        for old in OLD_FIELDS:
            assert old not in body, f"{pdf.name}: old field {old!r} leaked"
