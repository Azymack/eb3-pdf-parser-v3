"""
Field-level accuracy comparison: live extraction vs. fixture expected values.

Comparison rule: a field PASSES if every number found in the fixture value
also appears in the live value.  Fields with no numbers (text-only or empty)
are checked for populated/empty agreement only.

Usage:
    python tests/compare_extraction.py
    python tests/compare_extraction.py --category health
    python tests/compare_extraction.py --category health_3tier
    python tests/compare_extraction.py --fail-only
"""
import argparse
import json
import re
import sys
from pathlib import Path

import httpx

API_URL = "http://localhost:8002/extract_json_v2"
HEADERS = {"X-EB3-Token": "eb3-key-1"}
HEALTH_DIR = Path("tests/documents/health")
HEALTH_3TIER_DIR = Path("tests/documents/health_3tier")

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Fields that are unreliable / intentionally skipped in comparison
SKIP_FIELDS = {
    "Member Website",
    "Customer Service Phone Number",
    "Plan Year",
    "Carrier Name",
    "Plan Name",
    "Network Name",
}


def extract_numbers(s: str) -> set[str]:
    return set(_NUM_RE.findall(s or ""))


def compare_field(fixture_val: str, live_val: str) -> tuple[str, str]:
    """
    Returns (status, reason).
    Status: PASS | FAIL | MISS | EXTRA | SKIP
    """
    fv = (fixture_val or "").strip()
    lv = (live_val or "").strip()

    if not fv:
        if lv:
            return "EXTRA", "fixture empty, live has value"
        return "SKIP", "both empty"

    if not lv:
        return "MISS", "fixture has value, live is empty"

    f_nums = extract_numbers(fv)
    l_nums = extract_numbers(lv)

    if f_nums:
        missing = f_nums - l_nums
        if missing:
            return "FAIL", f"numbers missing from live: {sorted(missing)}"
        return "PASS", ""

    # No numbers — just check populated/empty agreement
    return "PASS", "(text-only field, populated in both)"


def run_doc(pdf_path: Path, category: str) -> list[dict]:
    fixture = json.loads(pdf_path.with_suffix(".json").read_text(encoding="utf-8"))
    try:
        resp = httpx.post(
            API_URL,
            headers=HEADERS,
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
            data={"category": category},
            timeout=180,
        )
        resp.raise_for_status()
        live = resp.json()
    except Exception as e:
        return [{"field": "REQUEST", "status": "ERROR", "reason": str(e),
                 "fixture": "", "live": ""}]

    rows = []
    for field, fval in fixture.items():
        if field in SKIP_FIELDS:
            continue
        lval = live.get(field, "")
        status, reason = compare_field(fval, lval)
        rows.append({
            "field": field,
            "status": status,
            "reason": reason,
            "fixture": (fval or "")[:80],
            "live": (lval or "")[:80],
        })
    return rows


def print_report(doc_name: str, rows: list[dict], fail_only: bool):
    counts = {"PASS": 0, "FAIL": 0, "MISS": 0, "EXTRA": 0, "SKIP": 0, "ERROR": 0}
    for r in rows:
        counts[r["status"]] += 1

    flag = "OK" if counts["FAIL"] == 0 and counts["MISS"] == 0 else "!!"
    print(f"\n{flag} {doc_name}  "
          f"PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"MISS={counts['MISS']}  EXTRA={counts['EXTRA']}  SKIP={counts['SKIP']}")

    for r in rows:
        if fail_only and r["status"] in ("PASS", "SKIP", "EXTRA"):
            continue
        if r["status"] in ("FAIL", "MISS", "ERROR"):
            print(f"  [{r['status']}] {r['field']}")
            print(f"         fixture: {r['fixture']!r}")
            print(f"         live:    {r['live']!r}")
            if r["reason"]:
                print(f"         reason:  {r['reason']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["health", "health_3tier"], default=None)
    parser.add_argument("--fail-only", action="store_true",
                        help="Only print failing fields")
    args = parser.parse_args()

    pairs: list[tuple[Path, str]] = []
    if args.category in (None, "health"):
        pairs += [(p, "health") for p in sorted(HEALTH_DIR.glob("*.pdf"))]
    if args.category in (None, "health_3tier"):
        pairs += [(p, "health_3tier") for p in sorted(HEALTH_3TIER_DIR.glob("*.pdf"))]

    total = {"PASS": 0, "FAIL": 0, "MISS": 0, "EXTRA": 0, "SKIP": 0}
    doc_results = []

    for pdf, cat in pairs:
        print(f"  running {pdf.name} ...", end="\r", flush=True)
        rows = run_doc(pdf, cat)
        doc_results.append((pdf.name, rows))
        for r in rows:
            if r["status"] in total:
                total[r["status"]] += 1

    print(" " * 60, end="\r")  # clear progress line

    for doc_name, rows in doc_results:
        print_report(doc_name, rows, args.fail_only)

    print("\n" + "=" * 60)
    print(f"TOTAL  PASS={total['PASS']}  FAIL={total['FAIL']}  "
          f"MISS={total['MISS']}  EXTRA={total['EXTRA']}  SKIP={total['SKIP']}")
    total_checked = total["PASS"] + total["FAIL"] + total["MISS"]
    if total_checked:
        pct = 100 * total["PASS"] / total_checked
        print(f"Accuracy (PASS / checked): {pct:.1f}%")

    if total["FAIL"] > 0 or total["MISS"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
