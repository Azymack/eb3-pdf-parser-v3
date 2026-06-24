"""Live integration test against the running API + real docling/VLM services.

Usage:
    python tests/run_live_test.py [health/1 health/2 ...]
    (defaults to all documents found under tests/documents/)

Compares extracted fields to expected JSON. Numeric values are normalized
before comparison (strips $, commas, spaces) so "$1,650" == "1650" == "1,650".
"""
import json
import re
import sys
from pathlib import Path

import httpx

API_URL = "http://localhost:8002/extract_json_v2"
HEADERS = {"X-EB3-Token": "eb3-key-1"}
DOCS_DIR = Path("tests/documents")


_EMPTY_SENTINELS = {
    "not covered", "not applicable", "not available", "n/a", "none",
    "no coverage", "not covered (hmo)", "not covered/not applicable",
}

def normalize(value: str | None) -> str:
    """Normalize values so semantically equivalent representations compare equal.

    Rules applied:
    - Treat empty string and "not covered"/"not applicable"/etc. as equivalent.
    - Strip leading $ and trailing % so numeric values compare equal to copay strings.
    - Strip common suffixes like '/ visit', 'copayment', 'coinsurance' to expose the number.
    - Lowercase and remove commas/spaces for consistent comparison.
    """
    if not value:
        return ""
    v = value.strip().lower()
    if v in _EMPTY_SENTINELS:
        return ""
    # Strip currency prefix and trailing % for coinsurance comparisons
    v = v.lstrip("$").rstrip("%").replace(",", "").strip()
    # Strip common per-visit/copay suffixes so "20 copayment/visit" -> "20"
    for suffix in (" / visit", " / admission", " copayment", " copayment/visit",
                   " coinsurance", "/visit", "/admission", " per visit"):
        if v.endswith(suffix):
            v = v[: -len(suffix)].strip()
    return v


def compare_fields(expected: dict, actual: dict) -> tuple[list, list, list]:
    """Return (matches, mismatches, missing) lists of field names."""
    matches, mismatches, missing = [], [], []
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        norm_exp = normalize(exp_val)
        # VLM returning null is equivalent to an empty-string expected value (field not on plan).
        if act_val is None:
            if norm_exp == "":
                matches.append(key)
            else:
                missing.append(key)
            continue
        if norm_exp == normalize(act_val):
            matches.append(key)
        else:
            mismatches.append((key, exp_val, act_val))
    return matches, mismatches, missing


def run_one(pdf_path: Path, category: str, expected: dict) -> dict:
    pdf_bytes = pdf_path.read_bytes()
    # include_metadata=true so we get stage_timings, pages_used, etc. alongside fields
    resp = httpx.post(
        API_URL,
        headers=HEADERS,
        files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
        data={"category": category},
        params={"include_metadata": "true"},
        timeout=120,
    )
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}

    body = resp.json()
    actual_fields = body.get("fields", {})
    matches, mismatches, missing = compare_fields(expected, actual_fields)

    total = len(expected)
    score = len(matches) / total if total else 0

    return {
        "score": f"{len(matches)}/{total} ({score:.0%})",
        "stage_timings": body.get("stage_timings"),
        "pages_used": body.get("pages_used"),
        "low_confidence": body.get("low_confidence_fields"),
        "mismatches": [
            {"field": k, "expected": e, "actual": a} for k, e, a in mismatches
        ],
        "missing": missing,
    }


def collect_cases() -> list[tuple[Path, str, dict]]:
    cases = []
    for cat_dir in sorted(DOCS_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for json_file in sorted(cat_dir.glob("*.json")):
            pdf_file = json_file.with_suffix(".pdf")
            if not pdf_file.exists():
                continue
            expected = json.loads(json_file.read_text())
            cases.append((pdf_file, category, expected))
    return cases


def main():
    cases = collect_cases()
    if not cases:
        print("No test documents found under tests/documents/")
        sys.exit(1)

    print(f"Running {len(cases)} document(s) through the live API...\n")
    all_results = {}

    for pdf_path, category, expected in cases:
        label = f"{category}/{pdf_path.stem}"
        print(f"-- {label} ", end="", flush=True)
        result = run_one(pdf_path, category, expected)
        all_results[label] = result
        if "error" in result:
            print(f"FAILED: {result['error']} — {result['detail']}")
        else:
            print(f"{result['score']}  pages={result['pages_used']}")
            if result["mismatches"]:
                for m in result["mismatches"]:
                    print(f"     MISMATCH  {m['field']!r}")
                    print(f"       expected: {m['expected']!r}")
                    print(f"       actual:   {m['actual']!r}")
            if result["missing"]:
                print(f"     MISSING fields: {result['missing']}")
            timings = result.get("stage_timings") or {}
            print(f"     timings: {timings}")

    print("\n-- Summary --------------------------------------------------")
    for label, result in all_results.items():
        if "error" in result:
            print(f"  FAIL  {label}: {result['error']}")
        else:
            print(f"  {result['score']:>12}  {label}")


if __name__ == "__main__":
    main()
