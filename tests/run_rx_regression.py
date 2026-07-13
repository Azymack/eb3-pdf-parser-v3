"""RX regression check: run fixture-backed documents through the live API and
compare RX fields against their expected JSON fixtures.

Fuzzy comparison (same philosophy as test_rx_extraction._compare_rx):
  - fixture empty     -> live must be empty or 'Not covered'
  - fixture populated -> live must be populated with a recognisable cost
Exact string equality is NOT required — consolidated formatting differs
between pipeline versions; what matters is no field goes silently missing.

Usage:
    python tests/run_rx_regression.py [--port 8003] [--dirs health health_3tier]
"""
import argparse
import concurrent.futures
import json
import re
from pathlib import Path

import httpx

DOCS_ROOT = Path("tests/documents")
HEADERS = {"X-EB3-Token": "eb3-key-1"}

_COST_RE = re.compile(
    r"\$\d|%|\bno charge\b|\bdeductible\b|\bcopay\b|\bcoinsurance\b|\bnot covered\b",
    re.I,
)

RX_COMPARE_FIELDS = {
    "health": ["In-Network RX", "Out-of-Network RX",
               "In-Network Mail Order RX", "Out-of-Network Mail Order RX",
               "In-Network RX Deductible"],
    "health_3tier": ["Designated Network RX", "In-Network RX", "Out-of-Network RX",
                     "Designated Network Mail Order RX",
                     "In-Network Mail Order RX", "Out-of-Network Mail Order RX"],
}


def check_doc(pdf: Path, category: str, port: int) -> tuple[str, list[str]]:
    fixture = json.loads(pdf.with_suffix(".json").read_text(encoding="utf-8"))
    try:
        resp = httpx.post(
            f"http://localhost:{port}/extract_json_v2",
            headers=HEADERS,
            files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
            data={"category": category},
            timeout=240,
        )
    except Exception as exc:
        return pdf.stem, [f"REQUEST ERROR: {exc}"]
    if resp.status_code != 200:
        return pdf.stem, [f"HTTP {resp.status_code}: {resp.text[:200]}"]
    live = resp.json()

    problems: list[str] = []
    for field in RX_COMPARE_FIELDS[category]:
        fix_val = (fixture.get(field) or "").strip()
        live_val = (live.get(field) or "").strip()
        fix_pop = bool(fix_val) and fix_val.lower() != "not covered"
        live_pop = bool(live_val) and live_val.lower() != "not covered"
        if fix_pop and not live_pop:
            problems.append(
                f"{field}: fixture populated but live lost it\n"
                f"      fixture: {fix_val[:120]!r}\n      live:    {live_val[:120]!r}"
            )
        elif fix_pop and live_pop and not _COST_RE.search(live_val):
            problems.append(f"{field}: live value has no cost: {live_val[:120]!r}")
        elif not fix_pop and live_pop:
            problems.append(
                f"{field}: fixture empty but live added value: {live_val[:120]!r}"
            )
    return pdf.stem, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dirs", nargs="*", default=["health", "health_3tier"])
    args = ap.parse_args()

    cases = []
    for d in args.dirs:
        for pdf in sorted((DOCS_ROOT / d).glob("*.pdf")):
            if pdf.with_suffix(".json").exists():
                cases.append((pdf, d))
    print(f"Regression: {len(cases)} fixture docs\n", flush=True)

    clean = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_doc, p, c, args.port): p for p, c in cases}
        for fut in concurrent.futures.as_completed(futures):
            stem, problems = fut.result()
            if problems:
                print(f"!! {stem}", flush=True)
                for pr in problems:
                    print(f"   {pr}", flush=True)
            else:
                clean += 1
                print(f"ok {stem}", flush=True)
    print(f"\n{clean}/{len(cases)} clean", flush=True)


if __name__ == "__main__":
    main()
