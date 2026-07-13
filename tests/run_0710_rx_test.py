"""Batch-run the 0710 failing documents through the live API and dump RX fields.

Usage:
    python tests/run_0710_rx_test.py [--out results_baseline] [--port 8003] [--only name1 name2]

Saves the full metadata response per document to
tests/documents/0710/<out>/<pdf-stem>.json and prints an RX-field summary.
"""
import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import httpx

DOCS_DIR = Path("tests/documents/0710")
HEADERS = {"X-EB3-Token": "eb3-key-1"}

RX_FIELDS = [
    "In-Network RX Deductible", "Out-of-Network RX Deductible",
    "In-Network RX", "Preferred Network RX", "Out-of-Network RX",
    "In-Network Generic RX", "Out-of-Network Generic RX",
    "In-Network Brand RX", "Out-of-Network Brand RX",
    "In-Network Tier 3 RX", "Out-of-Network Tier 3 RX",
    "In-Network Tier 4 RX", "Out-of-Network Tier 4 RX",
    "In-Network Tier 5 RX", "Out-of-Network Tier 5 RX",
    "In-Network Mail Order RX", "Out-of-Network Mail Order RX",
]


def run_one(pdf_path: Path, out_dir: Path, port: int) -> tuple[str, dict]:
    url = f"http://localhost:{port}/extract_json_v2"
    try:
        resp = httpx.post(
            url,
            headers=HEADERS,
            files={"file": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")},
            data={"category": "health"},
            params={"include_metadata": "true"},
            timeout=240,
        )
    except Exception as exc:
        return pdf_path.stem, {"error": str(exc)}
    if resp.status_code != 200:
        return pdf_path.stem, {"error": f"HTTP {resp.status_code}", "detail": resp.text[:500]}
    body = resp.json()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{pdf_path.stem}.json").write_text(
        json.dumps(body, indent=2), encoding="utf-8"
    )
    return pdf_path.stem, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_baseline")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None,
                    help="substring filters on pdf stem")
    args = ap.parse_args()

    out_dir = DOCS_DIR / args.out
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if any(s.lower() in p.stem.lower() for s in args.only)]
    if not pdfs:
        print("No PDFs matched")
        sys.exit(1)

    print(f"Running {len(pdfs)} document(s) against port {args.port} -> {out_dir}\n", flush=True)
    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, p, out_dir, args.port): p for p in pdfs}
        for fut in concurrent.futures.as_completed(futures):
            stem, body = fut.result()
            results[stem] = body
            if "error" in body:
                print(f"== {stem}: ERROR {body['error']} {body.get('detail','')[:200]}", flush=True)
                continue
            fields = body.get("fields", {})
            print(f"== {stem}  pages={body.get('pages_used')}", flush=True)
            for f in RX_FIELDS:
                if f in fields:
                    print(f"   {f}: {fields[f]!r}", flush=True)
            print("", flush=True)

    n_err = sum(1 for b in results.values() if "error" in b)
    print(f"\nDone: {len(results)} docs, {n_err} errors. Full responses in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
