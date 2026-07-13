# RX (Prescription Drug) Extraction Architecture

*Since 2026-07: structured extraction replaces consolidated-string parsing.*

## The problem with the old approach

The old pipeline asked the main VLM call to serialize the pharmacy table into
one consolidated string per network ("Label: cost / Label: cost / ..."), then
~1,700 lines of regex in `post_process.py` re-parsed those strings into the
per-tier output fields, and a ~200-line RX prompt block accumulated one
carrier-specific formatting rule per failing document (UHC cells, BCBS 6-row
tables, Premera commas, Anthem home-delivery parentheticals, ...).

That design fails structurally, not incidentally:

1. **Carrier tier vocabulary is open-ended.** "Level 1", "Value generic
   drugs", "Preferred Drugs", "Tier 1a" — a regex label mapper can never
   anticipate the next carrier's names. Unrecognized labels were silently
   dropped, so whole documents "didn't scrape at all" even though the VLM had
   extracted the data into the consolidated string.
2. **Per-cell channel formats vary endlessly.** "$10 (retail) and $25 (home
   delivery)", "30-day supply: $25; 90-day supply: $62.50",
   "Retail/Mail Order (1-30 days) $10/Fill. Mail Order (31-90 days) $20/Fill.",
   "…-Retail 84-90-day supply & mail order 31-90-day supply." Every new format
   needed a new splitting regex, and new regexes broke old documents.
3. **The intermediate representation was lossy.** Forcing a 2-D table
   (tiers × networks × channels) through a 1-D string and re-parsing it is
   strictly harder than extracting the table structure directly.

## The new approach

**The VLM does the semantic work; Python does only deterministic assembly.**

A dedicated RX-only VLM call (`app/rx_extractor.py`) runs concurrently with
the main field-extraction call and returns structured JSON via vLLM
`guided_json`:

```json
{
  "drug_rows": [
    {
      "label": "Level 2: Preferred brand drugs and certain higher cost preferred generic drugs",
      "standard_tier": "preferred_brand",
      "in_network_retail": "20% coinsurance ($50 max)",
      "in_network_mail_order": null,
      "preferred_pharmacy_retail": null,
      "preferred_pharmacy_mail_order": null,
      "out_of_network_retail": null,
      "out_of_network_mail_order": null
    }
  ],
  "mail_order_service": true,
  "out_of_network_pharmacy": "emergency_or_reimbursement_only",
  "rx_deductible_in_network": null,
  "rx_deductible_out_of_network": null
}
```

Key decisions:

- **Tier classification moved INTO the VLM** (`standard_tier` enum:
  `generic`, `preferred_brand`, `non_preferred_brand`, `preferred_specialty`,
  `non_preferred_specialty`, `other`). The model reads "Level 1: Preferred
  generic drugs..." and classifies by meaning — the thing regexes could not
  do. Combined rows fall back to the printed tier number (Tier 3 →
  non_preferred_brand, etc.).
- **Channel splitting moved INTO the VLM.** Each row carries separate
  retail / mail_order costs per network; the prompt shows the recurring cell
  formats once, as *examples of the concept*, not as per-carrier rules.
- **Python assembly is pure bookkeeping** (`assemble_rx_fields`, unit-tested
  without a server): bucket rows by `standard_tier`, join multiple rows in a
  slot with " / ", build the combined Mail Order fields in tier order
  (cost-only), propagate "Not covered" for OON pharmacies, normalize
  deductible noise. A cost-marker gate drops narrative cells the model failed
  to null out.
- **RX-specific page routing** (`select_rx_pages`): the pharmacy table can
  live on a page that scores poorly on general health keywords (e.g. BCBS AZ
  Benefit Summaries); the RX call selects its own pages and falls back to the
  category pages when nothing matches.
- **The main VLM call no longer sees RX fields at all.** The 200-line RX
  prompt block was deleted from `vlm_client.py`. On RX-call failure the RX
  fields degrade to empty strings; the rest of the extraction is unaffected.

## Standard tier slots (fixed output layout)

| standard_tier           | Output field suffix | Meaning                        |
|-------------------------|---------------------|--------------------------------|
| generic                 | Generic RX (Tier 1) | all generic rows (value/low-cost/preferred/non-preferred generic merge here) |
| preferred_brand         | Brand RX (Tier 2)   | preferred / formulary brand    |
| non_preferred_brand     | Tier 3 RX           | non-preferred / non-formulary brand |
| preferred_specialty     | Tier 4 RX           | specialty (or preferred specialty) |
| non_preferred_specialty | Tier 5 RX           | non-preferred specialty        |

Networks: `In-Network` / `Out-of-Network` (health) plus `Designated Network`
(health_3tier). health also supports a separate preferred-pharmacy column,
which merges into In-Network per-tier values as "preferred / standard" and
feeds `Preferred Network RX`.

## Rules that used to be post-process patches, now prompt-level concepts

- Mail order must be a **printed price**, never computed from multiplier rules
  ("2 copays apply to certain 90-day supply mail orders" → null).
- Narrative cells ("Prescriptions may be filled at an out-of-network pharmacy
  in emergency situations only…") are not costs → null (plus a Python
  cost-marker gate as backstop).
- `rx_deductible_*` only for deductibles explicitly labeled prescription/
  pharmacy — never the medical deductible.
- OON pharmacy status is an enum; `not_covered` propagates "Not covered" into
  every OON RX output field (testers want the printed "Not covered", not
  blanks).

## How to iterate now

When a new document extracts RX incorrectly:

1. Look at the structured JSON the model returned (log line
   `rx_extractor: extraction complete` and re-run with the doc via
   `tests/run_0710_rx_test.py`).
2. If a row is mis-**classified** → improve the `standard_tier` definitions in
   `_build_rx_system_prompt` (concept level, with the new label as an example).
3. If a cell is mis-**split** → add the cell shape to the CHANNELS example
   list (again: one example of the concept, not a carrier-specific branch).
4. If assembly is wrong → fix `assemble_rx_fields` and add a unit test in
   `tests/test_rx_extractor.py` (pure function, no server needed).

Never add per-carrier regexes back into post_process for RX — that is the
failure mode this architecture removed. The legacy RX parsing in
`post_process.py` remains only as dead code for non-RX-category safety and
can be deleted once the structured path has soaked in production.

## Testing

- `python -m pytest tests/test_rx_extractor.py` — assembly + routing unit tests.
- `python tests/run_0710_rx_test.py --out results_vN` — batch the 0710 corpus
  through the live API, dumping full responses + an RX field summary.
- `python tests/run_rx_regression.py` — fuzzy regression of fixture-backed
  documents (populated-ness + cost presence, not exact strings).
