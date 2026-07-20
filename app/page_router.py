"""Page routing — v1 heuristic.

Scores each docling page by keyword/heading matches for the requested category,
then returns the top-N pages plus page 1 (always included for carrier/plan header).

This is a deliberate first-pass: accuracy data from real documents will drive
refinement. Keep logic isolated here so it can be swapped without touching the
rest of the orchestrator.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keywords derived from each category's field list. Longer/more-specific phrases
# score just as much as short ones — adjust weights below if needed post-benchmarking.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "dental": [
        "deductible", "annual maximum", "coinsurance", "in-network", "out-of-network",
        "cleaning", "cleanings", "exam", "x-ray", "sealant", "filling", "extraction",
        "root canal", "periodontal", "gum disease", "oral surgery", "crown", "denture",
        "bridge", "implant", "orthodontia", "orthodontic", "waiting period", "frequency",
        "preventive", "basic", "major", "carrier", "plan name", "network type",
    ],
    "vision": [
        "eye exam", "vision", "lens", "frame", "contact", "bifocal", "trifocal",
        "lenticular", "single vision", "in-network", "out-of-network", "allowance",
        "exam frequency", "lens frequency", "frame frequency", "carrier", "plan name",
    ],
    "term_life": [
        "life insurance", "term life", "accidental death", "dismemberment", "ad&d",
        "age reduction", "beneficiary", "taxation", "coverage amount", "face amount",
        "carrier", "plan name",
    ],
    "std": [
        "short-term disability", "std", "weekly benefit", "elimination period",
        "benefit period", "payment period", "disability definition", "own occupation",
        "pre-existing", "maternity", "guaranteed insurability", "taxation",
        "carrier", "plan name",
    ],
    "ltd": [
        "long-term disability", "ltd", "monthly benefit", "elimination period",
        "benefit period", "payment period", "disability definition", "own occupation",
        "pre-existing", "guaranteed insurability", "taxation", "carrier", "plan name",
    ],
    "accident": [
        "accident", "accidental", "burn", "coma", "concussion", "dental injury",
        "dislocation", "fracture", "quadriplegia", "paraplegia", "loss of speech",
        "loss of hearing", "wellness", "accidental death", "dismemberment",
        "carrier", "plan name",
    ],
    "critical_illness": [
        "critical illness", "cancer", "carcinoma", "heart attack", "stroke",
        "organ failure", "major organ", "scheduled benefit", "minimum benefit",
        "maximum benefit", "guaranteed insurability", "pre-existing", "wellness",
        "carrier", "plan name",
    ],
    "sup_life": [
        "supplemental life", "employee life", "spouse life", "child life",
        "accidental death", "dismemberment", "ad&d", "age reduction",
        "guaranteed insurability", "beneficiary", "taxation", "carrier", "plan name",
    ],
    "health": [
        "deductible", "out-of-pocket", "oop", "coinsurance", "copay",
        "in-network", "out-of-network", "pcp", "primary care", "specialist",
        "urgent care", "emergency room", "preventive", "outpatient", "inpatient",
        "surgery", "newborn", "delivery", "diagnostic", "prescription", "generic",
        "brand", "mail order", "tier", "network type", "network name",
        "deductible explanation", "carrier", "plan name",
    ],
    "health_3tier": [
        "deductible", "out-of-pocket", "oop", "coinsurance", "copay",
        "designated network", "in-network", "out-of-network", "pcp", "primary care",
        "specialist", "urgent care", "emergency room", "preventive", "outpatient",
        "inpatient", "surgery", "newborn", "delivery", "diagnostic", "prescription",
        "generic", "brand", "mail order", "tier", "three tier", "3-tier",
        "network type", "network name", "deductible explanation", "carrier", "plan name",
    ],
}

# Narrative/definitional content often carries explanation fields (e.g. "Deductible
# Explanation", "Out of Network Explanation") that live outside benefits tables.
_NARRATIVE_KEYWORDS: list[str] = [
    "network type", "explanation", "out of network", "in-network", "definition",
    "means", "refers to", "defined as", "following services", "benefit summary",
]

# Health only: phrases that identify real benefit-table rows. These get a strong
# bonus so service pages (hospital stay, urgent care, imaging) beat the SBC
# "Coverage Examples" page, which is dense with generic keywords (deductible,
# copayments, specialist, childbirth) but contains no plan values.
_HEALTH_SERVICE_ROW_PHRASES: list[str] = [
    "hospital stay", "facility fee", "urgent care", "emergency room",
    "outpatient surgery", "childbirth", "imaging (ct",
    "advanced diagnostic imaging", "ambulatory surg", "physician/surgeon",
    "office visit", "primary care physician", "specialist office",
    # The real maternity benefit row ("Childbirth/delivery facility services")
    # — scores in addition to "childbirth" so benefit pages with the actual
    # row beat duplicated/translated summary pages.
    "delivery facility",
    # SBC "Important Questions" page — holds the deductible and
    # out-of-pocket limit answers, but few service keywords.
    "out-of-pocket limit", "overall deductible",
    # UnitedHealthcare "Benefit Summary" format (as opposed to SBC): same
    # four benefits, different row wording/word order, spread one section
    # per page across many pages. Without these, "Surgery - Outpatient",
    # "Hospital - Inpatient Stay", "Major Diagnostic and Imaging - Outpatient"
    # and "Pregnancy - Maternity Services" never match any phrase above and
    # lose to front-matter pages dense in generic keywords.
    "surgery - outpatient", "hospital - inpatient stay",
    "major diagnostic and imaging", "maternity services",
]

# Markers of the SBC "Coverage Examples" page (sample-cost illustrations,
# not plan benefits).
_COVERAGE_EXAMPLE_MARKERS: list[str] = [
    "total example cost", "about these coverage examples",
    "this is not a cost estimator", "isn't a cost estimator",
]

_HEALTH_CATEGORIES: frozenset[str] = frozenset({"health", "health_3tier"})

# UnitedHealthcare "Benefit Summary" format (Choice/Choice Plus plans, distinct
# from the federal SBC template): the entire "Copays ($) and Coinsurance (%)
# for Covered Health Care Services" table repeats this exact running header on
# EVERY one of its pages — 6 to 9 consecutive pages is typical — while it never
# appears on the cover, pharmacy, coverage-example, or exclusions pages. No
# amount of keyword-score tuning lets these sparse (2-4 row) pages consistently
# outrank keyword-dense front matter within a small top_n budget, so instead
# of scoring them, every page carrying this marker is unconditionally included.
_BENEFIT_TABLE_CONTINUATION_MARKER = "what you pay for services"


# RX-specific routing: the pharmacy table can sit on a page that scores poorly
# on general health keywords (e.g. Benefit Summaries with a dedicated pharmacy
# page), so the RX extraction call selects its own pages.
_RX_KEYWORDS: list[str] = [
    "prescription", "pharmacy", "generic", "formulary", "specialty",
    "mail order", "mail-order", "home delivery", "brand", "tier", "drug",
]

_RX_COST_MARKERS: tuple[str, ...] = ("$", "coinsurance", "copay", "no charge")


def select_rx_pages(
    docling_pages: list[dict[str, Any]],
    top_n: int = 5,
) -> list[int]:
    """Return 1-based page numbers likely to contain the pharmacy cost table.

    Pages mentioning drug tiers alongside cost markers get a strong boost so a
    dedicated pharmacy page beats generic benefit pages. Returns an empty list
    when no page looks RX-related (caller falls back to category routing).
    """
    scored: list[tuple[int, float]] = []
    for page in docling_pages:
        page_num: int = page["page_number"]
        text: str = page.get("markdown", "").lower()
        score = sum(1.0 for kw in _RX_KEYWORDS if kw in text)
        has_tier_words = "generic" in text and (
            "brand" in text or "tier" in text or "level" in text or "specialty" in text
        )
        if has_tier_words and any(m in text for m in _RX_COST_MARKERS):
            score += 6.0
        # SBC "Important Questions" row 'Are there other deductibles for
        # specific services?' often states the prescription drug deductible.
        if "other deductibles" in text and "prescription" in text:
            score += 4.0
        scored.append((page_num, score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    selected = [pnum for pnum, score in scored[:top_n] if score >= 3.0]
    selected.sort()

    logger.info(
        "page_router: RX pages selected",
        extra={"selected": selected, "top_n": top_n},
    )
    return selected


def select_pages(
    docling_pages: list[dict[str, Any]],
    category: str,
    top_n: int = 5,
) -> list[int]:
    """Return 1-based page numbers selected for VLM extraction.

    Selection = top-N pages by keyword score for the category, always including
    page 1 (carrier/plan header info). Ties broken by page order.
    """
    keywords = _CATEGORY_KEYWORDS.get(category, [])
    scored: list[tuple[int, float]] = []

    for page in docling_pages:
        page_num: int = page["page_number"]
        text: str = page.get("markdown", "").lower()

        score = sum(1.0 for kw in keywords if kw in text)
        score += 0.5 * sum(1.0 for kw in _NARRATIVE_KEYWORDS if kw in text)

        if category in _HEALTH_CATEGORIES:
            # Coverage-example pages name many services ("specialist office
            # visits", "childbirth") without containing plan values — they get
            # the penalty and never the service-row bonus.
            if any(marker in text for marker in _COVERAGE_EXAMPLE_MARKERS):
                score -= 10.0
            else:
                score += 2.0 * sum(
                    1.0 for phrase in _HEALTH_SERVICE_ROW_PHRASES if phrase in text
                )

        scored.append((page_num, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    selected_set = {pnum for pnum, _ in scored[:top_n]}

    table_pages: list[int] = []
    if category in _HEALTH_CATEGORIES:
        table_pages = [
            page["page_number"] for page in docling_pages
            if _BENEFIT_TABLE_CONTINUATION_MARKER in page.get("markdown", "").lower()
        ]
        selected_set |= set(table_pages)

    # The benefit-cost table always starts within the first few pages of an
    # SBC / Benefit Summary (right after the cover/header); noise sections
    # that outscore it on keywords (Coverage Examples, Glossary of Terms,
    # legal/language notices) always come later. Guaranteeing the first 5
    # pages sidesteps having to keep discovering and patching new noise-page
    # types one at a time.
    selected_set |= {
        page["page_number"] for page in docling_pages if page["page_number"] <= 5
    }
    selected = sorted(selected_set)

    logger.info(
        "page_router: pages selected",
        extra={
            "category": category,
            "top_n": top_n,
            "all_scores": [(pnum, round(s, 2)) for pnum, s in scored],
            "benefit_table_pages": table_pages,
            "selected": selected,
        },
    )
    return selected
