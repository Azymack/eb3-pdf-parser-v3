"""Page routing tests — health service-page bonus and coverage-example penalty."""
from app.page_router import select_pages


def _page(num: int, text: str) -> dict:
    return {"page_number": num, "markdown": text}


# A benefit page with real service rows but few generic keywords.
SERVICE_PAGE = (
    "Urgent care $50/visit; deductible does not apply. "
    "Facility fee (e.g., hospital room) $300/visit plus 40% coinsurance. "
    "If you have a hospital stay Physician/surgeon fees 40% coinsurance."
)

# The SBC "Coverage Examples" page — dense with generic keywords, no plan values.
EXAMPLE_PAGE = (
    "About these Coverage Examples: This is not a cost estimator. "
    "Peg is Having a Baby. Specialist office visits (prenatal care), "
    "Childbirth/Delivery Professional Services, Diagnostic tests (ultrasounds "
    "and blood work), Specialist visit (anesthesia). Total Example Cost $12,700. "
    "Deductibles $2,800 Copayments $300 Coinsurance $1,600. Managing Joe's "
    "type 2 Diabetes: primary care physician office visits, prescription drugs, "
    "urgent care, emergency room. The plan's overall deductible, specialist "
    "copayment, hospital facility coinsurance, generic brand preventive."
)

FILLER = "General plan information and definitions."


def test_service_page_beats_coverage_example_page_for_health():
    pages = [
        _page(1, FILLER),
        _page(2, (
            "deductible coinsurance copay in-network out-of-network "
            "pcp specialist preventive prescription generic brand tier"
        )),
        _page(3, SERVICE_PAGE),
        _page(4, EXAMPLE_PAGE),
    ]
    selected = select_pages(pages, "health", top_n=2)
    assert 3 in selected, f"service page not selected: {selected}"
    assert 4 not in selected, f"coverage-example page selected: {selected}"


def test_page_one_always_included():
    pages = [
        _page(1, FILLER),
        _page(2, SERVICE_PAGE),
        _page(3, SERVICE_PAGE + " deductible"),
    ]
    selected = select_pages(pages, "health", top_n=2)
    assert 1 in selected


def test_non_health_categories_have_no_example_penalty():
    # The same pages scored for dental must not apply the health-only penalty:
    # the example page still wins on raw keyword count.
    pages = [
        _page(1, "cleaning exam x-ray filling crown deductible"),
        _page(2, EXAMPLE_PAGE),
    ]
    selected = select_pages(pages, "dental", top_n=2)
    assert selected == [1, 2]
