"""Tests for POST /extract_json_v2.

All external calls (docling-service, VLM, PyMuPDF) are mocked so these tests
never hit real services and run without GPU/network.
"""
import pytest
import httpx
from httpx import Response as HttpxResponse
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app

# Minimal fake PDF bytes (passes %PDF magic check)
FAKE_PDF = b"%PDF-1.4 fake content"

VALID_TOKEN = "eb3-key-1"
HEADERS_OK = {"X-EB3-Token": VALID_TOKEN}

# What a successful docling response looks like
DOCLING_RESPONSE = {
    "status": "ok",
    "filename": "test.pdf",
    "page_count": 2,
    "ocr_mode_used": "auto",
    "processing_time_seconds": 1.5,
    "full_markdown": "# Test Plan",
    "structured_json": {},
    "pages": [
        {"page_number": 1, "markdown": "Carrier: Acme\nDeductible $500", "had_ocr": False},
        {"page_number": 2, "markdown": "In-Network coinsurance 80%", "had_ocr": False},
    ],
}

# What the VLM returns (pre-parsed dict)
VLM_RESPONSE = {
    "fields": {
        "Carrier Name": "Acme Insurance",
        "Plan Name": "Gold PPO",
        "In-Network Single Deductible": "$500",
        "In-Network Family Deductible": "$1000",
        "In-Network RX": "Tier 1 (Generic): $10 / Tier 2 (Brand): $40 / Tier 3: 50%",
        "In-Network Mail Order RX": "Tier 1: $20 / Tier 2: $80",
    },
    "low_confidence_fields": ["Plan Name"],
}

# Fake rendered images (page_router will select pages 1 and 2)
FAKE_IMAGES = [
    {"page_number": 1, "image_b64": "aaa=", "width": 800, "height": 1000},
    {"page_number": 2, "image_b64": "bbb=", "width": 800, "height": 1000},
]

# What the structured RX extraction returns (already-assembled flat fields)
RX_FIELDS_RESPONSE = {
    "In-Network RX": "Tier 1 (Generic): $10 / Tier 2 (Brand): $40 / Tier 3: 50%",
    "In-Network Mail Order RX": "$20 / $80",
    "In-Network Generic RX": "$10",
    "In-Network Brand RX": "$40",
    "In-Network Tier 3 RX": "50%",
    "Out-of-Network RX": "Not covered",
}


@pytest.fixture
def mock_pipeline():
    """Patch all external-touching functions (docling, both VLM calls, renderer)."""
    with (
        patch("app.main.convert_pdf", new_callable=AsyncMock) as mock_docling,
        patch("app.main.extract_fields", new_callable=AsyncMock) as mock_vlm,
        patch("app.main.extract_rx_fields", new_callable=AsyncMock) as mock_rx,
        patch("app.main.render_pages", return_value=FAKE_IMAGES) as mock_render,
    ):
        mock_docling.return_value = DOCLING_RESPONSE
        mock_vlm.return_value = VLM_RESPONSE
        mock_rx.return_value = dict(RX_FIELDS_RESPONSE)
        yield mock_docling, mock_vlm, mock_render


@pytest.mark.asyncio
async def test_success_default_returns_fields_only(mock_pipeline):
    """Default response (no include_metadata) is the flat fields object, not wrapped."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 200
    body = response.json()

    # Default shape is the fields dict directly — no metadata wrapper keys
    assert "Carrier Name" in body
    assert body["Carrier Name"] == "Acme Insurance"
    assert "low_confidence_fields" not in body
    assert "pages_used" not in body
    assert "stage_timings" not in body
    assert "fields" not in body

    mock_docling.assert_awaited_once()
    mock_vlm.assert_awaited_once()
    mock_render.assert_called_once()


@pytest.mark.asyncio
async def test_success_include_metadata_returns_full_shape(mock_pipeline):
    """?include_metadata=true returns the full metadata envelope."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
            params={"include_metadata": "true"},
        )

    assert response.status_code == 200
    body = response.json()

    # Full metadata shape
    assert "fields" in body
    assert "low_confidence_fields" in body
    assert "pages_used" in body
    assert "processing_time_seconds" in body
    assert "stage_timings" in body

    timings = body["stage_timings"]
    assert set(timings.keys()) == {
        "docling_seconds",
        "page_routing_seconds",
        "image_rendering_seconds",
        "vlm_extraction_seconds",
    }

    assert body["fields"]["Carrier Name"] == "Acme Insurance"
    assert body["low_confidence_fields"] == ["Plan Name"]
    assert isinstance(body["pages_used"], list)
    assert 1 in body["pages_used"]

    for key, val in timings.items():
        assert isinstance(val, float | int), f"{key} should be numeric"
        assert val >= 0


@pytest.mark.asyncio
async def test_missing_token_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_returns_401():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers={"X-EB3-Token": "not-a-real-key"},
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_invalid_category_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "not_a_real_category"},
        )
    assert response.status_code == 400
    assert "not_a_real_category" in response.json()["detail"]
    # Response should list valid values
    assert "dental" in response.json()["detail"]


@pytest.mark.asyncio
async def test_docling_failure_returns_502(mock_pipeline):
    mock_docling, mock_vlm, mock_render = mock_pipeline
    mock_docling.side_effect = httpx.ConnectError("connection refused")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 502
    assert "docling-service" in response.json()["detail"].lower()



@pytest.mark.asyncio
async def test_vlm_invalid_json_returns_502(mock_pipeline):
    mock_docling, mock_vlm, mock_render = mock_pipeline
    mock_vlm.side_effect = ValueError("VLM returned unparseable content")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 502
    assert "VLM" in response.json()["detail"]


@pytest.mark.asyncio
async def test_non_pdf_file_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("doc.pdf", b"this is not a pdf", "application/pdf")},
            data={"category": "dental"},
        )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_vlm_nginx_413_returns_502_with_body(mock_pipeline):
    """When the VLM's nginx returns 413, we surface the status and body in the 502 detail."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    nginx_413_body = (
        b"<html>\r\n<head><title>413 Request Entity Too Large</title></head>\r\n"
        b"<body>\r\n<center><h1>413 Request Entity Too Large</h1></center>\r\n"
        b"<hr><center>nginx/1.18.0 (Ubuntu)</center>\r\n</body>\r\n</html>"
    )
    mock_vlm.side_effect = httpx.HTTPStatusError(
        "413",
        request=httpx.Request("POST", "http://vlm/v1/chat/completions"),
        response=HttpxResponse(413, content=nginx_413_body),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "413" in detail
    assert "nginx" in detail.lower()


@pytest.mark.asyncio
async def test_nested_dict_field_serialized_as_flat_string(mock_pipeline):
    """VLM sometimes returns nested objects for multi-tier fields (e.g. Root Canal costs
    per tooth type). These must be flattened to a string rather than causing a 500."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    mock_vlm.return_value = {
        "fields": {
            "Carrier Name": "Acme",
            # VLM returned a nested dict instead of a string — happens for multi-tier fields
            "In-Network Root Canal": {"Anterior": "$100", "Bicuspid": "$150", "Molar": "$200"},
        },
        "low_confidence_fields": [],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 200, f"Expected 200, got 500: {response.text[:300]}"
    body = response.json()
    root_canal = body["In-Network Root Canal"]
    assert isinstance(root_canal, str)
    assert "$100" in root_canal
    assert "$150" in root_canal
    assert "$200" in root_canal


@pytest.mark.asyncio
async def test_null_fields_serialized_as_empty_string(mock_pipeline):
    """VLM null values must become '' in the API response, never JSON null."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    # Simulate a VLM response where some fields are null
    mock_vlm.return_value = {
        "fields": {
            "Carrier Name": "Acme Insurance",
            "Plan Name": None,          # null — not on plan
            "In-Network Single Deductible": "NOT_FOUND",  # sentinel — should be preserved
        },
        "low_confidence_fields": [],
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["Plan Name"] == ""            # null → ""
    assert body["Carrier Name"] == "Acme Insurance"
    assert body["In-Network Single Deductible"] == "NOT_FOUND"  # sentinel preserved


@pytest.mark.asyncio
async def test_rx_tier_string_passed_through_to_response(mock_pipeline):
    """New consolidated RX field is returned verbatim in the flat response."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "health"},
        )
    assert response.status_code == 200
    body = response.json()
    # RX fields come from the structured RX extraction, passed through verbatim
    assert body["In-Network RX"] == "Tier 1 (Generic): $10 / Tier 2 (Brand): $40 / Tier 3: 50%"
    assert body["In-Network Mail Order RX"] == "$20 / $80"
    assert body["In-Network Generic RX"] == "$10"
    assert body["Out-of-Network RX"] == "Not covered"
    assert "In-Network Generic Mail Order RX" not in body
    assert "In-Network Brand Mail Order RX" not in body


@pytest.mark.asyncio
async def test_vlm_prompt_does_not_request_old_rx_fields(mock_pipeline):
    """VLM is called with the new schema — old per-tier field names are absent."""
    mock_docling, mock_vlm, mock_render = mock_pipeline
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "health"},
        )
    # extract_fields was called — the main call must not see ANY RX fields;
    # those belong to the structured RX extraction (rx_extractor.py).
    call_args = mock_vlm.call_args
    field_names_used = call_args.kwargs.get("field_names") or call_args.args[1]
    assert not [f for f in field_names_used if "RX" in f], (
        f"main VLM call still requests RX fields: "
        f"{[f for f in field_names_used if 'RX' in f]}"
    )
    assert "Carrier Name" in field_names_used


@pytest.mark.asyncio
async def test_all_valid_categories_accepted(mock_pipeline):
    """Every category in VALID_CATEGORIES_SORTED should return 200."""
    from app.schemas import VALID_CATEGORIES_SORTED

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for cat in VALID_CATEGORIES_SORTED:
            response = await client.post(
                "/extract_json_v2",
                headers=HEADERS_OK,
                files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
                data={"category": cat},
            )
            assert response.status_code == 200, f"Category '{cat}' failed: {response.text}"
