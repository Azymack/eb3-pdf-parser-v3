"""Tests for POST /extract_json_v2.

All external calls (docling-service, VLM, PyMuPDF) are mocked so these tests
never hit real services and run without GPU/network.
"""
import pytest
import httpx
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
    },
    "low_confidence_fields": ["Plan Name"],
}

# Fake rendered images (page_router will select pages 1 and 2)
FAKE_IMAGES = [
    {"page_number": 1, "image_b64": "aaa=", "width": 800, "height": 1000},
    {"page_number": 2, "image_b64": "bbb=", "width": 800, "height": 1000},
]


@pytest.fixture
def mock_pipeline():
    """Patch all three external-touching functions."""
    with (
        patch("app.main.convert_pdf", new_callable=AsyncMock) as mock_docling,
        patch("app.main.extract_fields", new_callable=AsyncMock) as mock_vlm,
        patch("app.main.render_pages", return_value=FAKE_IMAGES) as mock_render,
    ):
        mock_docling.return_value = DOCLING_RESPONSE
        mock_vlm.return_value = VLM_RESPONSE
        yield mock_docling, mock_vlm, mock_render


@pytest.mark.asyncio
async def test_success_returns_correct_shape(mock_pipeline):
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

    # Shape assertions
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

    # All stage times must be non-negative numbers
    for key, val in timings.items():
        assert isinstance(val, float | int), f"{key} should be numeric"
        assert val >= 0

    mock_docling.assert_awaited_once()
    mock_vlm.assert_awaited_once()
    mock_render.assert_called_once()


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
async def test_proxy_failure_returns_502(mock_pipeline):
    mock_docling, mock_vlm, mock_render = mock_pipeline
    mock_docling.side_effect = httpx.ProxyError("proxy connection failed")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 502
    assert "proxy" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_vlm_proxy_failure_returns_502(mock_pipeline):
    mock_docling, mock_vlm, mock_render = mock_pipeline
    mock_vlm.side_effect = httpx.ProxyError("proxy connection failed")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/extract_json_v2",
            headers=HEADERS_OK,
            files={"file": ("plan.pdf", FAKE_PDF, "application/pdf")},
            data={"category": "dental"},
        )

    assert response.status_code == 502
    assert "proxy" in response.json()["detail"].lower()


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
