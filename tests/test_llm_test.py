"""Tests for POST /llm_test."""
import pytest
import httpx
from httpx import Response as HttpxResponse
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import app

VALID_TOKEN = "eb3-key-1"
HEADERS_OK = {"X-EB3-Token": VALID_TOKEN}


@pytest.fixture
def mock_complete_prompt():
    with patch("app.main.complete_prompt", new_callable=AsyncMock) as mock:
        mock.return_value = "Hello from the model"
        yield mock


@pytest.mark.asyncio
async def test_llm_test_success(mock_complete_prompt):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/llm_test",
            headers=HEADERS_OK,
            data={"prompt": "Say hello"},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the model"}
    mock_complete_prompt.assert_awaited_once_with("Say hello")


@pytest.mark.asyncio
async def test_llm_test_empty_prompt_returns_400():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/llm_test",
            headers=HEADERS_OK,
            data={"prompt": "   "},
        )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_llm_test_missing_token_returns_401(mock_complete_prompt):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/llm_test", data={"prompt": "hi"})

    assert response.status_code == 401
    mock_complete_prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_test_vlm_unreachable_returns_502(mock_complete_prompt):
    mock_complete_prompt.side_effect = httpx.ConnectError("connection refused")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/llm_test",
            headers=HEADERS_OK,
            data={"prompt": "hi"},
        )

    assert response.status_code == 502
    assert "VLM" in response.json()["detail"]


@pytest.mark.asyncio
async def test_llm_test_nginx_413_surfaces_body(mock_complete_prompt):
    """When the VLM's nginx returns 413, the 502 detail includes the status and body."""
    nginx_413_body = (
        b"<html>\r\n<head><title>413 Request Entity Too Large</title></head>\r\n"
        b"<body>\r\n<center><h1>413 Request Entity Too Large</h1></center>\r\n"
        b"<hr><center>nginx/1.18.0 (Ubuntu)</center>\r\n</body>\r\n</html>"
    )
    mock_complete_prompt.side_effect = httpx.HTTPStatusError(
        "413",
        request=httpx.Request("POST", "http://vlm/v1/chat/completions"),
        response=HttpxResponse(413, content=nginx_413_body),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/llm_test",
            headers=HEADERS_OK,
            data={"prompt": "Say hello"},
        )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "413" in detail
    assert "nginx" in detail.lower()
