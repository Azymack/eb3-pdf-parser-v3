from app.config import Settings


def test_outbound_proxy_disabled_ignores_url():
    settings = Settings(USE_OUTBOUND_PROXY=False, DOCLING_PROXY_URL="http://proxy:8080")
    assert settings.outbound_proxy_url is None


def test_outbound_proxy_enabled_returns_url():
    settings = Settings(USE_OUTBOUND_PROXY=True, DOCLING_PROXY_URL="http://proxy:8080")
    assert settings.outbound_proxy_url == "http://proxy:8080"


def test_outbound_proxy_enabled_without_url_returns_none():
    settings = Settings(USE_OUTBOUND_PROXY=True, DOCLING_PROXY_URL=None)
    assert settings.outbound_proxy_url is None
