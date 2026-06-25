from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DOCLING_SERVICE_URL: str = "http://129.212.178.134:8001"
    # When true, route docling-service and VLM through DOCLING_PROXY_URL (dev / non-allowlisted hosts).
    # Set false in production when this host can reach services directly.
    USE_OUTBOUND_PROXY: bool = False
    # Full proxy URL including credentials if needed: http://user:pass@host:port
    # Only used when USE_OUTBOUND_PROXY=true. Never commit a real value.
    DOCLING_PROXY_URL: str | None = None
    # Which docling endpoint to use (/convert or /convert-gpu)
    DOCLING_ENDPOINT: str = "/convert-gpu"
    DOCLING_OCR_MODE: str = "auto"
    DOCLING_TABLE_MODE: str = "accurate"

    # SECURITY PLACEHOLDER: comma-separated plaintext keys.
    # Replace with a real key store (hashed comparison, per-key rate limiting).
    VALID_API_KEYS: str = "eb3-key-1,eb3-key-2"

    PAGE_ROUTING_TOP_N: int = 5
    REQUEST_TIMEOUT_SECONDS: int = 120
    MAX_FILE_SIZE_MB: int = 50

    VLM_BASE_URL: str = "http://129.212.178.134"
    VLM_MODEL: str = "default-vlm-model"

    @property
    def outbound_proxy_url(self) -> str | None:
        if not self.USE_OUTBOUND_PROXY:
            return None
        return self.DOCLING_PROXY_URL

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.VALID_API_KEYS.split(",") if k.strip()}

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
