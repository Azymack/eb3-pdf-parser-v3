from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DOCLING_SERVICE_URL: str = "http://129.212.178.134:8001"
    # Which docling endpoint to use (/convert or /convert-gpu)
    DOCLING_ENDPOINT: str = "/convert-gpu"
    DOCLING_OCR_MODE: str = "auto"
    DOCLING_TABLE_MODE: str = "accurate"

    # SECURITY PLACEHOLDER: comma-separated plaintext keys.
    # Replace with a real key store (hashed comparison, per-key rate limiting).
    VALID_API_KEYS: str = "eb3-key-1,eb3-key-2"

    PAGE_ROUTING_TOP_N: int = 8
    # Cap PDF length before docling/OCR/VLM so large uploads do not overload the pipeline.
    # Set to 0 to disable truncation.
    MAX_PDF_PAGES: int = 13
    REQUEST_TIMEOUT_SECONDS: int = 120
    MAX_FILE_SIZE_MB: int = 50

    VLM_BASE_URL: str = "http://129.212.178.134"
    VLM_MODEL: str = "default-vlm-model"

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
