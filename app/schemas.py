import json
from pathlib import Path

from pydantic import BaseModel

_CATEGORY_KEYS_PATH = Path(__file__).parent / "category_keys.json"

# API category names (underscored) → keys in category_keys.json (may have spaces)
_API_TO_JSON_KEY: dict[str, str] = {
    "dental": "dental",
    "vision": "vision",
    "term_life": "term life",
    "std": "std",
    "ltd": "ltd",
    "accident": "accident",
    "critical_illness": "critical illness",
    "sup_life": "sup life",
    "health": "health",
    "health_3tier": "health_3tier",
}

VALID_CATEGORIES: set[str] = set(_API_TO_JSON_KEY.keys())
VALID_CATEGORIES_SORTED: list[str] = sorted(VALID_CATEGORIES)

_raw = json.loads(_CATEGORY_KEYS_PATH.read_text())
CATEGORY_FIELDS: dict[str, list[str]] = {
    api_key: _raw[json_key]
    for api_key, json_key in _API_TO_JSON_KEY.items()
}


class StageTimings(BaseModel):
    docling_seconds: float
    page_routing_seconds: float
    image_rendering_seconds: float
    vlm_extraction_seconds: float


class ExtractionResponse(BaseModel):
    # fields has null values pre-converted to ""; see _nulls_to_empty in main.py.
    fields: dict[str, str]
    low_confidence_fields: list[str]
    pages_used: list[int]
    processing_time_seconds: float
    stage_timings: StageTimings


class LlmTestResponse(BaseModel):
    response: str
