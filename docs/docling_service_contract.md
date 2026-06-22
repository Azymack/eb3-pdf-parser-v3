# docling-service API Contract

Confirmed live from `GET http://129.212.178.134:8001/openapi.json` on 2026-06-21.

## Base URL

`http://129.212.178.134:8001`

**Network note:** This host is behind an IP allowlist. This orchestrator must route all
requests through the HTTP proxy specified by `DOCLING_PROXY_URL` (see `.env.example`).

## Endpoints

### POST /convert-gpu  ← used by this orchestrator

GPU-accelerated PDF conversion. Returns the same response shape as `/convert` (CPU),
with two additional GPU-specific fields.

**Request:** `multipart/form-data`

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `file` | binary | yes | — | PDF file bytes |
| `ocr_mode` | string | no | `"auto"` | `"auto"` \| `"force"` \| `"off"` |
| `table_mode` | string | no | `"fast"` | `"fast"` \| `"accurate"` |

This orchestrator calls with `ocr_mode=auto`, `table_mode=accurate` (chosen over `fast`
based on benchmarking; exposed as env vars `DOCLING_OCR_MODE` / `DOCLING_TABLE_MODE`).

**Response (200):** `ConvertResponse`

```json
{
  "status": "string",
  "filename": "string",
  "page_count": 0,
  "ocr_mode_used": "string",
  "processing_time_seconds": 0.0,
  "pages": [
    {
      "page_number": 1,
      "markdown": "string",
      "had_ocr": false
    }
  ],
  "full_markdown": "string",
  "structured_json": {},
  "device_used": {"layout": "cuda:0", ...},
  "gpu_peak_memory_mb": 0.0
}
```

`page_number` is **1-based**.

### POST /convert

CPU variant. Identical request/response shape minus `device_used` and `gpu_peak_memory_mb`.

### GET /health

```json
{
  "status": "ready",
  "converter_ready": true,
  "in_flight": 0,
  "max_concurrent": 4,
  "gpu": {
    "device": "cuda:0",
    "converter_ready": true,
    "in_flight": 0,
    "max_concurrent": 1,
    "device_mapping": {"layout": "cuda:0", "table_structure": "cuda:0", "ocr": "cuda:0"}
  }
}
```

### GET /metrics

```json
{
  "total_requests": 0,
  "total_failures": 0,
  "average_processing_time_seconds": 0.0,
  "current_in_flight": 0
}
```

## Key facts for integration

- `pages` is always returned (even on errors that surface as 422).
- `page_number` starts at **1**, not 0.
- `table_mode` defaults to `"fast"` upstream; this orchestrator overrides to `"accurate"`.
- If GPU is unavailable, `/convert-gpu` falls back gracefully and reports it in `device_used`.
- 422 is returned for malformed multipart (e.g., wrong field name) — not 400.
