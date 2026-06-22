# eb3-pdf-parser-v2

Orchestrator service for insurance PDF extraction.  
Receives a PDF + category, runs it through the full pipeline, and returns structured JSON.

## API

### POST /extract_json_v2

**Headers**

| Header | Required | Notes |
|--------|----------|-------|
| `X-EB3-Token` | Yes | See [Security note](#security-note-x-eb3-token) |

**Body** — `multipart/form-data`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | binary | Yes | PDF document, max 50 MB (configurable) |
| `category` | string | Yes | One of: `accident`, `critical_illness`, `dental`, `health`, `health_3tier`, `ltd`, `std`, `sup_life`, `term_life`, `vision` |

**Response 200**

```json
{
  "fields": { "Carrier Name": "Acme", "Plan Name": "Gold PPO", ... },
  "low_confidence_fields": ["Plan Name"],
  "pages_used": [1, 3, 5],
  "processing_time_seconds": 12.3,
  "stage_timings": {
    "docling_seconds": 8.1,
    "page_routing_seconds": 0.001,
    "image_rendering_seconds": 0.4,
    "vlm_extraction_seconds": 3.8
  }
}
```

**Error codes**

| Code | Meaning |
|------|---------|
| 400 | Invalid category, empty file, or file is not a PDF |
| 401 | Missing or invalid `X-EB3-Token` |
| 413 | File exceeds `MAX_FILE_SIZE_MB` (default 50 MB) |
| 502 | docling-service or VLM unreachable/returned an error — detail field names which stage failed |
| 504 | Total pipeline exceeded `REQUEST_TIMEOUT_SECONDS` (default 45 s) |

## Pipeline stages

```
POST /extract_json_v2
  │
  ├─ [1] docling-service  — PDF → per-page markdown + structured data
  ├─ [2] page router      — keyword scoring, select top-N pages
  ├─ [3] image renderer   — PyMuPDF renders selected pages to PNG
  └─ [4] VLM (vLLM)       — guided_json extraction against category schema
```

`stage_timings` in every response shows the wall-clock cost of each stage,
making it easy to identify the bottleneck without guessing.

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

| Variable | Default | Notes |
|----------|---------|-------|
| `DOCLING_PROXY_URL` | *(none)* | **Required at deploy time.** See below. |
| `DOCLING_SERVICE_URL` | `http://129.212.178.134:8001` | |
| `DOCLING_ENDPOINT` | `/convert-gpu` | Switch to `/convert` for CPU-only |
| `DOCLING_OCR_MODE` | `auto` | `auto` \| `force` \| `off` |
| `DOCLING_TABLE_MODE` | `accurate` | `accurate` \| `fast` |
| `VLM_BASE_URL` | `http://localhost:8000` | vLLM server |
| `VLM_MODEL` | `default-vlm-model` | Model identifier passed to vLLM |
| `PAGE_ROUTING_TOP_N` | `5` | Number of pages sent to VLM |
| `REQUEST_TIMEOUT_SECONDS` | `45` | Total pipeline timeout |
| `MAX_FILE_SIZE_MB` | `50` | Upload size cap |
| `VALID_API_KEYS` | `eb3-key-1,eb3-key-2` | Comma-separated key allowlist |

## DOCLING_PROXY_URL — deployment requirement

docling-service at `http://129.212.178.134:8001` is behind an IP allowlist.
This orchestrator is **not** on that allowlist and must connect via an HTTP proxy.

**You must set `DOCLING_PROXY_URL` in the deployment environment** (via `.env`,
your CI/CD secrets store, or however your team manages secrets).  
Whoever deploys this service needs to obtain the proxy URL out-of-band
(direct communication or secrets manager) — it should never appear in tickets,
chat logs, or source control.

Do not add the value to `.env.example` or commit it anywhere.

## Security note — X-EB3-Token

The current implementation checks the incoming token against a short comma-separated
allowlist of **plaintext keys** (`VALID_API_KEYS`). This is a deliberate placeholder:

- No hashing or constant-time comparison
- No per-key identity or rate limiting
- Keys travel in HTTP headers (use TLS in production)

Before handling sensitive documents in production, replace this with a proper
API-key system: database-backed key store, Argon2/bcrypt hashed comparison,
per-key rate limits, and key rotation support. The `verify_token` dependency in
`app/auth.py` is the single place to swap this out.

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

## Running tests

```bash
pytest tests/ -v
```

Tests mock all external services (docling, VLM, PyMuPDF) — no network or GPU required.
