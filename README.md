# eb3-pdf-parser-v3

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

**Query parameters**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `include_metadata` | boolean | `false` | When `true`, wraps the response in a metadata envelope (see below) |

---

#### Default response (fields object directly)

By default the response body **is** the extracted fields object — a flat JSON object
where each key is a field name and each value is a string.  Fields not found on the
plan are returned as `""` (empty string).

```json
{
  "Carrier Name": "Acme",
  "Plan Name": "Gold PPO",
  "In-Network Single Deductible": "$500",
  "Out-of-Network Single Deductible": "",
  ...
}
```

#### With `?include_metadata=true`

Returns the full metadata envelope:

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

> **Breaking change (v2.1):** Prior to this version the default response was the
> full metadata envelope. If you were parsing `response["fields"]` from the default
> response, either add `?include_metadata=true` to your request, or update your
> parser to treat the response body as the fields object directly.

---

**Error codes**

| Code | Meaning |
|------|---------|
| 400 | Invalid category, empty file, or file is not a PDF |
| 401 | Missing or invalid `X-EB3-Token` |
| 413 | File exceeds `MAX_FILE_SIZE_MB` (default 50 MB) |
| 502 | docling-service or VLM unreachable/returned an error — detail field names which stage failed |
| 504 | Total pipeline exceeded `REQUEST_TIMEOUT_SECONDS` (default 120 s) |

### POST /llm_test

Sends a plain-text prompt to the VLM and returns the raw reply. Useful for
verifying connectivity and model behaviour without running a full PDF pipeline.

**Body** — `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `prompt` | string | Yes |

**Response 200**

```json
{ "response": "Hello from the model" }
```

## Pipeline stages

```
POST /extract_json_v2
  │
  ├─ [1] docling-service  — PDF → per-page markdown + structured data
  ├─ [2] page router      — keyword scoring, select top-N pages
  ├─ [3] image renderer   — PyMuPDF renders selected pages to JPEG
  └─ [4] VLM (vLLM)       — guided_json extraction against category schema
  └─ [*] post-processing  — compute derived fields (e.g. combined Mail Order RX)
```

`stage_timings` (available with `?include_metadata=true`) shows the wall-clock
cost of each stage, making it easy to identify the bottleneck.

### Field computation — Mail Order RX

For the `health` and `health_3tier` categories, the following combined fields
are **computed** from individual per-tier values rather than extracted directly:

- `In-Network Mail Order RX` = join of Generic / Brand / Tier 3 / Tier 4 / Tier 5 Mail Order RX values
- `Out-of-Network Mail Order RX` — same pattern
- `Designated Network Mail Order RX` — same (health_3tier only)

Format: non-empty tier values joined by `" / "` in tier order.
Example: `"$10 / $30 / $30"` (Generic=$10, Brand=$30, Tier3=$30, Tier4 and Tier5 absent).

If all tier values are absent the combined field is `""`.

## Configuration

Copy `.env.example` to `.env` and fill in the required values.

| Variable | Default | Notes |
|----------|---------|-------|
| `USE_OUTBOUND_PROXY` | `false` | Set `true` to route docling and VLM through the proxy below |
| `DOCLING_PROXY_URL` | *(none)* | Proxy URL — only used when `USE_OUTBOUND_PROXY=true`. See below. |
| `DOCLING_SERVICE_URL` | `http://129.212.178.134:8001` | |
| `DOCLING_ENDPOINT` | `/convert-gpu` | Switch to `/convert` for CPU-only |
| `DOCLING_OCR_MODE` | `auto` | `auto` \| `force` \| `off` |
| `DOCLING_TABLE_MODE` | `accurate` | `accurate` \| `fast` |
| `VLM_BASE_URL` | `http://localhost:8000` | vLLM server |
| `VLM_MODEL` | `default-vlm-model` | Model identifier passed to vLLM |
| `PAGE_ROUTING_TOP_N` | `5` | Number of pages sent to VLM |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Total pipeline timeout |
| `MAX_FILE_SIZE_MB` | `50` | Upload size cap |
| `VALID_API_KEYS` | `eb3-key-1,eb3-key-2` | Comma-separated key allowlist |

## Outbound proxy (dev vs production)

docling-service and the VLM server may be behind an IP allowlist. Hosts **not** on that
allowlist must connect through an HTTP proxy.

| Environment | `.env` |
|-------------|--------|
| **Production** (direct access) | `USE_OUTBOUND_PROXY=false` — no proxy needed |
| **Local / staging** (not allowlisted) | `USE_OUTBOUND_PROXY=true` and set `DOCLING_PROXY_URL` |

Obtain the proxy URL out-of-band (secrets manager or direct handoff) — never commit it or
paste it into tickets or chat logs.

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

$env:RUN_LIVE_RX_TESTS=1; python -m pytest tests/test_rx_extraction.py -v
```

Tests mock all external services (docling, VLM, PyMuPDF) — no network or GPU required.
