"""Test the exact payload our extract_fields() sends, progressively."""
import base64, json, os
from pathlib import Path
from dotenv import load_dotenv
import fitz, httpx

load_dotenv()

PROXY = os.getenv("DOCLING_PROXY_URL")
VLM_BASE = os.getenv("VLM_BASE_URL", "http://129.212.178.134")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
ENDPOINT = f"{VLM_BASE}/v1/chat/completions"
TIMEOUT = httpx.Timeout(connect=10, read=120, write=30, pool=5)

# Check max model length
print("=== Model info ===")
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
    r = c.get(f"{VLM_BASE}/v1/models")
    print(r.text[:800])
print()

# Render 5 pages of health/1.pdf
pdf_path = Path("tests/documents/health/1.pdf")
doc = fitz.open(str(pdf_path))
scale = fitz.Matrix(150/72, 150/72)
images = []
for i in range(min(5, doc.page_count)):
    pix = doc[i].get_pixmap(matrix=scale, alpha=False)
    jpeg_bytes = pix.tobytes("jpeg", jpg_quality=85)
    images.append(base64.b64encode(jpeg_bytes).decode())
    print(f"  Page {i+1}: {len(jpeg_bytes)//1024} KB JPEG")
doc.close()
print(f"  Total base64 chars: {sum(len(x) for x in images):,}")
print()

# --- Test 1: 5 images, simple text prompt, no schema ---
print("=== Test 1: 5 images, simple prompt (no schema) ===")
content_parts = [{"type": "text", "text": "What insurance carrier and plan name is shown in these pages?"}]
for img in images:
    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

payload = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": content_parts}],
    "temperature": 0.0,
    "max_tokens": 100,
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
    resp = c.post(ENDPOINT, json=payload)
print(f"HTTP {resp.status_code}")
try:
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(f"Content: {content!r}")
    print(f"Usage: {usage}")
except Exception as e:
    print(f"Error: {e}\nRaw: {resp.text[:500]}")
print()

# --- Test 2: 5 images + exact schema + no max_tokens (mirrors our real call) ---
print("=== Test 2: 5 images + full extraction schema (mirrors real call) ===")
import sys; sys.path.insert(0, ".")
from app.vlm_client import _build_extraction_schema, _build_messages
from app.schemas import CATEGORY_FIELDS

field_names = CATEGORY_FIELDS["health"]
schema = _build_extraction_schema(field_names)
print(f"  Schema size: {len(json.dumps(schema))} chars, {len(field_names)} fields")

# Use empty page markdowns to isolate image issue
page_markdowns = [{"page_number": i+1, "markdown": ""} for i in range(5)]
page_images_fmt = [{"page_number": i+1, "image_b64": img, "mime_type": "image/jpeg"} for i, img in enumerate(images)]
messages = _build_messages("health", field_names, page_markdowns, page_images_fmt)

# Count approximate chars in payload
payload2 = {
    "model": VLM_MODEL,
    "messages": messages,
    "temperature": 0.0,
    "guided_json": schema,
}
print(f"  Payload size: {len(json.dumps(payload2)):,} chars")

with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
    resp = c.post(ENDPOINT, json=payload2)
print(f"HTTP {resp.status_code}")
try:
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(f"Content (first 300): {content[:300]!r}")
    print(f"Usage: {usage}")
except Exception as e:
    print(f"Error: {e}\nRaw: {resp.text[:600]}")
