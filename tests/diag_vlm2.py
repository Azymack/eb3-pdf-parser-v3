"""Diagnose the extraction-specific VLM failure: test with real page image."""
import base64, json, os, sys
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

PROXY = os.getenv("DOCLING_PROXY_URL")
VLM_BASE = os.getenv("VLM_BASE_URL", "http://129.212.178.134")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
ENDPOINT = f"{VLM_BASE}/v1/chat/completions"
TIMEOUT = httpx.Timeout(connect=10, read=90, write=30, pool=5)

# Render page 1 of health/1.pdf at 150 DPI JPEG
import fitz
pdf_path = Path("tests/documents/health/1.pdf")
doc = fitz.open(str(pdf_path))
scale = fitz.Matrix(150/72, 150/72)
pix = doc[0].get_pixmap(matrix=scale, alpha=False)
jpeg_bytes = pix.tobytes("jpeg", jpg_quality=85)
doc.close()
img_b64 = base64.b64encode(jpeg_bytes).decode()
print(f"Image size: {len(jpeg_bytes)//1024} KB JPEG  |  base64 length: {len(img_b64)} chars")
print()

# --- Test 1: image only (minimal prompt) ---
print("=== Test 1: single image, minimal prompt ===")
payload = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this page in one sentence."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
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

# --- Test 2: same but with a small JSON schema ---
print("=== Test 2: image + small extraction schema ===")
schema = {
    "type": "object",
    "properties": {
        "carrier_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "plan_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["carrier_name", "plan_name"],
    "additionalProperties": False,
}
payload2 = {
    "model": VLM_MODEL,
    "messages": [
        {"role": "system", "content": "Extract fields from this insurance document. Return JSON only."},
        {"role": "user", "content": [
            {"type": "text", "text": "Extract carrier_name and plan_name from this page."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]},
    ],
    "temperature": 0.0,
    "max_tokens": 200,
    "guided_json": schema,
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
    resp = c.post(ENDPOINT, json=payload2)
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

# --- Test 3: check how many tokens one image costs ---
print("=== Test 3: token count for image ===")
payload3 = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        {"type": "text", "text": "x"},
    ]}],
    "temperature": 0.0,
    "max_tokens": 1,
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
    resp = c.post(ENDPOINT, json=payload3)
print(f"HTTP {resp.status_code}")
try:
    body = resp.json()
    usage = body.get("usage", {})
    content = body["choices"][0]["message"]["content"]
    print(f"Usage: {usage}")
    print(f"Content: {content!r}")
except Exception as e:
    print(f"Error: {e}\nRaw: {resp.text[:500]}")
