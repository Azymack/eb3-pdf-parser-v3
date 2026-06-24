"""Confirm: extra_body guided_json vs response_format vs plain prompt."""
import base64, json, os, sys
sys.path.insert(0, ".")
from pathlib import Path
from dotenv import load_dotenv
import fitz, httpx

load_dotenv()

PROXY = os.getenv("DOCLING_PROXY_URL")
VLM_BASE = os.getenv("VLM_BASE_URL", "http://129.212.178.134")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
ENDPOINT = f"{VLM_BASE}/v1/chat/completions"
TIMEOUT = httpx.Timeout(connect=10, read=120, write=30, pool=5)

from app.vlm_client import _build_extraction_schema
from app.schemas import CATEGORY_FIELDS

field_names = CATEGORY_FIELDS["health"]
schema = _build_extraction_schema(field_names)

# One image only (page 1) to keep it fast
pdf_path = Path("tests/documents/health/1.pdf")
doc = fitz.open(str(pdf_path))
pix = doc[0].get_pixmap(matrix=fitz.Matrix(150/72, 150/72), alpha=False)
img_b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode()
doc.close()

fields_list = "\n".join(f"- {f}" for f in field_names)
system_prompt = (
    "You are an insurance document extraction specialist. "
    "Extract the specified fields from the provided health insurance plan document. "
    "You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences. "
    "Use null for any field not present. List uncertain fields in low_confidence_fields."
)
user_text = (
    f"Extract all fields for this health insurance plan.\n\n"
    f"**Fields to extract:**\n{fields_list}\n\n"
    "Return ONLY raw JSON matching this structure: "
    '{"fields": {"Field Name": "value or null", ...}, "low_confidence_fields": ["Field Name", ...]}'
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]},
]


def call(label, payload):
    print(f"=== {label} ===")
    with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as c:
        resp = c.post(ENDPOINT, json=payload)
    print(f"HTTP {resp.status_code}")
    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        print(f"Content (200 chars): {(content or '')[:200]!r}")
        print(f"Usage: {usage}")
        if content:
            stripped = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                parsed = json.loads(stripped)
                print(f"JSON parse: OK  (fields count={len(parsed.get('fields', {}))})")
            except Exception as e:
                print(f"JSON parse: FAILED — {e}")
    except Exception as e:
        print(f"Response error: {e}\nRaw: {resp.text[:400]}")
    print()


# Test A: extra_body (current broken approach)
call("A: extra_body guided_json (current code)", {
    "model": VLM_MODEL, "messages": messages, "temperature": 0.0,
    "extra_body": {"guided_json": schema},
})

# Test B: guided_json at top level
call("B: guided_json at top level", {
    "model": VLM_MODEL, "messages": messages, "temperature": 0.0,
    "guided_json": schema,
})

# Test C: response_format json_object
call("C: response_format json_object", {
    "model": VLM_MODEL, "messages": messages, "temperature": 0.0,
    "response_format": {"type": "json_object"},
})

# Test D: plain prompt, no schema enforcement, just a strong system prompt
call("D: no schema enforcement, strong JSON-only prompt", {
    "model": VLM_MODEL, "messages": messages, "temperature": 0.0,
})
