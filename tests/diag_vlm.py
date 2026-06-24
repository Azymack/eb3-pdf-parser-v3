"""Diagnostic: call vLLM directly to isolate the empty-content issue."""
import json
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

PROXY = os.getenv("DOCLING_PROXY_URL")
VLM_BASE = os.getenv("VLM_BASE_URL", "http://129.212.178.134")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
ENDPOINT = f"{VLM_BASE}/v1/chat/completions"

print(f"VLM endpoint : {ENDPOINT}")
print(f"Model        : {VLM_MODEL}")
print(f"Via proxy    : {'yes' if PROXY else 'no'}")
print()

TIMEOUT = httpx.Timeout(connect=10, read=60, write=10, pool=5)

# --- Test 1: plain text, no guided_json ---
print("=== Test 1: plain text prompt, no guided_json ===")
payload1 = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": "Reply with just the word HELLO."}],
    "temperature": 0.0,
    "max_tokens": 10,
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as client:
    resp = client.post(ENDPOINT, json=payload1)
print(f"HTTP status: {resp.status_code}")
try:
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    print(f"Content: {content!r}")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw body: {resp.text[:500]}")

print()

# --- Test 2: same prompt but with guided_json at TOP LEVEL (correct for vLLM) ---
print("=== Test 2: guided_json at top level ===")
schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False,
}
payload2 = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": 'Reply with JSON: {"answer": "HELLO"}'}],
    "temperature": 0.0,
    "max_tokens": 50,
    "guided_json": schema,
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as client:
    resp = client.post(ENDPOINT, json=payload2)
print(f"HTTP status: {resp.status_code}")
try:
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    print(f"Content: {content!r}")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw body: {resp.text[:500]}")

print()

# --- Test 3: guided_json nested under extra_body (how we currently send it - WRONG) ---
print("=== Test 3: guided_json under extra_body (current broken code) ===")
payload3 = {
    "model": VLM_MODEL,
    "messages": [{"role": "user", "content": 'Reply with JSON: {"answer": "HELLO"}'}],
    "temperature": 0.0,
    "max_tokens": 50,
    "extra_body": {"guided_json": schema},
}
with httpx.Client(proxy=PROXY, timeout=TIMEOUT) as client:
    resp = client.post(ENDPOINT, json=payload3)
print(f"HTTP status: {resp.status_code}")
try:
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    print(f"Content: {content!r}")
except Exception as e:
    print(f"Parse error: {e}")
    print(f"Raw body: {resp.text[:500]}")
