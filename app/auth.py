from fastapi import Header, HTTPException

from .config import get_settings

# SECURITY PLACEHOLDER: X-EB3-Token is checked against a short allowlist of
# hardcoded plaintext keys. This is NOT real auth — no hashing, no per-key
# identity, no rate limiting. Replace with a proper key store (DB-backed,
# bcrypt/Argon2 comparison, per-key rate limits) before handling sensitive data
# in production. See README for details.
async def verify_token(x_eb3_token: str | None = Header(default=None)) -> str:
    settings = get_settings()
    if not x_eb3_token or x_eb3_token not in settings.api_key_set:
        raise HTTPException(status_code=401, detail="Missing or invalid X-EB3-Token")
    return x_eb3_token
