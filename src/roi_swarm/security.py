from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException


def provider_api_key() -> str:
    return os.getenv("ROI_SWARM_API_KEY", "").strip()


def require_provider_auth(authorization: str | None = Header(default=None)) -> None:
    expected = provider_api_key()
    if not expected:
        return
    supplied = authorization or ""
    if not hmac.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
