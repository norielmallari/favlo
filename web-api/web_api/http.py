"""HTTP helpers shared by the route modules."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def key(request: Request) -> str | None:
    value = request.headers.get("idempotency-key")
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def json_body(status: int, payload: dict):
    return JSONResponse(status_code=status, content=payload)
