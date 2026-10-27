"""Prefixed identifiers for application records."""

from __future__ import annotations

import uuid


def mint(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"
