"""Registry card. The DiT schedule does not read these numbers."""

from __future__ import annotations

CONTEXT_LENGTH = 2048

VARIANTS = {
    "default": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "presence_penalty": 0.0,
    }
}

DEFAULT_VARIANT = "default"
