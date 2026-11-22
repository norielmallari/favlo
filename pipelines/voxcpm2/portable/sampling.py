"""Sampling card for VoxCPM2.

Registration copies this module. Speech reads the checkpoint's flow-matching
card, not these fields. The row exists because the registry requires one variant.
"""

from typing import Any

DEFAULT_VARIANT = "voice"
CONTEXT_LENGTH = 8192

VARIANTS = {
    "voice": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    },
}


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """Speech has no thinking flag. The registry still imports the name."""
    del body
    return None
