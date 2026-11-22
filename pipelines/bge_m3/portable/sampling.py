"""Sampling card for BGE-M3.

Registration copies this module. An embedding request does not read the card.
The row exists because the registry requires one variant. ``CONTEXT_LENGTH``
is the longest function quantize builds, not the checkpoint's 8192. The open
record serves the longest function it contains.
"""

from typing import Any

DEFAULT_VARIANT = "embed"
CONTEXT_LENGTH = 4096

VARIANTS = {
    "embed": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    },
}


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """Embeddings have no thinking flag. The registry still imports the name."""
    del body
    return None
