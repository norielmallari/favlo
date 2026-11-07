"""Sampling card for Z-Image-Turbo.

Registration copies this module. Diffusion does not read the card. The row
exists because the registry requires one variant. ``CONTEXT_LENGTH`` is the
RoPE temporal limit from the transformer config, not the tokenizer file's
131072.
"""

from typing import Any

DEFAULT_VARIANT = "generate"
CONTEXT_LENGTH = 1536

VARIANTS = {
    "generate": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    },
}


def requested_thinking(body: dict[str, Any]) -> bool | None:
    """Diffusion has no thinking flag. The registry still imports the name."""
    del body
    return None
