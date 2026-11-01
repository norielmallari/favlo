"""Sampling card from the checkpoint ``generation_config.json``.

The three names change the chat template. They share one sampler.
``top_k`` 0 leaves the nucleus uncut. The checkpoint stores that as ``-1``.
"""

from __future__ import annotations

CONTEXT_LENGTH = 1_048_576

_ROW = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 0,
    "presence_penalty": 0.0,
}

VARIANTS = {
    "high": dict(_ROW),
    "low": dict(_ROW),
    "none": dict(_ROW),
}

DEFAULT_VARIANT = "high"
