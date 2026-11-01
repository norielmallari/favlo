"""Sampling card from the Qwen3.5-4B model card.

Thinking is the default. Instruct closes thinking in the template.
Precise-coding stays here for harness use and is not a registered suffix.
"""

from __future__ import annotations

CONTEXT_LENGTH = 262_144

VARIANTS = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
}

DEFAULT_VARIANT = "thinking"

PRECISE_CODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
}
