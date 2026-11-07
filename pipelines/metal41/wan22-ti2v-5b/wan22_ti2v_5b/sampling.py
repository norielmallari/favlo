"""Sampling card. Registration loads this file by path.

A video request does not read these numbers. The row still needs one variant.
"""

from __future__ import annotations

VARIANTS = {
    "default": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    }
}
DEFAULT_VARIANT = "default"
