"""Sampling card. Registration loads this file by path."""

VARIANTS = {
    "generate": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "presence_penalty": 0.0,
    }
}
DEFAULT_VARIANT = "generate"
DEFAULT_THINKING = False
CONTEXT_LENGTH = 512
