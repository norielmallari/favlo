"""Sampling card for Qwen3.8-27B.

Registration copies this module onto the registry row. A request that names
no variant uses ``DEFAULT_VARIANT``.
"""

DEFAULT_VARIANT = "thinking"

VARIANTS = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    },
}
