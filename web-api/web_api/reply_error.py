"""Plain language for a failed chat reply."""

from __future__ import annotations

import re

_DETAIL = re.compile(r"""['"]message['"]\s*:\s*['"]([^'"]+)['"]""")
_SMOKE = "the gpu is held for a pipeline smoke"


def spoken_error(raw: str) -> str:
    """A sentence a person can read. A class name and a traceback stay out."""
    text = raw.strip()
    if not text:
        return "The listener did not complete the reply. Send again."
    if _SMOKE in text.lower():
        return "The GPU is held for a pipeline smoke. Send again when it is free."
    if not _dumped(text):
        return text
    found = _DETAIL.search(text)
    detail = found.group(1).strip() if found else ""
    if detail == _SMOKE:
        return "The GPU is held for a pipeline smoke. Send again when it is free."
    if not detail:
        return "The listener did not complete the reply. Send again."
    sentence = detail[:1].upper() + detail[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return f"{sentence} Send again."


def _dumped(text: str) -> bool:
    return (
        "service failed" in text
        or "<class " in text
        or text.startswith("(")
        or "Error code:" in text
    )
