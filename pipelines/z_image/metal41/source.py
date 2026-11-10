"""Resumable download of one file from a Hugging Face resolve URL."""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path


def fetch(base: str, folder: Path, rel: str, expected: int | None) -> None:
    """Write ``folder/rel``. A partial file resumes from its size.

    ``base`` ends with ``/``. ``expected`` is the byte size, or ``None`` for a
    small file that is kept once it is non-empty.
    """
    path = folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if expected is not None and path.is_file() and path.stat().st_size == expected:
        return
    if expected is None and path.is_file() and path.stat().st_size > 0:
        return
    partial = Path(str(path) + ".partial")
    have = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "favlo-setup"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(base + rel, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(getattr(response, "status", 200))
        if status == 200 and have:
            have = 0
        mode = "ab" if status == 206 and have else "wb"
        written = have if mode == "ab" else 0
        last = time.time()
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(8 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.time()
                if expected and now - last > 30:
                    print(f"setup {rel} {written}/{expected}", flush=True)
                    last = now
    if expected is not None and partial.stat().st_size != expected:
        message = f"{rel} size {partial.stat().st_size} != {expected}"
        raise RuntimeError(message)
    os.replace(partial, path)
