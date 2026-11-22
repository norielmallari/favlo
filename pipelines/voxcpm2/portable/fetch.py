"""Download the VoxCPM2 checkout. A finished file is left in place."""

from __future__ import annotations

import urllib.request
from pathlib import Path

HUB = "openbmb/VoxCPM2"

FILES = {
    "config.json": 4336,
    "tokenizer.json": 3_676_772,
    "tokenizer_config.json": 5059,
    "special_tokens_map.json": 1632,
    "model.safetensors": 4_580_080_592,
    "audiovae.pth": 376_951_122,
}


def checkout(root: Path) -> Path:
    return root / "models" / "openbmb" / "VoxCPM2"


def download(dest: Path) -> list[str]:
    """Fetch each published file. Returns the names written or already present."""
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for name, size in FILES.items():
        target = dest / name
        _download_one(_url(name), target, size)
        names.append(name)
    return names


def _url(name: str) -> str:
    return f"https://huggingface.co/{HUB}/resolve/main/{name}"


def _download_one(url: str, target: Path, size: int) -> None:
    if target.is_file() and target.stat().st_size == size:
        return
    have = target.stat().st_size if target.is_file() else 0
    headers = {"User-Agent": "favlo"}
    if 0 < have < size:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    temporary = target.with_name(target.name + ".part")
    with urllib.request.urlopen(request, timeout=600) as response:
        status = getattr(response, "status", 200)
        if status == 206 and have:
            mode = "ab"
            sink = target
        else:
            mode = "wb"
            sink = temporary
        with sink.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    if sink is temporary:
        temporary.replace(target)
    written = target.stat().st_size
    if written != size:
        message = f"{target.name} is {written} bytes, the checkout is {size}"
        raise RuntimeError(message)
