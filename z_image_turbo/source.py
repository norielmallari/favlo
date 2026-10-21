"""Fetch Tongyi-MAI/Z-Image-Turbo from the repository itself.

Files land under ``models/Z-Image-Turbo``. The transfer uses the resolve URL
and does not read or write a hub cache.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

REPO = "Tongyi-MAI/Z-Image-Turbo"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"

SMALL = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
)

WEIGHTS = {
    "text_encoder/model-00001-of-00003.safetensors": 3_957_900_840,
    "text_encoder/model-00002-of-00003.safetensors": 3_987_450_520,
    "text_encoder/model-00003-of-00003.safetensors": 99_630_640,
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": 9_973_693_184,
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": 9_973_714_824,
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": 4_672_282_880,
    "vae/diffusion_pytorch_model.safetensors": 167_666_902,
}


def source_dir(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[1]
    return base / "models" / "Z-Image-Turbo"


def download(dest: Path | None = None) -> Path:
    """Download the tree into ``dest``. A partial file resumes from its size."""
    folder = dest or source_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for rel in SMALL:
        _fetch(folder, rel, None)
        print(f"setup {rel}", flush=True)
    for rel, size in WEIGHTS.items():
        _fetch(folder, rel, size)
        print(f"setup {rel}", flush=True)
    return folder


def _fetch(folder: Path, rel: str, expected: int | None) -> None:
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
    request = urllib.request.Request(BASE + rel, headers=headers)
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
