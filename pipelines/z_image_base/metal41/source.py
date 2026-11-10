"""Fetch Tongyi-MAI/Z-Image from the repository itself.

Files land under ``models/Z-Image``. The transfer uses the resolve URL and
does not read or write a hub cache.
"""

from __future__ import annotations

from pathlib import Path

from z_image.metal41.source import fetch

REPO = "Tongyi-MAI/Z-Image"
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
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": 9_973_727_144,
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": 2_336_146_728,
    "vae/diffusion_pytorch_model.safetensors": 167_666_902,
}


def source_dir(root: Path | None = None) -> Path:
    from favlo.cli import repo_root

    base = root or repo_root()
    return base / "models" / "Z-Image"


def download(dest: Path | None = None) -> Path:
    """Download the tree into ``dest``. A partial file resumes from its size."""
    folder = dest or source_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for rel in SMALL:
        fetch(BASE, folder, rel, None)
        print(f"setup {rel}", flush=True)
    for rel, size in WEIGHTS.items():
        fetch(BASE, folder, rel, size)
        print(f"setup {rel}", flush=True)
    return folder
