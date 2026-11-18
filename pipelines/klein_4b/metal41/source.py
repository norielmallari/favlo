"""Locate the Hugging Face snapshot. This does not start a download."""

from __future__ import annotations

from pathlib import Path

from .constants import REPO, WEIGHTS


def snapshot() -> Path:
    """The local snapshot directory. Missing files raise ``FileNotFoundError``."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(REPO, local_files_only=True)
    except Exception as exc:
        message = f"{REPO} is not in the local Hugging Face cache"
        raise FileNotFoundError(message) from exc
    root = Path(path)
    missing = [name for name in WEIGHTS if not (root / name).is_file()]
    if missing:
        message = f"{REPO} is missing {', '.join(missing)}"
        raise FileNotFoundError(message)
    return root


def weights_ready(root: Path | None = None) -> bool:
    """True when each packed safetensors file has its published byte count."""
    try:
        folder = snapshot() if root is None else root
    except FileNotFoundError:
        return False
    for name, size in WEIGHTS.items():
        path = folder / name
        if not path.is_file() or path.stat().st_size != size:
            return False
    return True


def tokenizer_file(root: Path | None = None) -> Path:
    """``tokenizer.json`` beside the snapshot, or inside a packed record."""
    folder = snapshot() if root is None else root
    path = folder / "tokenizer" / "tokenizer.json"
    if path.is_file():
        return path
    packed = folder / "tokenizer.json"
    if packed.is_file():
        return packed
    message = "tokenizer.json is missing"
    raise FileNotFoundError(message)
