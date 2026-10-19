"""Resolve Hugging Face hub cache snapshot directories."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub.file_download import repo_folder_name


def hub_cache_dir() -> Path:
    if hf_hub_cache := os.getenv("HF_HUB_CACHE"):
        return Path(hf_hub_cache).expanduser().resolve()
    if hf_home := os.getenv("HF_HOME"):
        return (Path(hf_home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def snapshot_dir(repo_id: str, revision: str = "main") -> Path:
    """Return ``snapshots/<commit>`` for a cached model repo."""
    storage = hub_cache_dir() / repo_folder_name(repo_id=repo_id, repo_type="model")
    ref_path = storage / "refs" / revision
    if not ref_path.is_file():
        message = f"no cached revision {revision!r} for {repo_id} under {storage}"
        raise FileNotFoundError(message)
    commit = ref_path.read_text(encoding="utf-8").strip()
    snapshot = storage / "snapshots" / commit
    if not snapshot.is_dir():
        message = f"missing snapshot {commit} for {repo_id}"
        raise FileNotFoundError(message)
    return snapshot
