"""Paths and the Hugging Face download for umT5-XXL."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_EXTRAS = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)
DEFAULT_REPO = "google/umt5-xxl"
PACK_DIR = ("google", "umt5-xxl-int8")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_LINK_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})


def models_dir(root: Path | None = None) -> Path:
    from favlo.cli import repo_root

    base = root or repo_root()
    return base / "models"


def source_dir(root: Path | None = None, repo: str | None = None) -> Path:
    """BF16 encoder and tokenizer checkout under ``models/<org>/<name>/``."""
    name = DEFAULT_REPO if repo is None else repo_id(repo)
    return models_dir(root) / name


def pack_dir(root: Path | None = None) -> Path:
    """Shared int8 pack under ``models/google/umt5-xxl-int8``."""
    return models_dir(root).joinpath(*PACK_DIR)


def encoder_path(root: Path | None = None, repo: str | None = None) -> Path:
    return source_dir(root, repo) / ENCODER_FILE


def tokenizer_path(root: Path | None = None, repo: str | None = None) -> Path:
    return source_dir(root, repo) / TOKENIZER_FILE


def pack_tokenizer_path(root: Path | None = None) -> Path:
    return pack_dir(root) / TOKENIZER_FILE


def repo_id(text: str) -> str:
    """``org/name``, or the same id taken from a huggingface.co model link."""
    raw = text.strip()
    if "://" in raw or raw.startswith(("huggingface.co/", "www.huggingface.co/", "hf.co/")):
        if "://" not in raw:
            raw = "https://" + raw
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()
        if parsed.username or parsed.password or host not in _LINK_HOSTS:
            message = "repository link is a huggingface.co model"
            raise ValueError(message)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            message = "repository is org/name or a huggingface.co link"
            raise ValueError(message)
        raw = f"{parts[0]}/{parts[1]}"
    if _REPO_ID.fullmatch(raw) is None:
        message = "repository is org/name or a huggingface.co link"
        raise ValueError(message)
    return raw


def download(repo: str | None = None, root: Path | None = None) -> Path:
    """Download the BF16 encoder and every tokenizer file. Returns the source directory."""
    from huggingface_hub import hf_hub_download

    named = DEFAULT_REPO if repo is None else repo_id(repo)
    dest = source_dir(root, named)
    dest.mkdir(parents=True, exist_ok=True)
    hf_hub_download(repo_id=named, filename=ENCODER_FILE, local_dir=str(dest))
    print(f"setup {ENCODER_FILE}", flush=True)
    hf_hub_download(repo_id=named, filename=TOKENIZER_FILE, local_dir=str(dest))
    print(f"setup {TOKENIZER_FILE}", flush=True)
    for name in TOKENIZER_EXTRAS:
        try:
            hf_hub_download(repo_id=named, filename=name, local_dir=str(dest))
        except (OSError, ValueError, RuntimeError):
            continue
        print(f"setup {name}", flush=True)
    return dest
