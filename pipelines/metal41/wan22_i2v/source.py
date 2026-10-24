"""Download the Wan 2.2 I2V weights with plain HTTP.

Each file lands under ``models/<org>/<repo>/``, the repository it is fetched
from. The umT5 encoder is fetched from the Wan repository and stored with the
tokenizer, so that directory is the text encoder a later model replaces. The
transfer uses the resolve URL and does not read or write a hub cache. Sixteen
connections share one file. A ``.parts`` map records finished ranges so a
short file can continue.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote, urlparse

HOST = "https://huggingface.co"
VAE_REPO = "Wan-AI/Wan2.2-I2V-A14B"
VAE_FILE = "Wan2.1_VAE.pth"
UMT5_REPO = "google/umt5-xxl"
ENCODER_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_FILE = "tokenizer.json"
DIT_REPO = "lightx2v/Wan2.2-Distill-Models"
HIGH_FILE = "high.safetensors"
LOW_FILE = "low.safetensors"
_CHUNK = 8 << 20
_WORKERS = 16
_AGENT = "favlo"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_LINK_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})

# repo, remote name, local path under models/, bytes. Sizes were read from the resolve headers.
FILES = (
    (
        VAE_REPO,
        VAE_FILE,
        f"{VAE_REPO}/{VAE_FILE}",
        507_609_880,
    ),
    (
        VAE_REPO,
        ENCODER_FILE,
        f"{UMT5_REPO}/{ENCODER_FILE}",
        11_361_920_418,
    ),
    (
        DIT_REPO,
        "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors",
        f"{DIT_REPO}/{HIGH_FILE}",
        57_155_717_592,
    ),
    (
        DIT_REPO,
        "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors",
        f"{DIT_REPO}/{LOW_FILE}",
        57_155_717_592,
    ),
    (
        UMT5_REPO,
        TOKENIZER_FILE,
        f"{UMT5_REPO}/{TOKENIZER_FILE}",
        16_853_013,
    ),
)


def models_dir(root: Path | None = None) -> Path:
    from favlo.cli import repo_root

    base = root or repo_root()
    return base / "models"


def _repo_dir(root: Path | None, repo: str) -> Path:
    parts = Path(repo).parts
    if not parts or Path(repo).is_absolute() or ".." in parts:
        message = f"refusing {repo}"
        raise RuntimeError(message)
    return models_dir(root) / repo


def vae_dir(root: Path | None = None) -> Path:
    """Directory of the causal VAE."""
    return _repo_dir(root, VAE_REPO)


def umt5_dir(root: Path | None = None, repo: str | None = None) -> Path:
    """Directory of one umT5 encoder and its tokenizer.

    ``repo`` selects another Hugging Face repository. That directory must
    contain the encoder file and ``tokenizer.json``.
    """
    return _repo_dir(root, UMT5_REPO if repo is None else repo)


def dit_dir(root: Path | None = None) -> Path:
    """Directory of the high-noise and low-noise experts."""
    return _repo_dir(root, DIT_REPO)


def expert_paths(folder: Path | None = None) -> tuple[Path, Path]:
    """High-noise file, then the low-noise file."""
    dest = folder or dit_dir()
    return dest / HIGH_FILE, dest / LOW_FILE


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


def file_list(
    umt5: str | None = None,
    *,
    sizes: Callable[[str, str], int] | None = None,
) -> tuple[tuple[str, str, str, int], ...]:
    """Download rows. ``umt5`` replaces the default encoder and tokenizer."""
    if umt5 is None:
        return FILES
    repo = repo_id(umt5)
    lookup = published_size if sizes is None else sizes
    prefix = f"{UMT5_REPO}/"
    kept = [row for row in FILES if not row[2].startswith(prefix)]
    encoder = (repo, ENCODER_FILE, f"{repo}/{ENCODER_FILE}", lookup(repo, ENCODER_FILE))
    tokenizer = (repo, TOKENIZER_FILE, f"{repo}/{TOKENIZER_FILE}", lookup(repo, TOKENIZER_FILE))
    return (kept[0], encoder, *kept[1:], tokenizer)


def published_size(repo: str, name: str) -> int:
    """Byte size of one file published by a Hugging Face model repository."""
    url = f"{HOST}/api/models/{quote(repo, safe='/')}"
    request = urllib.request.Request(url, headers={"User-Agent": _AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeError) as exc:
        message = f"{repo} file list cannot be read"
        raise RuntimeError(message) from exc
    siblings = body.get("siblings") if isinstance(body, dict) else None
    if isinstance(siblings, list):
        for item in siblings:
            if not isinstance(item, dict) or item.get("rfilename") != name:
                continue
            size = item.get("size")
            if isinstance(size, int) and size > 0:
                return size
    message = f"{name} is not in {repo}"
    raise RuntimeError(message)


def download(dest: Path | None = None, *, umt5: str | None = None) -> Path:
    """Write the VAE, the umT5 files, and the two experts. Returns ``models/``."""
    files = file_list(umt5)
    folder = dest or models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    missing = _missing(folder, files)
    free = shutil.disk_usage(folder).free
    if free < missing:
        message = f"the checkout needs {_gib(missing)}, {_gib(free)} free"
        raise RuntimeError(message)
    for repo, remote, local, size in files:
        _fetch(folder, repo, remote, local, size)
        print(f"setup {local}", flush=True)
    return folder


def _missing(folder: Path, files: tuple[tuple[str, str, str, int], ...] = FILES) -> int:
    total = 0
    for _repo, _remote, local, size in files:
        have = _have(folder / local, size)
        if have < size:
            total += size - have
    return total


def mask_from_prefix(size: int, have: int, chunk: int = _CHUNK) -> bytearray:
    """Mark every range that is fully covered by a leading byte count."""
    count = (size + chunk - 1) // chunk
    done = bytearray(count)
    covered = min(max(have, 0), size)
    for index in range(count):
        end = min((index + 1) * chunk, size)
        if covered >= end:
            done[index] = 1
    return done


def _parts_path(path: Path) -> Path:
    return path.with_name(path.name + ".parts")


def _have(path: Path, size: int) -> int:
    parts = _parts_path(path)
    count = (size + _CHUNK - 1) // _CHUNK
    if parts.is_file():
        raw = parts.read_bytes()
        if len(raw) == count:
            return _bytes_done(bytearray(raw), size)
    if not path.is_file():
        return 0
    have = path.stat().st_size
    if have == size:
        return size
    return min(have, size)


def _bytes_done(mask: bytearray, size: int) -> int:
    total = 0
    for index, bit in enumerate(mask):
        if not bit:
            continue
        start = index * _CHUNK
        total += min(_CHUNK, size - start)
    return total


def _load_mask(path: Path, size: int) -> bytearray:
    parts = _parts_path(path)
    count = (size + _CHUNK - 1) // _CHUNK
    if parts.is_file():
        raw = parts.read_bytes()
        if len(raw) == count:
            return bytearray(raw)
    have = path.stat().st_size if path.is_file() else 0
    if have == size and not parts.is_file():
        return bytearray([1]) * count
    if have > size:
        have = 0
    return mask_from_prefix(size, have)


def _fetch(folder: Path, repo: str, remote: str, local: str, size: int) -> None:
    path = folder / local
    if path.is_symlink() or ".." in Path(local).parts or Path(local).is_absolute():
        message = f"refusing {local}"
        raise RuntimeError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = _parts_path(path)
    count = (size + _CHUNK - 1) // _CHUNK
    if path.is_file() and path.stat().st_size > size:
        valid = parts.is_file() and len(parts.read_bytes()) == count
        if not valid:
            path.unlink()
    mask = _load_mask(path, size)
    if all(mask) and path.is_file() and path.stat().st_size == size:
        parts.unlink(missing_ok=True)
        return
    if not path.exists():
        path.touch()
    parts.write_bytes(bytes(mask))
    os.truncate(path, size)
    url = f"{HOST}/{repo}/resolve/main/{quote(remote, safe='/')}"
    pending = [index for index, bit in enumerate(mask) if not bit]
    lock = threading.Lock()
    stop = threading.Event()
    written = _bytes_done(mask, size)
    last = [time.time()]
    fd = os.open(path, os.O_RDWR)

    def work(index: int) -> None:
        nonlocal written
        if stop.is_set():
            return
        start = index * _CHUNK
        end = min(start + _CHUNK, size) - 1
        data = _read_range(url, local, start, end)
        with lock:
            os.pwrite(fd, data, start)
            mask[index] = 1
            parts.write_bytes(bytes(mask))
            written += len(data)
            now = time.time()
            if now - last[0] >= 30:
                print(f"setup {local} {written}/{size}", flush=True)
                last[0] = now

    try:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            futures = [pool.submit(work, index) for index in pending]
            first: BaseException | None = None
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    stop.set()
                    if first is None:
                        first = exc
            if first is not None:
                raise first
    finally:
        os.close(fd)
    if path.stat().st_size != size or not all(mask):
        message = f"{local} is incomplete"
        raise RuntimeError(message)
    parts.unlink(missing_ok=True)


def _read_range(url: str, local: str, start: int, end: int) -> bytes:
    expect = end - start + 1
    delay = 2
    last_error: Exception | None = None
    for _attempt in range(5):
        headers = {"User-Agent": _AGENT, "Range": f"bytes={start}-{end}"}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if int(response.getcode()) != 206:
                    message = f"range was refused for {local}"
                    raise RuntimeError(message)
                data = response.read()
            if len(data) == expect:
                return data
            last_error = RuntimeError(f"short read {len(data)}")
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 404}:
                message = f"download failed: {exc.code} {local}"
                raise RuntimeError(message) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(delay)
        delay = min(delay * 2, 30)
    message = f"download failed: {local} at {start}: {last_error}"
    raise RuntimeError(message)


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.1f} GiB"
