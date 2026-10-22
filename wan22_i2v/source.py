"""Download the Wan 2.2 I2V weights with plain HTTP.

Files land under ``models/Wan2.2-I2V-A14B``. The transfer uses the resolve
URL and does not read or write a hub cache. Sixteen connections share one
file. A ``.parts`` map records finished ranges so a short file can continue.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

HOST = "https://huggingface.co"
FOLDER = "Wan2.2-I2V-A14B"
_CHUNK = 8 << 20
_WORKERS = 16
_AGENT = "favlo"

# repo, remote name, local name, bytes. Sizes were read from the resolve headers.
FILES = (
    (
        "Wan-AI/Wan2.2-I2V-A14B",
        "Wan2.1_VAE.pth",
        "Wan2.1_VAE.pth",
        507_609_880,
    ),
    (
        "Wan-AI/Wan2.2-I2V-A14B",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        11_361_920_418,
    ),
    (
        "lightx2v/Wan2.2-Distill-Models",
        "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors",
        "high.safetensors",
        57_155_717_592,
    ),
    (
        "lightx2v/Wan2.2-Distill-Models",
        "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors",
        "low.safetensors",
        57_155_717_592,
    ),
    (
        "google/umt5-xxl",
        "tokenizer.json",
        "tokenizer.json",
        16_853_013,
    ),
)


def source_dir(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[1]
    return base / "models" / FOLDER


def expert_paths(folder: Path | None = None) -> tuple[Path, Path]:
    """High-noise file, then the low-noise file."""
    dest = folder or source_dir()
    return dest / "high.safetensors", dest / "low.safetensors"


def download(dest: Path | None = None) -> Path:
    """Write the weight files and the umT5 tokenizer. Returns the directory."""
    folder = dest or source_dir()
    folder.mkdir(parents=True, exist_ok=True)
    missing = _missing(folder)
    free = shutil.disk_usage(folder).free
    if free < missing:
        message = f"the checkout needs {_gib(missing)}, {_gib(free)} free"
        raise RuntimeError(message)
    for repo, remote, local, size in FILES:
        _fetch(folder, repo, remote, local, size)
        print(f"setup {local}", flush=True)
    return folder


def _missing(folder: Path) -> int:
    total = 0
    for _repo, _remote, local, size in FILES:
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
    if path.is_symlink() or ".." in Path(local).parts:
        message = f"refusing {local}"
        raise RuntimeError(message)
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
