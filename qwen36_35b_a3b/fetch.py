"""Download the official Qwen3.6-35B-A3B checkout with plain HTTP.

The listener does not import this module. ``setup`` does.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from qwen36_35b_a3b.convert import LANGUAGE_LAYERS
from qwen36_35b_a3b.formats import EXPERT_IMAGE

REPO = "Qwen/Qwen3.6-35B-A3B"
_HOST = "https://huggingface.co"
_CHUNK = 1024 * 1024
_AGENT = "favlo"

Progress = Callable[[int, int, str], None]


def record_bytes() -> int:
    """Language expert images, one MTP layer, and headroom for resident.bin."""
    images = (LANGUAGE_LAYERS + 1) * 256 * EXPERT_IMAGE
    return images + images // 10


def shortfall(need_download: int, need_record: int, download: Path, record: Path) -> str | None:
    """Bytes still required against free space. One volume must hold both."""
    download.mkdir(parents=True, exist_ok=True)
    record.mkdir(parents=True, exist_ok=True)
    free_download = shutil.disk_usage(download).free
    free_record = shutil.disk_usage(record).free
    same = os.stat(download).st_dev == os.stat(record).st_dev
    if same:
        need = need_download + need_record
        if free_download >= need:
            return None
        return (
            f"the volume needs {_gib(need)} for the checkout and the record, "
            f"{_gib(free_download)} free"
        )
    if free_download < need_download:
        return f"the checkout volume needs {_gib(need_download)}, {_gib(free_download)} free"
    if free_record < need_record:
        return f"the record volume needs {_gib(need_record)}, {_gib(free_record)} free"
    return None


def list_files(host: str = _HOST, repo: str = REPO) -> list[tuple[str, int]]:
    """Published files and sizes. Smaller files come first so the bar moves at once."""
    url = f"{host}/api/models/{repo}"
    with _request(url) as response:
        body = json.loads(response.read().decode())
    siblings = body.get("siblings") if isinstance(body, dict) else None
    if not isinstance(siblings, list) or not siblings:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files: list[tuple[str, int]] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str) or not name or name.startswith(".."):
            continue
        size = item.get("size")
        if not isinstance(size, int) or size < 0:
            size = _remote_size(host, repo, name)
        files.append((name, size))
    if not files:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files.sort(key=lambda item: (item[1], item[0]))
    return files


def download(
    dest: Path,
    files: list[tuple[str, int]],
    on_progress: Progress,
    *,
    host: str = _HOST,
    repo: str = REPO,
) -> None:
    """Write ``files`` under ``dest``. A short file is continued with a range request."""
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(size for _name, size in files)
    finished = 0
    for name, size in files:
        path = dest / name
        if path.is_symlink() or ".." in Path(name).parts:
            message = f"refusing {name}"
            raise RuntimeError(message)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = _file_url(host, repo, name)
        have = path.stat().st_size if path.is_file() else 0
        label = Path(name).name
        if have > size:
            remote = _remote_size(host, repo, name)
            if have == remote:
                total += remote - size
                size = remote
            else:
                path.unlink()
                have = 0
                if remote != size:
                    total += remote - size
                    size = remote
        if have == size:
            finished += size
            on_progress(finished, total, label)
            continue
        base = finished

        def _report(written: int, _base: int = base, _label: str = label) -> None:
            on_progress(_base + written, total, _label)

        _stream(url, path, have, _report)
        final = path.stat().st_size
        if final != size:
            remote = _remote_size(host, repo, name)
            if final != remote:
                path.unlink()
                message = f"{name} is {final} bytes, expected {remote}"
                raise RuntimeError(message)
            total += remote - size
            size = remote
        finished += size
        on_progress(finished, total, label)


def remaining_download(dest: Path, files: list[tuple[str, int]]) -> int:
    """Bytes not yet on disk. A file longer than its listing counts as missing."""
    missing = 0
    for name, size in files:
        path = dest / name
        have = path.stat().st_size if path.is_file() else 0
        if have > size:
            missing += size
        else:
            missing += size - have
    return missing


def _file_url(host: str, repo: str, name: str) -> str:
    return f"{host}/{repo}/resolve/main/{quote(name, safe='/')}"


def _remote_size(host: str, repo: str, name: str) -> int:
    url = _file_url(host, repo, name)
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        length = response.headers.get("Content-Length")
    if length is None:
        message = f"{name} has no size"
        raise RuntimeError(message)
    return int(length)


def _stream(url: str, path: Path, have: int, on_file_bytes: Callable[[int], None]) -> None:
    headers: dict[str, str] = {}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        response = _request(url, headers)
    except urllib.error.HTTPError as exc:
        message = f"download failed: {exc.code} {url}"
        raise RuntimeError(message) from exc
    with response:
        status = response.getcode()
        if status == 206:
            mode = "ab"
            written = have
        else:
            mode = "wb"
            written = 0
        with path.open(mode) as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                on_file_bytes(written)


def _request(url: str, headers: dict[str, str] | None = None) -> object:
    merged = {"User-Agent": _AGENT}
    if headers:
        merged.update(headers)
    request = urllib.request.Request(url, headers=merged)
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as exc:
        message = f"download failed: {exc.reason}"
        raise RuntimeError(message) from exc


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.1f} GiB"
