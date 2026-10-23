"""Download the official Qwen3.8-27B checkout with plain HTTP.

The listener does not import this module. ``download`` does.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

REPO = "Qwen/Qwen3.8-27B"
CHECKOUT = "Qwen3.8-27B-bf16"
_HOST = "https://huggingface.co"
_CHUNK = 1024 * 1024
_AGENT = "favlo"

Progress = Callable[[int, int, str], None]


def shortfall(need: int, dest: Path) -> str | None:
    """Bytes still required for the checkout. None when the volume has room."""
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if free >= need:
        return None
    return f"the checkout volume needs {_gib(need)}, {_gib(free)} free"


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

        def _report(
            written: int,
            _base: int = base,
            _label: str = label,
            _total: int = total,
        ) -> None:
            on_progress(_base + written, _total, _label)

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
    marker = dest / "fetch_complete.json"
    marker.write_text(json.dumps({"repo": repo, "files": len(files)}) + "\n", encoding="utf-8")


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
