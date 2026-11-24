"""Download the official Inkling 2 Mini checkpoint with plain HTTP.

The listener does not import this module. ``setup`` does. The experts are
already NVFP4 and MXFP8 in the file. ``mtp.safetensors`` stays its own file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

REPO = "thinkingmachines/Inkling-2-Mini"
_HOST = "https://huggingface.co"
_CHUNK = 8 * 1024 * 1024
_AGENT = "favlo"
_SKIP = frozenset({".gitattributes", "README.md"})
# The calibration input file ``calibrate`` writes beside the checkpoint.
HESSIAN_BYTES = 2_243_711_948
MARKER = "fetch_complete.json"

Progress = Callable[[int, int, str], None]


class File:
    """One published file."""

    def __init__(self, name: str, size: int, sha256: str | None) -> None:
        self.name = name
        self.size = size
        self.sha256 = sha256


def dense_pack_bytes() -> int:
    """Bytes of ``dense-fp4.safetensors``, including a small header slack."""
    from inkling_2_mini.metal41.layout import HIDDEN, LAYERS, Q_DIM
    from inkling_2_mini.metal41.weights import ATTN_ROWS, HEAD_ROWS

    def fp4(rows: int, cols: int) -> int:
        return rows * cols // 2 + rows * cols // 32

    body = LAYERS * (fp4(ATTN_ROWS, HIDDEN) + fp4(HIDDEN, Q_DIM)) + fp4(HEAD_ROWS, HIDDEN)
    return body + (1 << 20)


def keep(name: str) -> bool:
    """The served checkpoint. The drafter and the repo card stay on the hub."""
    if not name or name.endswith("/") or ".." in Path(name).parts:
        return False
    return Path(name).name not in _SKIP


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
            f"the volume needs {_gib(need)} for the checkpoint and the packed weight files, "
            f"{_gib(free_download)} free"
        )
    if free_download < need_download:
        return f"the checkpoint volume needs {_gib(need_download)}, {_gib(free_download)} free"
    if free_record < need_record:
        return f"the calibration volume needs {_gib(need_record)}, {_gib(free_record)} free"
    return None


def list_files(host: str = _HOST, repo: str = REPO) -> list[File]:
    """Published files this pipeline serves. Smaller files come first."""
    url = f"{host}/api/models/{repo}"
    with _request(url) as response:
        body = json.loads(response.read().decode())
    siblings = body.get("siblings") if isinstance(body, dict) else None
    if not isinstance(siblings, list) or not siblings:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files: list[File] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = item.get("rfilename")
        if not isinstance(name, str) or not keep(name):
            continue
        size = item.get("size")
        if not isinstance(size, int) or size < 0:
            size = _remote_size(host, repo, name)
        lfs = item.get("lfs")
        sha = None
        if isinstance(lfs, dict) and isinstance(lfs.get("oid"), str) and lfs["oid"]:
            sha = lfs["oid"].removeprefix("sha256:")
        files.append(File(name, size, sha))
    if not files:
        message = f"{repo} published no files"
        raise RuntimeError(message)
    files.sort(key=lambda item: (item.size, item.name))
    return files


def _weight_file(name: str) -> bool:
    return Path(name).name.endswith(".safetensors")


def _record_ready(dest: Path) -> bool:
    from inkling_2_mini.metal41.record import ready

    return ready(dest)


def _repacked(dest: Path) -> bool:
    """The language shard was rewritten to float16 and no longer matches the hub size."""
    marker = dest / "repack.json"
    language = dest / "model-00001-of-00001.safetensors"
    vision = dest / "vision.safetensors"
    if not marker.is_file() or not language.is_file() or not vision.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("format") == "favlo-inkling-f16-1"
        and record.get("language") == language.stat().st_size
    )


def remaining_download(dest: Path, files: list[File]) -> int:
    """Bytes not yet on disk. A file longer than its listing counts as missing."""
    missing = 0
    marked = _marked(dest)
    done = _repacked(dest)
    sealed = _record_ready(dest)
    for item in files:
        path = dest / item.name
        have = path.stat().st_size if path.is_file() else 0
        if sealed and _weight_file(item.name):
            continue
        if done and item.name == "model-00001-of-00001.safetensors":
            continue
        if have == item.size and _marked_ok(marked, item):
            continue
        if have > item.size:
            missing += item.size
        else:
            missing += item.size - have
    return missing


def remaining_expert(dest: Path, config_root: Path | None = None) -> int:
    """Expert-pack bytes not yet in the additive folder."""
    from inkling_2_mini.metal41.resident import expert_pack_bytes, open_expert_pack

    if _record_ready(dest):
        return 0
    record = open_expert_pack(dest, config_root or dest)
    if record is not None:
        record.close()
        return 0
    return expert_pack_bytes()


def remaining_calibration(dest: Path) -> int:
    """Calibration bytes not yet beside the checkpoint."""
    from inkling_2_mini.metal41.calibration import HESSIANS_NAME
    from inkling_2_mini.metal41.gptq import PACK_NAME

    need = 0
    for name, size in ((PACK_NAME, dense_pack_bytes()), (HESSIANS_NAME, HESSIAN_BYTES)):
        path = dest / name
        have = path.stat().st_size if path.is_file() else 0
        need += max(0, size - have)
    return need


def download(
    dest: Path,
    files: list[File],
    on_progress: Progress,
    *,
    host: str = _HOST,
    repo: str = REPO,
) -> set[str]:
    """Write ``files`` under ``dest``. A short file continues with a range request."""
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(item.size for item in files)
    finished = 0
    marked = _marked(dest)
    digests = dict(marked)
    fetched: set[str] = set()
    done = _repacked(dest)
    sealed = _record_ready(dest)
    for item in files:
        path = dest / item.name
        if path.is_symlink():
            message = f"refusing {item.name}"
            raise RuntimeError(message)
        path.parent.mkdir(parents=True, exist_ok=True)
        have = path.stat().st_size if path.is_file() else 0
        label = Path(item.name).name
        if (sealed and _weight_file(item.name)) or (
            done and item.name == "model-00001-of-00001.safetensors"
        ):
            finished += item.size
            on_progress(finished, total, label)
            continue
        if have == item.size and (_marked_ok(marked, item) or item.sha256 is None):
            finished += item.size
            on_progress(finished, total, label)
            if item.sha256:
                digests[item.name] = item.sha256
            continue
        if have == item.size and item.sha256 and _sha256(path) == item.sha256:
            digests[item.name] = item.sha256
            finished += item.size
            on_progress(finished, total, label)
            continue
        if have >= item.size:
            path.unlink()
            have = 0
        url = _file_url(host, repo, item.name)
        base = finished

        def _report(written: int, _base: int = base, _label: str = label) -> None:
            on_progress(_base + written, total, _label)

        _stream(url, path, have, _report)
        final = path.stat().st_size
        if final != item.size:
            path.unlink()
            message = f"{item.name} is {final} bytes, expected {item.size}"
            raise RuntimeError(message)
        if item.sha256 is not None:
            found = _sha256(path)
            if found != item.sha256:
                path.unlink()
                message = f"{item.name} checksum does not match the published file"
                raise RuntimeError(message)
            digests[item.name] = found
        fetched.add(item.name)
        finished += item.size
        on_progress(finished, total, label)
    _write_marker(dest, repo, digests)
    return fetched


def _marked(dest: Path) -> dict[str, str]:
    path = dest / MARKER
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    raw = body.get("sha256") if isinstance(body, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        name: sha for name, sha in raw.items() if isinstance(name, str) and isinstance(sha, str)
    }


def _marked_ok(marked: dict[str, str], item: File) -> bool:
    if item.sha256 is None:
        return True
    return marked.get(item.name) == item.sha256


def _write_marker(dest: Path, repo: str, digests: dict[str, str]) -> None:
    if not digests:
        return
    path = dest / MARKER
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps({"repo": repo, "sha256": digests}, indent=2) + "\n")
    partial.replace(path)


def _file_url(host: str, repo: str, name: str) -> str:
    return f"{host}/{repo}/resolve/main/{quote(name, safe='/')}"


def _remote_size(host: str, repo: str, name: str) -> int:
    url = _file_url(host, repo, name)
    request = urllib.request.Request(url, method="HEAD", headers=_headers())
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
        mode = "ab" if status == 206 else "wb"
        written = have if status == 206 else 0
        with path.open(mode) as handle:
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                on_file_bytes(written)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": _AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _request(url: str, headers: dict[str, str] | None = None) -> object:
    request = urllib.request.Request(url, headers=_headers(headers))
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as exc:
        message = f"download failed: {exc.reason}"
        raise RuntimeError(message) from exc


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.1f} GiB"
