"""Rewrite the checkpoint so the language open can pread float16 planes.

Vision tensors move to ``vision.safetensors``. MTP tensors move to
``mtp.safetensors`` when they are inside the language shard. BF16 payloads
become float16 of the same shape. Every other tensor is copied.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from inkling_2_mini.metal41.weights import checkpoint_file

_PIECE = 8 << 20
FORMAT = "favlo-inkling-f16-1"
MARKER = "repack.json"
LANGUAGE = "model-00001-of-00001.safetensors"
VISION = "vision.safetensors"
MTP = "mtp.safetensors"


def repack(root: Path, log: Callable[[str], None] | None = None) -> None:
    """Split and convert ``root`` when the language shard is still the hub file.

    A standalone ``mtp.safetensors`` is left as published. The language open
    does not read it.
    """
    path = checkpoint_file(root)
    if path.is_file() and _needs_split(path):
        _split(path, Path(root), log)


def _needs_split(path: Path) -> bool:
    header = _header(path)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if _kind(name) != "language" or info.get("dtype") == "BF16":
            return True
    return False


def _kind(name: str) -> str:
    if name.startswith("model.visual.") or name.startswith("visual."):
        return "vision"
    parts = name.split(".")
    if name.startswith("mtp.") or "mtp" in parts:
        return "mtp"
    return "language"


def _header(path: Path) -> dict:
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
    if not isinstance(header, dict):
        message = f"{path.name} header is not an object"
        raise ValueError(message)
    return header


def _split(path: Path, root: Path, log: Callable[[str], None] | None) -> None:
    header = _header(path)
    groups: dict[str, list[str]] = {"language": [], "vision": [], "mtp": []}
    for name in header:
        if name == "__metadata__":
            continue
        groups[_kind(name)].append(name)
    fd = os.open(path, os.O_RDONLY)
    try:
        data = 8 + int.from_bytes(os.pread(fd, 8, 0), "little")
        if groups["vision"]:
            _write(fd, header, data, groups["vision"], root / VISION, log)
        if groups["mtp"]:
            _write(fd, header, data, groups["mtp"], root / MTP, log)
        _write(fd, header, data, groups["language"], root / f"{LANGUAGE}.next", log)
    finally:
        os.close(fd)
    (root / f"{LANGUAGE}.next").replace(path)
    _mark(root, path)
    if log is not None:
        log("language planes stored as float16")


def _write(
    fd: int,
    header: dict,
    data: int,
    names: list[str],
    dest: Path,
    log: Callable[[str], None] | None,
) -> None:
    body: dict[str, object] = {}
    meta = header.get("__metadata__")
    if isinstance(meta, dict) and dest.name.startswith(LANGUAGE):
        body["__metadata__"] = meta
    cursor = 0
    pieces: list[tuple[int, int, bool]] = []
    for name in names:
        info = header[name]
        start, end = info["data_offsets"]
        nbytes = int(end) - int(start)
        dtype = "F16" if info["dtype"] == "BF16" else info["dtype"]
        body[name] = {
            "dtype": dtype,
            "shape": info["shape"],
            "data_offsets": [cursor, cursor + nbytes],
        }
        pieces.append((data + int(start), nbytes, info["dtype"] == "BF16"))
        cursor += nbytes
    raw = json.dumps(body, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if log is not None:
        log(dest.name)
    with dest.open("wb") as raw_out:
        out = _Progress(raw_out, log, dest.name)
        out.write(len(raw).to_bytes(8, "little"))
        out.write(raw)
        for start, nbytes, convert in pieces:
            if convert:
                _convert(fd, start, nbytes, out)
            else:
                _copy(fd, start, nbytes, out)


class _Progress:
    """Log each GiB written."""

    def __init__(self, raw, log: Callable[[str], None] | None, name: str) -> None:
        self.raw = raw
        self.log = log
        self.name = name
        self.done = 0
        self.noted = 0

    def write(self, data: bytes) -> None:
        self.raw.write(data)
        self.done += len(data)
        if self.log is not None and self.done - self.noted >= (1 << 30):
            self.noted = self.done
            self.log(f"{self.name} {self.done >> 30} GiB")


def _convert(fd: int, start: int, nbytes: int, out) -> None:
    offset = 0
    while offset < nbytes:
        step = min(_PIECE, nbytes - offset)
        raw = os.pread(fd, step, start + offset)
        if len(raw) != step:
            message = f"read stopped at {start + offset}"
            raise OSError(message)
        wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        out.write(np.ascontiguousarray(wide.view(np.float32).astype(np.float16)).tobytes())
        offset += step


def _copy(fd: int, start: int, nbytes: int, out) -> None:
    offset = 0
    while offset < nbytes:
        step = min(_PIECE, nbytes - offset)
        raw = os.pread(fd, step, start + offset)
        if len(raw) != step:
            message = f"read stopped at {start + offset}"
            raise OSError(message)
        out.write(raw)
        offset += step


def _mark(root: Path, language: Path) -> None:
    record = {"format": FORMAT, "language": language.stat().st_size}
    (root / MARKER).write_text(json.dumps(record, separators=(",", ":")) + "\n")
