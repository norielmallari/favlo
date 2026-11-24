"""Raw weight record for an 8 MiB pread.

``resident.bin`` holds the float language planes and the nearest fp8 attention
planes. ``experts.bin`` holds the Metal fp4 and MXFP8 expert payloads.
``vision.bin`` and ``mtp.bin`` are separate files. ``manifest.json`` indexes
each tensor. The language open does not read a safetensors weight file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from inkling_2_mini.metal41.gptq import config_digest

FORMAT = "favlo-inkling-record-1"
MANIFEST = "manifest.json"
RESIDENT = "resident.bin"
EXPERTS = "experts.bin"
VISION = "vision.bin"
MTP = "mtp.bin"
_PIECE = 8 << 20
_DROPPED = frozenset({"wq_du", "wr_du", "gate_attn", "wk_dv", "wv_dv", "wo_ud"})
_GROUPS = {
    "language": RESIDENT,
    "experts": EXPERTS,
    "vision": VISION,
    "mtp": MTP,
}


def ready(root: Path) -> bool:
    """True when the manifest matches ``config.json`` and the four files are complete."""
    path = Path(root) / MANIFEST
    if not path.is_file() or not (Path(root) / "config.json").is_file():
        return False
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if (
        not isinstance(body, dict)
        or body.get("format") != FORMAT
        or body.get("complete") is not True
    ):
        return False
    if body.get("config") != config_digest(root):
        return False
    files = body.get("files")
    if not isinstance(files, dict):
        return False
    for name in _GROUPS.values():
        listed = files.get(name)
        file = Path(root) / name
        if not isinstance(listed, int) or not file.is_file() or file.stat().st_size != listed:
            return False
        group = _group_for(body, name)
        if not isinstance(group, dict) or not group:
            return False
    return True


def publish(root: Path, log: Callable[[str], None] | None = None) -> None:
    """Copy the baked packs into the raw record and drop the safetensors weights."""
    root = Path(root)
    if ready(root):
        _drop_sources(root)
        if log is not None:
            log("record is current")
        return
    from inkling_2_mini.metal41.resident import bake
    from inkling_2_mini.metal41.vision import ensure_pack
    from inkling_2_mini.metal41.weights import checkpoint_file, ensure_dense

    if not checkpoint_file(root).is_file():
        message = "the language shard is missing"
        raise FileNotFoundError(message)
    bake(root, log=log)
    ensure_dense(root, log=log)
    ensure_pack(root, log=log)
    seal(root, log=log)


def seal(root: Path, log: Callable[[str], None] | None = None) -> None:
    """Write the record from the safetensors packs already in ``root``."""
    root = Path(root)
    if ready(root):
        _drop_sources(root)
        return
    from inkling_2_mini.metal41.resident import EXPERT_PACK
    from inkling_2_mini.metal41.vision import PACK_NAME
    from inkling_2_mini.metal41.weights import DENSE_PACK, checkpoint_file

    language = checkpoint_file(root)
    experts = root / EXPERT_PACK
    dense = root / DENSE_PACK
    vision = root / PACK_NAME
    missing = [path.name for path in (language, experts, dense, vision) if not path.is_file()]
    if missing:
        message = f"record is missing {', '.join(missing)}"
        raise FileNotFoundError(message)
    mtp = root / "mtp.safetensors"
    try:
        groups = {
            "language": _resident(language, dense),
            "experts": _export_payload(experts, root / f"{EXPERTS}.partial"),
            "vision": _export_payload(vision, root / f"{VISION}.partial"),
            "mtp": _mtp(root, language, mtp),
        }
        _check_groups(root, groups)
        for name in (RESIDENT, EXPERTS, VISION, MTP):
            partial = root / f"{name}.partial"
            partial.replace(root / name)
            if log is not None:
                log(f"wrote {name}")
        _matches_payload(experts, root / EXPERTS)
        _matches_payload(vision, root / VISION)
        _write_manifest(root, groups)
        if not ready(root):
            message = "record does not match config.json"
            raise RuntimeError(message)
        _drop_sources(root)
    except Exception:
        for name in (RESIDENT, EXPERTS, VISION, MTP, MANIFEST):
            (root / f"{name}.partial").unlink(missing_ok=True)
        raise


def open_language(root: Path, on_read: Any = None) -> tuple[RawFile, RawFile]:
    """The resident store and the expert store. Vision and MTP stay closed."""
    body = _manifest(root)
    folder = Path(root)
    resident = RawFile(folder / RESIDENT, body["language"], on_read=on_read)
    try:
        experts = RawFile(folder / EXPERTS, body["experts"], on_read=on_read)
    except Exception:
        resident.close()
        raise
    return resident, experts


def open_vision(root: Path) -> RawFile | None:
    """The vision store when the record is complete."""
    folder = Path(root)
    if not ready(folder):
        return None
    body = _manifest(folder)
    return RawFile(folder / VISION, body["vision"])


class RawFile:
    """One raw file. Tensor offsets are from byte 0. Reads are 8 MiB preads."""

    def __init__(self, path: Path, tensors: dict[str, Any], on_read: Any = None) -> None:
        self.path = Path(path)
        self.header = tensors
        self.metadata: dict[str, Any] = {}
        self._data = 0
        self._on_read = on_read
        self._file = self.path.open("rb")

    def close(self) -> None:
        self._file.close()

    def fileno(self) -> int:
        return self._file.fileno()

    def span(self, name: str, shape: tuple[int, ...]) -> tuple[int, int]:
        info = self._info(name, shape)
        start, end = info["data_offsets"]
        return int(start), int(end) - int(start)

    def pread(
        self,
        handle: Any,
        file_off: int,
        nbytes: int,
        *,
        dest_off: int = 0,
        note: Any = None,
    ) -> None:
        self.into(handle, file_off, nbytes, dest_off=dest_off, note=note)

    def into(
        self,
        handle: Any,
        file_off: int,
        nbytes: int,
        *,
        dest_off: int = 0,
        note: Any = None,
    ) -> None:
        from inkling_2_mini.metal41.resident import pread_handle

        pread_handle(
            self.fileno(), handle, file_off, nbytes, dest_off=dest_off, note=note or self._on_read
        )

    def raw(self, name: str) -> tuple[bytes, dict[str, Any]]:
        info = self.header.get(name)
        if not isinstance(info, dict):
            message = f"record is missing {name}"
            raise KeyError(message)
        start, end = info["data_offsets"]
        nbytes = int(end) - int(start)
        buf = bytearray(nbytes)
        from inkling_2_mini.metal41.resident import pread_view

        pread_view(self.fileno(), memoryview(buf), self._data + int(start), self._on_read)
        return bytes(buf), info

    def vector(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        raw, info = self.raw(name)
        found = tuple(info["shape"])
        if found != shape:
            message = f"{name} shape {found} is not {shape}"
            raise ValueError(message)
        if info["dtype"] == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        if info["dtype"] == "F16":
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        if info["dtype"] == "BF16":
            wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            return wide.view(np.float32).reshape(shape).copy()
        message = f"{name} dtype {info['dtype']} is not a float tensor"
        raise ValueError(message)

    def blob(self, offset: int, nbytes: int) -> bytes:
        buf = bytearray(int(nbytes))
        from inkling_2_mini.metal41.resident import pread_view

        pread_view(self.fileno(), memoryview(buf), int(offset), self._on_read)
        return bytes(buf)

    def _info(self, name: str, shape: tuple[int, ...]) -> dict[str, Any]:
        info = self.header.get(name)
        if not isinstance(info, dict):
            message = f"record is missing {name}"
            raise KeyError(message)
        found = tuple(info.get("shape", ()))
        if found != shape:
            message = f"{name} shape {found} is not {shape}"
            raise ValueError(message)
        return info


def _group_for(body: dict[str, Any], filename: str) -> Any:
    for group, name in _GROUPS.items():
        if name == filename:
            return body.get(group)
    return None


def _manifest(root: Path) -> dict[str, Any]:
    if not ready(root):
        message = "record is not complete"
        raise FileNotFoundError(message)
    body = json.loads((Path(root) / MANIFEST).read_text())
    if not isinstance(body, dict):
        message = "record manifest is not an object"
        raise ValueError(message)
    return body


def _keep_language(name: str) -> bool:
    parts = name.split(".")
    if name.startswith("model.visual.") or name.startswith("visual."):
        return False
    if "mtp" in parts or "experts" in parts:
        return False
    return not _DROPPED.intersection(parts)


def _resident(language: Path, dense: Path) -> dict[str, Any]:
    dest = language.parent / f"{RESIDENT}.partial"
    fd, header, base = _open_header(language)
    try:
        with dest.open("wb") as out:
            entries = _append(out, fd, base, header, _keep_language)
    finally:
        os.close(fd)
    fd, header, base = _open_header(dense)
    try:
        with dest.open("ab") as out:
            added = _append(out, fd, base, header, lambda _name: True)
    finally:
        os.close(fd)
    overlap = set(entries).intersection(added)
    if overlap:
        message = f"dense name collides with a language tensor: {sorted(overlap)[0]}"
        raise RuntimeError(message)
    entries.update(added)
    if not entries:
        message = "resident record is empty"
        raise RuntimeError(message)
    return entries


def _mtp(root: Path, language: Path, standalone: Path) -> dict[str, Any]:
    dest = root / f"{MTP}.partial"
    if standalone.is_file():
        return _export_payload(standalone, dest, float16=True)
    fd, header, base = _open_header(language)
    try:
        chosen = {
            name: info
            for name, info in header.items()
            if name != "__metadata__" and "mtp" in name.split(".")
        }
        if not chosen:
            message = "record is missing mtp.safetensors"
            raise FileNotFoundError(message)
        with dest.open("wb") as out:
            return _append(out, fd, base, chosen, lambda _name: True)
    finally:
        os.close(fd)


def _export_payload(src: Path, dest: Path, *, float16: bool = False) -> dict[str, Any]:
    fd, header, base = _open_header(src)
    try:
        size = os.fstat(fd).st_size - base
        with dest.open("wb") as out:
            _copy(fd, base, size, out)
    finally:
        os.close(fd)
    entries = {
        name: {
            "dtype": "F16" if float16 and info.get("dtype") == "BF16" else info.get("dtype"),
            "shape": info.get("shape"),
            "data_offsets": info.get("data_offsets"),
        }
        for name, info in header.items()
        if name != "__metadata__" and isinstance(info, dict)
    }
    if float16:
        _convert_bf16(dest, header)
    if not entries:
        message = f"{src.name} has no tensors"
        raise RuntimeError(message)
    return entries


def _convert_bf16(path: Path, header: dict[str, Any]) -> None:
    fd = os.open(path, os.O_RDWR)
    try:
        for info in header.values():
            if not isinstance(info, dict) or info.get("dtype") != "BF16":
                continue
            start, end = info["data_offsets"]
            offset = 0
            nbytes = int(end) - int(start)
            while offset < nbytes:
                step = min(_PIECE, nbytes - offset)
                raw = os.pread(fd, step, int(start) + offset)
                if len(raw) != step:
                    message = f"{path.name} stopped at {int(start) + offset}"
                    raise OSError(message)
                wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                out = np.ascontiguousarray(wide.view(np.float32).astype(np.float16)).tobytes()
                written = os.pwrite(fd, out, int(start) + offset)
                if written != step:
                    message = f"{path.name} stopped at {int(start) + offset}"
                    raise OSError(message)
                offset += step
    finally:
        os.close(fd)


def _append(
    out: Any,
    fd: int,
    base: int,
    header: dict[str, Any],
    keep: Callable[[str], bool],
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, info in header.items():
        if name == "__metadata__" or not isinstance(info, dict) or not keep(name):
            continue
        start, end = info["data_offsets"]
        nbytes = int(end) - int(start)
        at = out.tell()
        dtype = info.get("dtype")
        if dtype == "BF16":
            _convert_range(fd, base + int(start), nbytes, out)
            dtype = "F16"
        else:
            _copy(fd, base + int(start), nbytes, out)
        entries[name] = {
            "dtype": dtype,
            "shape": info.get("shape"),
            "data_offsets": [at, at + nbytes],
        }
    return entries


def _convert_range(fd: int, start: int, nbytes: int, out: Any) -> None:
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


def _copy(fd: int, start: int, nbytes: int, out: Any) -> None:
    offset = 0
    while offset < nbytes:
        step = min(_PIECE, nbytes - offset)
        raw = os.pread(fd, step, start + offset)
        if len(raw) != step:
            message = f"read stopped at {start + offset}"
            raise OSError(message)
        out.write(raw)
        offset += step


def _open_header(path: Path) -> tuple[int, dict[str, Any], int]:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = int.from_bytes(os.pread(fd, 8, 0), "little")
        raw = os.pread(fd, size, 8)
        if len(raw) != size:
            message = f"{path.name} header is short"
            raise OSError(message)
        header = json.loads(raw)
        if not isinstance(header, dict):
            message = f"{path.name} header is not an object"
            raise ValueError(message)
        return fd, header, 8 + size
    except Exception:
        os.close(fd)
        raise


def _matches_payload(src: Path, dest: Path) -> None:
    src_fd, _header, base = _open_header(src)
    try:
        dest_fd = os.open(dest, os.O_RDONLY)
        try:
            size = os.fstat(src_fd).st_size - base
            if os.fstat(dest_fd).st_size != size:
                message = f"{dest.name} does not match {src.name}"
                raise RuntimeError(message)
            offset = 0
            while offset < size:
                step = min(_PIECE, size - offset)
                left = os.pread(src_fd, step, base + offset)
                right = os.pread(dest_fd, step, offset)
                if left != right or len(left) != step:
                    message = f"{dest.name} does not match {src.name}"
                    raise RuntimeError(message)
                offset += step
        finally:
            os.close(dest_fd)
    finally:
        os.close(src_fd)


def _check_groups(root: Path, groups: dict[str, dict[str, Any]]) -> None:
    for group, filename in _GROUPS.items():
        entries = groups[group]
        if not entries:
            message = f"{filename} has no tensors"
            raise RuntimeError(message)
        path = root / f"{filename}.partial"
        size = path.stat().st_size
        for name, info in entries.items():
            offsets = info.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                message = f"{name} has no data range"
                raise RuntimeError(message)
            start, end = int(offsets[0]), int(offsets[1])
            if start < 0 or end < start or end > size:
                message = f"{name} is outside {filename}"
                raise RuntimeError(message)


def _write_manifest(root: Path, groups: dict[str, dict[str, Any]]) -> None:
    files = {name: (root / name).stat().st_size for name in _GROUPS.values()}
    body = {
        "format": FORMAT,
        "complete": True,
        "pipeline": "inkling_2_mini",
        "config": config_digest(root),
        "files": files,
        **groups,
    }
    partial = root / f"{MANIFEST}.partial"
    partial.write_text(json.dumps(body, separators=(",", ":")) + "\n")
    partial.replace(root / MANIFEST)


def _drop_sources(root: Path) -> None:
    folder = Path(root)
    for path in folder.glob("*.safetensors"):
        if path.is_file():
            path.unlink()
    (folder / "model.safetensors.index.json").unlink(missing_ok=True)


__all__ = [
    "EXPERTS",
    "FORMAT",
    "MANIFEST",
    "MTP",
    "RESIDENT",
    "VISION",
    "RawFile",
    "open_language",
    "open_vision",
    "publish",
    "ready",
    "seal",
]
