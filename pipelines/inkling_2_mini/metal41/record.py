"""Raw weight record for an 8 MiB pread.

``resident.bin`` holds the float language planes, including float16 attention
and the float16 output projection. ``experts.bin`` holds the Metal fp4 and MXFP8 expert payloads.
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
    from inkling_2_mini.metal41.weights import checkpoint_file

    if not checkpoint_file(root).is_file():
        message = "the language shard is missing"
        raise FileNotFoundError(message)
    bake(root, log=log)
    ensure_pack(root, log=log)
    seal(root, log=log)


def seal(
    root: Path,
    log: Callable[[str], None] | None = None,
    *,
    language: Path | None = None,
) -> None:
    """Write the record from the safetensors packs already in ``root``.

    ``language`` is the BF16 folder, or one shard when it stays outside ``root``. The partial
    resident file is written beside the dense pack.
    """
    root = Path(root)
    if ready(root):
        _drop_sources(root)
        return
    from inkling_2_mini.metal41.resident import EXPERT_PACK
    from inkling_2_mini.metal41.vision import PACK_NAME
    from inkling_2_mini.metal41.weights import DENSE_PACK, checkpoint_file, safetensor_files

    if language is not None:
        shards = safetensor_files(language)
    else:
        shards = [checkpoint_file(root)]
    experts = root / EXPERT_PACK
    dense = root / DENSE_PACK
    vision = root / PACK_NAME
    missing = [path.name for path in (*shards, experts, vision) if not path.is_file()]
    if missing:
        message = f"record is missing {', '.join(missing)}"
        raise FileNotFoundError(message)
    mtp = root / "mtp.safetensors"
    if not mtp.is_file():
        sibling = shards[0].parent / "mtp.safetensors"
        if sibling.is_file():
            mtp = sibling
    try:
        groups = {
            "language": _resident(shards, root, dense if dense.is_file() else None),
            "experts": _export_payload(experts, root / f"{EXPERTS}.partial"),
            "vision": _export_payload(vision, root / f"{VISION}.partial"),
            "mtp": _mtp(root, shards, mtp),
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
    return True


def _resident(shards: list[Path], root: Path, dense: Path | None) -> dict[str, Any]:
    dest = Path(root) / f"{RESIDENT}.partial"
    entries: dict[str, Any] = {}
    for language in shards:
        fd, header, base = _open_header(language)
        try:
            with dest.open("ab" if entries else "wb") as out:
                added = _append(out, fd, base, header, _keep_language)
        finally:
            os.close(fd)
        overlap = set(entries).intersection(added)
        if overlap:
            message = f"language name collides across shards: {sorted(overlap)[0]}"
            raise RuntimeError(message)
        entries.update(added)
    if dense is not None:
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


def _mtp(root: Path, shards: list[Path], standalone: Path) -> dict[str, Any]:
    dest = root / f"{MTP}.partial"
    if standalone.is_file():
        return _export_payload(standalone, dest, float16=True)
    entries: dict[str, Any] = {}
    for language in shards:
        fd, header, base = _open_header(language)
        try:
            chosen = {
                name: info
                for name, info in header.items()
                if name != "__metadata__" and "mtp" in name.split(".")
            }
            if not chosen:
                continue
            with dest.open("ab" if entries else "wb") as out:
                added = _append(out, fd, base, chosen, lambda _name: True)
        finally:
            os.close(fd)
        overlap = set(entries).intersection(added)
        if overlap:
            message = f"mtp name collides across shards: {sorted(overlap)[0]}"
            raise RuntimeError(message)
        entries.update(added)
    if not entries:
        message = "record is missing mtp.safetensors"
        raise FileNotFoundError(message)
    return entries


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


def _copy_sidecars(src: Path, dest: Path) -> None:
    """Copy tokenizer and config files into ``dest`` without touching ``src``."""
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    for path in dest.glob("*.safetensors"):
        path.unlink()
    for path in src.rglob("*"):
        if path.name.endswith(".incomplete") or not path.is_file() or path.suffix == ".safetensors":
            continue
        rel = path.relative_to(src)
        target = dest / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _index_shards(shards: list[Path]) -> dict[str, tuple[Path, dict[str, Any], int]]:
    index: dict[str, tuple[Path, dict[str, Any], int]] = {}
    for shard in shards:
        fd, header, base = _open_header(shard)
        os.close(fd)
        for name, info in header.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            index[name] = (shard, info, base)
    return index


def _read_f32_tensor(
    index: dict[str, tuple[Path, dict[str, Any], int]],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    found = index.get(name)
    if found is None:
        message = f"checkpoint is missing {name}"
        raise KeyError(message)
    shard, info, base = found
    if tuple(info.get("shape", ())) != shape or info.get("dtype") != "F32":
        message = f"{name} is not F32 {shape}"
        raise ValueError(message)
    start, end = info["data_offsets"]
    fd = os.open(shard, os.O_RDONLY)
    try:
        raw = os.pread(fd, int(end) - int(start), base + int(start))
    finally:
        os.close(fd)
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()


def _nvfp4_experts(
    shards: list[Path],
    dest: Path,
    log: Callable[[str], None] | None,
) -> dict[str, Any]:
    """Copy NVFP4 and MXFP8 experts verbatim into ``dest`` for direct pread."""
    from inkling_2_mini.metal41.layout import (
        EXPERTS as NUM_EXPERTS,
        HIDDEN,
        INTERMEDIATE,
        LAYERS,
        NVFP4_LAYERS,
        OUTER_SCALE_DENOMINATOR,
    )

    index = _index_shards(shards)
    entries: dict[str, Any] = {}
    fds: dict[Path, int] = {}
    try:
        with dest.open("wb") as out:
            for layer in range(LAYERS):
                is_nvfp4 = layer < NVFP4_LAYERS
                for kind, rows, cols in (("w13", INTERMEDIATE * 2, HIDDEN), ("w2", HIDDEN, INTERMEDIATE)):
                    stem = f"model.llm.layers.{layer}.mlp.experts.{kind}"
                    data_shape = [NUM_EXPERTS, rows, cols // 2 if is_nvfp4 else cols]
                    scale_shape = [NUM_EXPERTS, rows, cols // 16 if is_nvfp4 else cols // 32]
                    for suffix, tag, shape in (
                        ("_weight", "data", data_shape),
                        ("_weight_scale", "scale", scale_shape),
                    ):
                        src_name = f"{stem}{suffix}"
                        found = index.get(src_name)
                        if found is None:
                            message = f"checkpoint is missing {src_name}"
                            raise KeyError(message)
                        shard, info, base = found
                        if list(info.get("shape", ())) != shape:
                            message = f"{src_name} shape {info.get('shape')} is not {shape}"
                            raise ValueError(message)
                        fd = fds.get(shard)
                        if fd is None:
                            fd = os.open(shard, os.O_RDONLY)
                            fds[shard] = fd
                        start, end = info["data_offsets"]
                        nbytes = int(end) - int(start)
                        at = out.tell()
                        _copy(fd, base + int(start), nbytes, out)
                        entries[f"layers.{layer}.{kind}.{tag}"] = {
                            "dtype": info.get("dtype"),
                            "shape": shape,
                            "data_offsets": [at, at + nbytes],
                        }
                    if is_nvfp4:
                        amax = _read_f32_tensor(index, f"{stem}_weight_amax", (NUM_EXPERTS,))
                        denom = _read_f32_tensor(
                            index, f"{stem}_weight_outer_scale_denominator", (1,)
                        )
                        if float(denom[0]) != OUTER_SCALE_DENOMINATOR:
                            message = (
                                f"{stem} outer scale denominator is not {OUTER_SCALE_DENOMINATOR}"
                            )
                            raise ValueError(message)
                        global_scale = np.ascontiguousarray(
                            (amax / denom[0]).astype(np.float32)
                        ).tobytes()
                        at = out.tell()
                        out.write(global_scale)
                        entries[f"layers.{layer}.{kind}.global_scale"] = {
                            "dtype": "F32",
                            "shape": [NUM_EXPERTS],
                            "data_offsets": [at, at + len(global_scale)],
                        }
                if log is not None:
                    log(f"repacked experts layer {layer}")
    finally:
        for fd in fds.values():
            os.close(fd)
    return entries


def repack_nvfp4(
    src: Path,
    dest: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Seal an NVFP4 checkpoint in ``src`` into a raw pread record in ``dest``.

    Layers 0–19 stay NVFP4 (packed E2M1, E4M3 block scales, fp32 global scale).
    Layers 20–23 stay MXFP8. BF16 language and MTP planes become float16.
    Source files in ``src`` are left untouched.
    """
    from inkling_2_mini.metal41.commands import register_folder
    from inkling_2_mini.metal41.vision import PACK_NAME, ensure_pack
    from inkling_2_mini.metal41.weights import safetensor_files

    source = Path(src).resolve()
    output = Path(dest).resolve()
    if source == output:
        message = "repack writes a new folder"
        raise ValueError(message)
    shards = safetensor_files(source)
    _copy_sidecars(source, output)
    if ready(output):
        if log is not None:
            log("record is current")
        register_folder(output, alias=None, source=source)
        return output
    ensure_pack(output, log=log, language=source)
    vision = output / PACK_NAME
    mtp = source / "mtp.safetensors"
    try:
        groups = {
            "language": _resident(shards, output, None),
            "experts": _nvfp4_experts(shards, output / f"{EXPERTS}.partial", log),
            "vision": _export_payload(vision, output / f"{VISION}.partial"),
            "mtp": _mtp(output, shards, mtp),
        }
        _check_groups(output, groups)
        for name in (RESIDENT, EXPERTS, VISION, MTP):
            partial = output / f"{name}.partial"
            partial.replace(output / name)
            if log is not None:
                log(f"wrote {name}")
        _matches_payload(vision, output / VISION)
        _write_manifest(output, groups)
        if not ready(output):
            message = "record does not match config.json"
            raise RuntimeError(message)
        _drop_sources(output)
    except Exception:
        for name in (RESIDENT, EXPERTS, VISION, MTP, MANIFEST):
            (output / f"{name}.partial").unlink(missing_ok=True)
        raise
    register_folder(output, alias=None, source=source)
    if log is not None:
        log(f"registered {output.name}")
    return output


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
    "repack_nvfp4",
    "seal",
]
