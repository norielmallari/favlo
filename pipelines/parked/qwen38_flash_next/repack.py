"""Copy a version-5 all-fp4 record into one file per quant.

The images are already fp4. This does not quantize them again.
"""

from __future__ import annotations

import json
import mmap
import os
import sys
from pathlib import Path

from qwen38_flash_next.allocate import preallocate
from qwen38_flash_next.convert import pread_exact
from qwen38_flash_next.formats import QUANT_LAYOUT_VERSION, RECORD_VERSION, quant_file

_SKIP = {"manifest.json", "quantize_progress.json"}
_DEST_NAME = "Qwen3.8-Flash-Next-fp4"


def version5_place(hot: list[int], supers: list[int], experts: int) -> list[tuple[str, int]]:
    """Source file and row for each expert id in a version-5 layer.

    ``hot`` is the hot-file order with the pin removed. Cold is every other id,
    ascending. The returned row is the image's place in that source file.
    """
    hot_at = {int(expert): row for row, expert in enumerate(hot)}
    super_at = {int(expert): row for row, expert in enumerate(supers)}
    if set(hot_at) & set(super_at):
        message = "a super expert is still in the hot file"
        raise RuntimeError(message)
    if any(expert < 0 or expert >= experts for expert in (*hot_at, *super_at)):
        message = f"expert id is outside 0..{experts - 1}"
        raise RuntimeError(message)
    cold = [
        expert
        for expert in range(experts)
        if expert not in hot_at and expert not in super_at
    ]
    cold_at = {expert: row for row, expert in enumerate(cold)}
    placed: list[tuple[str, int]] = []
    for expert in range(experts):
        if expert in super_at:
            placed.append(("super", super_at[expert]))
        elif expert in hot_at:
            placed.append(("hot", hot_at[expert]))
        else:
            placed.append(("cold", cold_at[expert]))
    return placed


def rewrite_fp4(source: Path, dest: Path | None = None) -> Path:
    """Write ``dest`` from a uniform fp4 version-5 directory. ``dest`` defaults beside it."""
    source = source.resolve()
    if dest is None:
        dest = source.parent / _DEST_NAME
    dest = dest.resolve()
    if dest == source:
        message = "rewrite destination is the version-5 directory"
        raise RuntimeError(message)
    manifest = json.loads((source / "manifest.json").read_text())
    if int(manifest.get("version") or 0) != RECORD_VERSION or not manifest.get("complete"):
        message = f"{source.name} is not a complete version-{RECORD_VERSION} record"
        raise RuntimeError(message)
    bits = (
        int(manifest["hot_bits"]),
        int(manifest["cold_bits"]),
        int(manifest["super_bits"]),
        int(manifest["mtp_bits"]),
    )
    if bits != (4, 4, 4, 4):
        message = f"{source.name} is not uniform fp4"
        raise RuntimeError(message)
    slot = int(manifest["expert_slot"])
    if int(manifest["cold_slot"]) != slot or int(manifest["super_slot"]) != slot:
        message = "fp4 strides differ"
        raise RuntimeError(message)
    if int(manifest["expert_live"]) != slot:
        message = "fp4 image is shorter than its stride"
        raise RuntimeError(message)
    layers = int(manifest["layers"])
    experts = int(manifest["experts"])
    hot = manifest["hot"]
    supers = manifest["super"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "experts").mkdir(exist_ok=True)
    for child in source.iterdir():
        if child.name in _SKIP or child.name == "experts":
            continue
        if child.is_dir():
            message = f"unexpected directory in the record: {child.name}"
            raise RuntimeError(message)
        _link(child, dest / child.name)
    for layer in range(layers):
        _write_layer(
            source,
            dest,
            layer,
            version5_place(list(hot[layer]), list(supers[layer]), experts),
            experts,
            slot,
        )
        print(f"layer {layer} fp4", file=sys.stderr)
    _check_images(source, dest, manifest, slot)
    written = dict(manifest)
    for key in (
        "hot",
        "hot_count",
        "cold_count",
        "hot_bits",
        "cold_bits",
        "hot_format",
        "cold_format",
    ):
        written.pop(key, None)
    written["version"] = QUANT_LAYOUT_VERSION
    written["expert_quant"] = 4
    written["expert_format"] = "fp4_e2m1"
    path = dest / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(written))
    os.replace(tmp, path)
    return dest


def _link(source: Path, dest: Path) -> None:
    if dest.exists():
        dest_stat = dest.stat()
        source_stat = source.stat()
        same = dest_stat.st_ino == source_stat.st_ino and dest_stat.st_dev == source_stat.st_dev
        if same or dest_stat.st_size == source_stat.st_size:
            return
        message = f"{dest.name} already exists and does not match the source"
        raise RuntimeError(message)
    try:
        os.link(source, dest)
    except OSError:
        _copy_file(source, dest)


def _copy_file(source: Path, dest: Path) -> None:
    src = os.open(source, os.O_RDONLY)
    dst = os.open(dest, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        offset = 0
        total = source.stat().st_size
        while offset < total:
            chunk = min(1 << 20, total - offset)
            os.sendfile(dst, src, offset, chunk)
            offset += chunk
    finally:
        os.close(src)
        os.close(dst)


def _write_layer(
    source: Path,
    dest: Path,
    layer: int,
    placed: list[tuple[str, int]],
    experts: int,
    slot: int,
) -> None:
    path = dest / "experts" / quant_file(layer, 4)
    expected = experts * slot
    if path.is_file() and path.stat().st_size == expected:
        return
    names = {"hot": "hot", "cold": "cold", "super": "super"}
    fds: dict[str, int] = {}
    out = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    scratch = mmap.mmap(-1, slot)
    try:
        preallocate(out, expected)
        for kind in ("hot", "cold", "super"):
            src = source / "experts" / f"{layer:02d}-{names[kind]}.bin"
            if kind == "super" and not src.is_file():
                continue
            fds[kind] = os.open(src, os.O_RDONLY)
        for expert, (kind, row) in enumerate(placed):
            view = pread_exact(fds[kind], slot, row * slot, scratch)
            blob = bytes(view)
            del view
            wrote = os.pwrite(out, blob, expert * slot)
            if wrote != slot:
                message = f"layer {layer} expert {expert} wrote {wrote} bytes"
                raise RuntimeError(message)
        os.fsync(out)
    finally:
        scratch.close()
        os.close(out)
        for fd in fds.values():
            os.close(fd)


def _check_images(source: Path, dest: Path, manifest: dict, slot: int) -> None:
    """One ordinary image and one pin image must match the version-5 bytes."""
    experts = int(manifest["experts"])
    hot = [int(expert) for expert in manifest["hot"][0]]
    pin_layer = 2
    pin_expert = 322
    ordinary = next(expert for expert in range(experts) if expert not in set(hot))
    _compare(source, dest, manifest, 0, ordinary, slot)
    _compare(source, dest, manifest, pin_layer, pin_expert, slot)


def _compare(
    source: Path,
    dest: Path,
    manifest: dict,
    layer: int,
    expert: int,
    slot: int,
) -> None:
    experts = int(manifest["experts"])
    placed = version5_place(list(manifest["hot"][layer]), list(manifest["super"][layer]), experts)
    kind, row = placed[expert]
    src_path = source / "experts" / f"{layer:02d}-{kind}.bin"
    dst_path = dest / "experts" / quant_file(layer, 4)
    src = os.open(src_path, os.O_RDONLY)
    dst = os.open(dst_path, os.O_RDONLY)
    scratch = mmap.mmap(-1, slot)
    try:
        left = bytes(pread_exact(src, slot, row * slot, scratch))
        right = bytes(pread_exact(dst, slot, expert * slot, scratch))
    finally:
        scratch.close()
        os.close(src)
        os.close(dst)
    if left != right:
        message = f"layer {layer} expert {expert} does not match the version-5 image"
        raise RuntimeError(message)


def main() -> int:
    from favlo.cli import repo_root

    root = repo_root() / "models"
    source = root / "Qwen3.8-Flash-Next-h128-fp4-c384-fp4"
    dest = rewrite_fp4(source)
    print(dest.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
