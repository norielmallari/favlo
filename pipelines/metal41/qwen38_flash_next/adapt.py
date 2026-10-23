"""Lull repack. One promotion and one demotion, in the served record."""

from __future__ import annotations

import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from favlo.registry import load_registry
from qwen38_flash_next.allocate import preallocate
from qwen38_flash_next.convert import (
    ShardStore,
    _expert_image,
    _expert_matrix,
    _plane_format,
    quant_groups,
)
from qwen38_flash_next.firing import (
    FiringStore,
    RouteScratch,
    heartbeat_picks,
)
from qwen38_flash_next.formats import quant_file, tier_image_bytes
from qwen38_flash_next.session import _log

QUIET_SECONDS = 0.25
_lock = threading.Lock()
_pending: dict[int, list[tuple[str, RouteScratch]]] = {}
_stop_pack: set[str] = set()
_warned: set[str] = set()


def telemetry_dir(folder: str) -> Path:
    return Path.home() / ".favlo" / "telemetry" / folder


def _store(folder: str) -> FiringStore:
    return FiringStore(telemetry_dir(folder) / "firing.sqlite")


def _source(folder: str) -> Path | None:
    row = load_registry().get("models", {}).get(folder) or {}
    source = row.get("source")
    if not isinstance(source, str) or not source:
        return None
    path = Path(source).expanduser()
    if not (path / "config.json").is_file():
        return None
    return path


def _grid(manifest: dict[str, Any], layers: int, experts: int) -> list[list[int]]:
    quant = manifest.get("expert_quant")
    if isinstance(quant, int):
        return [[int(quant) for _ in range(experts)] for _ in range(layers)]
    return [[int(bits) for bits in row] for row in quant]


def _pins(manifest: dict[str, Any], layers: int) -> list[set[int]]:
    raw = manifest.get("super") or [[] for _ in range(layers)]
    return [{int(expert) for expert in row} for row in raw]


def _read_image(root: Path, layer: int, expert: int, row: list[int]) -> bytes:
    bits = int(row[expert])
    rank = sum(1 for index, width in enumerate(row) if int(width) == bits and index < expert)
    live, slot = tier_image_bytes(bits)
    fd = os.open(root / "experts" / quant_file(layer, bits), os.O_RDONLY)
    try:
        blob = os.pread(fd, live, rank * slot)
    finally:
        os.close(fd)
    if len(blob) != live:
        message = f"layer {layer} expert {expert} image is short"
        raise RuntimeError(message)
    return blob


def _write_width(path: Path, ids: list[int], images: dict[int, bytes], slot: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        preallocate(fd, len(ids) * slot)
        for rank, expert in enumerate(ids):
            os.pwrite(fd, images[expert], rank * slot)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def rewrite_layer(
    root: Path,
    layer: int,
    old: list[int],
    new: list[int],
    fresh: dict[int, bytes],
) -> None:
    """Write one layer's quant files. A fresh image replaces that expert."""
    images = {
        expert: fresh[expert] if expert in fresh else _read_image(root, layer, expert, old)
        for expert in range(len(new))
    }
    groups = quant_groups(new)
    for bits, ids in groups.items():
        _live, slot = tier_image_bytes(bits)
        _write_width(root / "experts" / quant_file(layer, bits), ids, images, slot)
    for bits in (8, 4, 2):
        if bits in groups:
            continue
        path = root / "experts" / quant_file(layer, bits)
        if path.is_file():
            path.unlink()


def write_grid_manifest(path: Path, manifest: dict[str, Any], grid: list[list[int]]) -> None:
    body = dict(manifest)
    widths = sorted({bits for row in grid for bits in row}, reverse=True)
    if len(widths) == 1:
        live, slot = tier_image_bytes(widths[0])
        body["expert_quant"] = widths[0]
        body["expert_format"] = _plane_format(widths[0])
        body["expert_live"] = live
        body["expert_slot"] = slot
        body.pop("quants", None)
    else:
        body["expert_quant"] = grid
        body["quants"] = {
            str(bits): {
                "format": _plane_format(bits),
                "live": tier_image_bytes(bits)[0],
                "slot": tier_image_bytes(bits)[1],
            }
            for bits in widths
        }
        for key in ("expert_format", "expert_live", "expert_slot"):
            body.pop(key, None)
    body["complete"] = True
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)


def _descriptors_open(served: Path) -> bool:
    """True when any process holds a file in the served record."""
    if not served.is_dir():
        return False
    found = subprocess.run(
        ["lsof", "-t", "+D", str(served)],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(found.stdout.strip())


def _expert_bytes(source: Path, layer: int, expert: int, bits: int) -> bytes:
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    store = ShardStore(source)
    try:
        gate = _expert_matrix(store, f"{prefix}.gate_up_proj", expert)
        down = _expert_matrix(store, f"{prefix}.down_proj", expert)
        image = _expert_image(gate, down, int(bits))
    finally:
        store.close()
    live, _slot = tier_image_bytes(int(bits))
    if len(image) != live:
        message = f"expert image is {len(image)} bytes, expected {live}"
        raise RuntimeError(message)
    return image


def _apply(
    served: Path,
    source: Path,
    store: FiringStore,
    move: tuple[int, int, int],
    total: int,
) -> bool:
    """Write one expert. False when a descriptor opened before the write."""
    layer, expert, bits = move
    manifest = json.loads((served / "manifest.json").read_text())
    layers = int(manifest["layers"])
    experts = int(manifest["experts"])
    current = _grid(manifest, layers, experts)
    if int(current[layer][expert]) == int(bits):
        store.mark_packed(layer, expert, total)
        return True
    image = _expert_bytes(source, layer, expert, int(bits))
    if _descriptors_open(served):
        return False
    updated = [row[:] for row in current]
    updated[layer][expert] = int(bits)
    rewrite_layer(served, layer, current[layer], updated[layer], {expert: image})
    write_grid_manifest(served / "manifest.json", manifest, updated)
    store.mark_packed(layer, expert, total)
    _log(f"repack layer {layer} expert {expert} to {bits}")
    return True


def heartbeat(
    folder: str,
    record: str,
    served: Path,
    rng: random.Random | None = None,
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int] | None] | None:
    """Promote one expert and demote another. None when the record is open."""
    if folder in _stop_pack:
        return None
    source = _source(folder)
    if source is None:
        if folder not in _warned:
            _warned.add(folder)
            _log("BF16 source is missing; expert repack stopped")
        _stop_pack.add(folder)
        return None
    if _descriptors_open(served):
        return None
    draw = rng or random.Random()
    store = _store(folder)
    try:
        manifest = json.loads((served / "manifest.json").read_text())
        layers = int(manifest["layers"])
        experts = int(manifest["experts"])
        served_grid = _grid(manifest, layers, experts)
        eligible, hits = store.window(record, layers, experts)
        total = store.eligible_total(record)
        up, down = heartbeat_picks(
            served_grid,
            hits,
            eligible=eligible,
            pins=_pins(manifest, layers),
            cooled=store.cooled(total),
            rng=draw,
        )
        if up is not None and not _apply(served, source, store, up, total):
            up = None
        if down is not None and _descriptors_open(served):
            down = None
        elif down is not None and not _apply(served, source, store, down, total):
            down = None
        return up, down
    finally:
        store.close()


def _analyze(folder: str, record: str, served: Path) -> None:
    """Promote one expert and demote one when the record is not open."""
    heartbeat(folder, record, served)


def _lull(runtime: Any, folder: str, record: str, served: Path, token: int) -> None:
    time.sleep(QUIET_SECONDS)
    with _lock:
        if getattr(runtime, "_firing_open", False):
            return
        batch = _pending.pop(token, [])
    if not batch:
        return
    store = _store(folder)
    try:
        for identity, routes in batch:
            if identity != record:
                continue
            store.commit_request(identity, routes, complete=True)
    finally:
        store.close()
    if getattr(runtime, "_firing_open", False):
        return
    try:
        _analyze(folder, record, served)
    except Exception as exc:
        _log(f"expert repack stopped: {type(exc).__name__}: {exc}")
        _stop_pack.add(folder)


def _width_counts(files: Any) -> tuple[int, int, int]:
    """fp8, fp4, and int2 counts. A uniform record uses its stored width."""
    fp8 = 0
    fp4 = 0
    int2 = 0
    if files is None:
        return fp8, fp4, int2
    grid = getattr(files, "_quant_grid", None)
    if grid is not None:
        for row in grid:
            for bits in row:
                width = int(bits)
                if width == 8:
                    fp8 += 1
                elif width == 4:
                    fp4 += 1
                elif width == 2:
                    int2 += 1
        return fp8, fp4, int2
    manifest = getattr(files, "manifest", None) or {}
    quant = manifest.get("expert_quant")
    if not isinstance(quant, int):
        return fp8, fp4, int2
    total = int(manifest.get("layers") or 0) * int(manifest.get("experts") or 0)
    if quant == 8:
        fp8 = total
    elif quant == 4:
        fp4 = total
    elif quant == 2:
        int2 = total
    return fp8, fp4, int2


def _stamp_turn(routes: RouteScratch, runtime: Any) -> None:
    fp8, fp4, int2 = _width_counts(getattr(runtime, "files", None))
    routes.note_turn(
        cached_tokens=int(getattr(runtime, "_cached_tokens", 0) or 0),
        finished_at=int(time.time()),
        prefill_seconds=float(getattr(runtime, "_prefill_seconds", 0.0) or 0.0),
        decode_seconds=float(getattr(runtime, "_decode_seconds", 0.0) or 0.0),
        fp8_experts=fp8,
        fp4_experts=fp4,
        int2_experts=int2,
    )


def open_request(runtime: Any) -> None:
    routes = getattr(runtime, "routes", None)
    if routes is not None:
        routes.begin()
    runtime._firing_open = True


def finish_request(runtime: Any, selection: Any, *, complete: bool) -> None:
    """Queue this request. The daemon runs the lull after the request closes."""
    runtime._firing_open = False
    routes = getattr(runtime, "routes", None)
    if routes is None or selection is None:
        return
    if not complete:
        routes.drop_decode()
        routes.begin()
        return
    _stamp_turn(routes, runtime)
    snap = routes.snapshot()
    routes.begin()
    folder = selection.folder
    record = str(runtime.files.root)
    token = id(runtime)
    with _lock:
        _pending.setdefault(token, []).append((record, snap))
    runtime._idle_heartbeat = (folder, record, Path(runtime.files.root), token)


def idle_heartbeat(runtime: Any, selection: Any = None) -> None:
    """Write the queued routes and repack. The daemon calls this."""
    del selection
    job = getattr(runtime, "_idle_heartbeat", None)
    if job is None:
        return
    runtime._idle_heartbeat = None
    _lull(runtime, *job)

