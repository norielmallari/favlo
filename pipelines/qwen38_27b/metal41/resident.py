"""Device image of the text tensors this forward is allowed to keep.

Layout 2 reads the wired prefix. When the MTP head is resident that prefix is
one read from the start of the file through the last wired layer. When the head
stays in the file, the text block is one read and the wired layer prefix is the
next, so the MTP bytes are not pulled in. An unwired layer is one read of its
gate, up, and down. Vision is one read of the dense suffix.

An older manifest seeks to each span and writes it at the device offset.
"""

from __future__ import annotations

import mmap
from typing import Any, BinaryIO

from qwen38_27b.metal41.budget import LAYOUT_PREFIX, LAYOUT_SCATTER, Group, Wire, layout_copies

_CHUNK = 8 << 20
_FFN_PARTS = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


class DeviceImage:
    """One Metal buffer of wired tensors plus a feed-forward stage."""

    def __init__(
        self,
        session: Any,
        record: BinaryIO,
        group: Group,
        wire: Wire,
        note: Any = None,
        layout: int = LAYOUT_SCATTER,
    ) -> None:
        copies, stage_base, total = layout_copies(group, mtp=wire.mtp, wired=wire.wired)
        buffer = session.allocate(max(total, 16))
        if buffer is None:
            message = f"device refused a {total} byte buffer"
            raise RuntimeError(message)
        mapping = mmap.mmap(record.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(buffer.contents().as_buffer(max(total, 16)))
        try:
            if layout == LAYOUT_PREFIX:
                _fill_prefix(record, view, copies, note)
            else:
                _fill(record, view, copies, note)
            if note is not None and wire.stage:
                note(wire.stage)
        except Exception:
            mapping.close()
            raise
        self.buffer = buffer
        self.nbytes = total
        self.stage_base = stage_base
        self.stage = wire.stage
        self.wired = frozenset(wire.wired)
        self.offsets = {name: device for name, _file, device, _span in copies}
        self.group = group
        self.layout = layout
        self._map = mapping
        self._view = view
        self._file = record
        self.staged: int | None = None
        self.stage_busy = False
        self._staged_offsets: dict[str, int] = {}

    def offset(self, name: str) -> int:
        if name in self.offsets:
            return self.offsets[name]
        if name in self._staged_offsets:
            return self._staged_offsets[name]
        message = f"{name} is not on the device"
        raise KeyError(message)

    def load_layer(self, layer: int) -> None:
        """Copy one unwired language feed-forward layer into the stage."""
        if layer in self.wired:
            return
        found = _layer_parts(self.group, layer)
        if self.layout == LAYOUT_PREFIX:
            start, span, staged = _contiguous_layer(layer, found, self.stage_base)
            if span > self.stage:
                message = f"layer {layer} does not fit the feed-forward stage"
                raise RuntimeError(message)
            _read_into(self._file, self._view, start, self.stage_base, span, None)
        else:
            cursor = self.stage_base
            staged = {}
            for part in _FFN_PARTS:
                name, file_offset, span = found[part]
                _copy(self._map, file_offset, self._view, cursor, span, None)
                staged[name] = cursor
                cursor += span
            if cursor - self.stage_base > self.stage:
                message = f"layer {layer} does not fit the feed-forward stage"
                raise RuntimeError(message)
        self._staged_offsets = staged
        self.staged = layer

    def load_span(self, session: Any, start: int, nbytes: int) -> Any:
        """Copy one file range into a new buffer. Vision weights stay out of the text image."""
        buffer = session.allocate(nbytes)
        if buffer is None:
            message = f"device refused a {nbytes} byte vision buffer"
            raise RuntimeError(message)
        view = memoryview(buffer.contents().as_buffer(nbytes))
        if self.layout == LAYOUT_PREFIX:
            _read_into(self._file, view, start, 0, nbytes, None)
        else:
            _copy(self._map, start, view, 0, nbytes, None)
        return buffer

    def close(self) -> None:
        mapping = self._map
        self._map = None
        self._view = None
        self.buffer = None
        self.offsets = {}
        self._staged_offsets = {}
        if hasattr(self, "_file"):
            self._file = None
        if mapping is not None:
            _discard_mapping(mapping)
            mapping.close()


def _discard_mapping(mapping: mmap.mmap) -> None:
    """Mark the weight file's cached pages reusable before the mapping closes."""
    for advice in (mmap.MADV_FREE, mmap.MADV_FREE_REUSABLE):
        try:
            mapping.madvise(advice)
        except (OSError, BufferError, ValueError):
            continue


def _layer_parts(group: Group, layer: int) -> dict[str, tuple[str, int, int]]:
    found: dict[str, tuple[str, int, int]] = {}
    for name, file_offset, span in group.layers[layer]:
        for part in _FFN_PARTS:
            if name.endswith(part):
                found[part] = (name, file_offset, span)
    missing = [part for part in _FFN_PARTS if part not in found]
    if missing:
        message = f"layer {layer} is missing {missing[0]}"
        raise RuntimeError(message)
    return found


def _contiguous_layer(
    layer: int,
    found: dict[str, tuple[str, int, int]],
    stage_base: int,
) -> tuple[int, int, dict[str, int]]:
    """File start, span, and stage offsets when gate, up, and down are one run."""
    start = 0
    file_cursor = 0
    device = stage_base
    staged: dict[str, int] = {}
    for index, part in enumerate(_FFN_PARTS):
        name, file_offset, span = found[part]
        if index == 0:
            start = file_offset
            file_cursor = file_offset
        elif file_offset != file_cursor:
            message = f"layer {layer} feed-forward is not one span"
            raise RuntimeError(message)
        staged[name] = device
        file_cursor += span
        device += span
    return start, file_cursor - start, staged


def _fill_prefix(
    record: BinaryIO,
    view: memoryview,
    copies: list[tuple[str, int, int, int]],
    note: Any,
) -> None:
    """Read each contiguous wired run straight into the device image.

    A resident MTP head is one prefix of the file. A dropped head leaves a hole,
    so the text block and the wired layer prefix are two reads. Vision and
    unwired layers are not in ``copies``.
    """
    for file_offset, device_offset, span in _runs(copies):
        _read_into(record, view, file_offset, device_offset, span, note)


def _runs(copies: list[tuple[str, int, int, int]]) -> list[tuple[int, int, int]]:
    """``(file offset, device offset, span)`` with adjacent copies merged."""
    runs: list[tuple[int, int, int]] = []
    for _name, file_offset, device_offset, span in copies:
        if span <= 0:
            continue
        if file_offset < 0 or device_offset < 0:
            message = f"resident span at {file_offset} is negative"
            raise RuntimeError(message)
        for file_at, _device_at, size in runs:
            if file_offset < file_at + size and file_offset + span > file_at:
                message = f"resident span at {file_offset} overlaps byte {file_at + size}"
                raise RuntimeError(message)
        if runs:
            file_at, device_at, size = runs[-1]
            if file_offset == file_at + size and device_offset == device_at + size:
                runs[-1] = (file_at, device_at, size + span)
                continue
        runs.append((file_offset, device_offset, span))
    return runs


def _read_into(
    record: BinaryIO,
    view: memoryview,
    file_offset: int,
    device_offset: int,
    span: int,
    note: Any,
) -> None:
    """One sequential read of ``span`` bytes, in pieces of at most 8 MiB."""
    if span <= 0:
        return
    record.seek(file_offset)
    copied = 0
    while copied < span:
        step = min(_CHUNK, span - copied)
        chunk = record.read(step)
        if len(chunk) != step:
            message = f"resident read stopped at {file_offset + copied + len(chunk)}"
            raise RuntimeError(message)
        dest = device_offset + copied
        view[dest : dest + step] = chunk
        copied += step
        if note is not None:
            note(step)


def _fill(
    record: BinaryIO,
    view: memoryview,
    copies: list[tuple[str, int, int, int]],
    note: Any,
) -> None:
    """Read wired spans in file order and place each one at its device offset."""
    cursor = -1
    for _name, file_offset, device_offset, span in sorted(copies, key=lambda item: item[1]):
        if span <= 0:
            continue
        if file_offset < cursor:
            message = f"resident span at {file_offset} overlaps byte {cursor}"
            raise RuntimeError(message)
        if file_offset != cursor:
            record.seek(file_offset)
            cursor = file_offset
        copied = 0
        while copied < span:
            step = min(_CHUNK, span - copied)
            chunk = record.read(step)
            if len(chunk) != step:
                message = f"resident read stopped at {cursor + len(chunk)}"
                raise RuntimeError(message)
            dest = device_offset + copied
            view[dest : dest + step] = chunk
            copied += step
            cursor += step
            if note is not None:
                note(step)


def _copy(
    mapping: mmap.mmap,
    file_offset: int,
    view: memoryview,
    device_offset: int,
    span: int,
    note: Any,
) -> None:
    copied = 0
    while copied < span:
        step = min(_CHUNK, span - copied)
        start = file_offset + copied
        dest = device_offset + copied
        view[dest : dest + step] = mapping[start : start + step]
        copied += step
        if note is not None:
            note(step)
