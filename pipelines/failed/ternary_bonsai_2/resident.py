"""Device image of the text tensors this forward is allowed to keep.

The wired spans are read in file order into the device image. Vision stays in
the record file. An unwired feed-forward layer is three range reads into the
stage at the end of the image.
"""

from __future__ import annotations

import mmap
from typing import Any, BinaryIO

from ternary_bonsai_2.budget import Group, Wire, layout_copies

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
    ) -> None:
        copies, stage_base, total = layout_copies(group, mtp=wire.mtp, wired=wire.wired)
        buffer = session.allocate(max(total, 16))
        if buffer is None:
            message = f"device refused a {total} byte buffer"
            raise RuntimeError(message)
        mapping = mmap.mmap(record.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(buffer.contents().as_buffer(max(total, 16)))
        try:
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
        self._map = mapping
        self._view = view
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
        found: dict[str, tuple[str, int, int]] = {}
        for name, file_offset, span in self.group.layers[layer]:
            for part in _FFN_PARTS:
                if name.endswith(part):
                    found[part] = (name, file_offset, span)
        missing = [part for part in _FFN_PARTS if part not in found]
        if missing:
            message = f"layer {layer} is missing {missing[0]}"
            raise RuntimeError(message)
        cursor = self.stage_base
        staged: dict[str, int] = {}
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
        _copy(self._map, start, view, 0, nbytes, None)
        return buffer

    def close(self) -> None:
        mapping = self._map
        self._map = None
        self._view = None
        self.buffer = None
        self.offsets = {}
        self._staged_offsets = {}
        if mapping is not None:
            mapping.close()


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
