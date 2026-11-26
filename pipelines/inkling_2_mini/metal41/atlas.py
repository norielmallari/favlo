"""Resident expert images.

A sealed ``experts.bin`` is read straight into device buffers in 8 MiB pieces:
NVFP4 tensors with per-expert tensor scales stay packed for the NVFP4 decode
kernels, Metal fp4 packs and MXFP8 tensors are read as stored, and mixed
fp4/fp8 packs use per-expert offset tables. When every image does not fit
beside the dense weights, the half key/value cache, and the activation buffers,
the atlas keeps eight slots per projection and stages the experts a token
routes before they run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from favlo.memory import device_buffer, memory
from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, stored_k

# Largest stored tensor of one expert: MXFP8 fused gate/up, then the widest block scale.
_MAX_DATA = HIDDEN * HIDDEN
_MAX_SCALE = HIDDEN * (HIDDEN // 16)
CONTEXTS = (131072, 16384, 8192, 4096, 2048, 1024, 512)


def expert_file_bytes(header: dict[str, Any]) -> int:
    """Packed expert weights and their block scales. Norms and amax stay out."""
    total = 0
    for name, info in header.items():
        if ".mlp.experts." not in name:
            continue
        if not (name.endswith("_weight") or name.endswith("_weight_scale")):
            continue
        start, end = info["data_offsets"]
        total += int(end) - int(start)
    if total < 1:
        message = "checkpoint has no expert tensors"
        raise ValueError(message)
    return total


def image_bytes() -> int:
    """One expert, both projections, for the larger of the two stored quants."""
    nv_w13 = HIDDEN * (HIDDEN // 2) + HIDDEN * (HIDDEN // 16)
    nv_w2 = HIDDEN * (INTERMEDIATE // 2) + HIDDEN * (INTERMEDIATE // 16)
    mx_w13 = HIDDEN * HIDDEN + HIDDEN * (HIDDEN // 32)
    mx_w2 = HIDDEN * INTERMEDIATE + HIDDEN * (INTERMEDIATE // 32)
    return max(nv_w13 + nv_w2, mx_w13 + mx_w2)


def partial_bytes() -> int:
    """Eight slots per projection. NVFP4 images are decoded in place."""
    return 2 * TOP_K * (_MAX_DATA + _MAX_SCALE)


def plan_opening(remaining: int, expert_bytes: int, activation) -> tuple[int, bool]:
    """Pick a context and whether every expert image stays resident.

    ``remaining`` is the budget after the dense weights. A full atlas is
    preferred at the largest context. The context shrinks only when eight
    staged images do not fit beside its cache and activations.
    """
    if remaining < 1 or expert_bytes < 1:
        message = "Inkling does not fit in the memory budget"
        raise MemoryError(message)
    eight = partial_bytes()
    full_act = int(activation(CONTEXTS[0]))
    if expert_bytes + full_act <= remaining:
        return CONTEXTS[0], True
    if eight + full_act <= remaining:
        return CONTEXTS[0], False
    for ctx in CONTEXTS[1:]:
        need = int(activation(ctx))
        if expert_bytes + need <= remaining:
            return ctx, True
        if eight + need <= remaining:
            return ctx, False
    message = "Inkling does not fit at the minimum context"
    raise MemoryError(message)


def _converted(experts: Any) -> None:
    """Set the stored layout of one tensor after the fp4 requant."""
    dst_k = stored_k(experts.cols)
    experts.inner = dst_k
    experts.data_stride = experts.rows * (dst_k // 2)
    experts.scale_stride = experts.rows * (dst_k // 32)
    experts.fp4 = True


class ExpertAtlas:
    """Device copy of the expert bytes the forward is allowed to read."""

    def __init__(self, session: Any, weights: Any, *, owner: str, complete: bool) -> None:
        self.session = session
        self.weights = weights
        self.owner = owner
        self.complete = bool(complete)
        self._spans: list[Any] = []
        self._staged: dict[int, tuple[int, ...]] = {}
        self.slot_data_off = [session.zeros(EXPERTS * 4) for _ in range(2)]
        self.slot_scale_off = [session.zeros(EXPERTS * 4) for _ in range(2)]
        self.slot_widths = session.zeros(EXPERTS)
        if complete:
            self._load_all()
            return
        stage_data = [self._span(TOP_K * _MAX_DATA) for _ in range(2)]
        stage_scale = [self._span(TOP_K * _MAX_SCALE) for _ in range(2)]
        for layer in weights.layers:
            for which, experts in enumerate((layer.w13, layer.w2)):
                experts.data = stage_data[which]
                experts.scale = stage_scale[which]
                if experts.nvfp4 and experts.global_scale is not None:
                    experts.global_handle = session.zeros(EXPERTS * 4)
                elif experts.nvfp4:
                    _converted(experts)
        self._bind_offsets()

    def _device(self, nbytes: int) -> Any:
        handle = device_buffer(self.owner, nbytes)
        return self.session._keep(handle)

    def _span(self, nbytes: int) -> Any:
        handle = device_buffer(self.owner, nbytes)
        self._spans.append(handle)
        return handle

    def close(self) -> None:
        for handle in self._spans:
            memory.release(handle)
        self._spans = []

    @property
    def bytes(self) -> int:
        if self.complete:
            return int(self.weights.expert_bytes)
        return partial_bytes()

    def _load_all(self) -> None:
        if getattr(self.weights, "_mixed", False):
            self._load_mixed()
            return
        weights = self.weights
        record = weights.experts
        for layer in weights.layers:
            for experts in (layer.w13, layer.w2):
                data_off, data_n = experts.file_data
                scale_off, scale_n = experts.file_scale
                experts.data = self._device(data_n)
                experts.scale = self._device(scale_n)
                record.pread(experts.data, data_off, data_n, note=weights._read)
                record.pread(experts.scale, scale_off, scale_n, note=weights._read)
                if experts.nvfp4 and experts.global_scale is not None:
                    experts.global_handle = self.session.upload(
                        np.ascontiguousarray(experts.global_scale, dtype=np.float32)
                    )
                elif experts.nvfp4:
                    _converted(experts)
        self._bind_offsets()

    def _load_mixed(self) -> None:
        record = self.weights.experts
        for layer in self.weights.layers:
            handle = self.session.upload(np.ascontiguousarray(layer.widths, dtype=np.uint8))
            layer.w13.widths = handle
            layer.w2.widths = handle
            for experts in (layer.w13, layer.w2):
                data_off, data_n = experts.file_data
                experts.data = self._device(data_n)
                experts.scale = experts.data
                record.pread(experts.data, data_off, data_n, note=self.weights._read)
                experts.data_offsets = self.session.upload(
                    np.ascontiguousarray(experts.host_data_off, dtype=np.uint32)
                )
                experts.scale_offsets = self.session.upload(
                    np.ascontiguousarray(experts.host_scale_off, dtype=np.uint32)
                )

    def _bind_offsets(self) -> None:
        """Identity tables for a uniform stride. A mixed layer already has its own."""
        for layer in self.weights.layers:
            if getattr(layer, "mixed", False) and getattr(layer, "widths", None) is not None:
                if getattr(layer.w13, "widths", None) is None:
                    handle = self.session.upload(np.ascontiguousarray(layer.widths, dtype=np.uint8))
                    layer.w13.widths = handle
                    layer.w2.widths = handle
            for experts in (layer.w13, layer.w2):
                if experts.data_offsets is not None:
                    continue
                if getattr(layer, "mixed", False) and int(layer.uniform) == 0:
                    continue
                count = EXPERTS
                data = np.arange(count, dtype=np.uint32) * np.uint32(experts.data_stride)
                scale = np.arange(count, dtype=np.uint32) * np.uint32(experts.scale_stride)
                experts.data_offsets = self.session.upload(data)
                experts.scale_offsets = self.session.upload(scale)

    def _stage_mixed(self, layer: Any, chosen: list[int]) -> None:
        """Copy interleaved fp4 and fp8 images into per-slot strides."""
        key = tuple(chosen)
        if self._staged.get(layer.index) == key and self._staged.get(-1) == (layer.index,):
            return
        record = self.weights.experts
        for which, experts in enumerate((layer.w13, layer.w2)):
            data_base = int(experts.file_data[0])
            slot_data = int(np.max(experts.host_data_bytes))
            slot_scale = int(np.max(experts.host_scale_bytes))
            data_off = np.zeros(EXPERTS, dtype=np.uint32)
            scale_off = np.zeros(EXPERTS, dtype=np.uint32)
            widths = np.zeros(EXPERTS, dtype=np.uint8)
            for slot, expert in enumerate(chosen):
                record.pread(
                    experts.data,
                    data_base + int(experts.host_data_off[expert]),
                    int(experts.host_data_bytes[expert]),
                    dest_off=slot * slot_data,
                )
                record.pread(
                    experts.scale,
                    data_base + int(experts.host_scale_off[expert]),
                    int(experts.host_scale_bytes[expert]),
                    dest_off=slot * slot_scale,
                )
                data_off[slot] = slot * slot_data
                scale_off[slot] = slot * slot_scale
                widths[slot] = int(layer.widths[expert])
            memory.store(self.slot_data_off[which], 0, data_off.tobytes())
            memory.store(self.slot_scale_off[which], 0, scale_off.tobytes())
            if which == 0:
                memory.store(self.slot_widths, 0, widths.tobytes())
        self._staged = {layer.index: key, -1: (layer.index,)}

    def stage(self, layer: Any, chosen: list[int]) -> None:
        """Copy up to eight routed experts into slots 0..n-1 of both projections.

        The device must be idle. A baked pack copies the stored expert. Otherwise
        NVFP4 slots are requantized ahead of the kernels that read them.
        """
        if len(chosen) > TOP_K:
            message = f"{len(chosen)} experts do not fit {TOP_K} slots"
            raise RuntimeError(message)
        if getattr(layer, "mixed", False):
            self._stage_mixed(layer, chosen)
            return
        key = tuple(chosen)
        if self._staged.get(layer.index) == key and self._staged.get(-1) == (layer.index,):
            return
        weights = self.weights
        record = weights.experts
        for experts in (layer.w13, layer.w2):
            data_off, _data_n = experts.file_data
            scale_off, _scale_n = experts.file_scale
            if experts.nvfp4 and not experts.fp4 and experts.global_scale is not None:
                picked = np.asarray(chosen, dtype=np.int64)
                staged = np.zeros(EXPERTS, dtype=np.float32)
                staged[: len(chosen)] = experts.global_scale[picked]
                memory.store(experts.global_handle, 0, staged.tobytes())
            for slot, expert in enumerate(chosen):
                record.pread(
                    experts.data,
                    data_off + expert * experts.data_stride,
                    experts.data_stride,
                    dest_off=slot * experts.data_stride,
                )
                record.pread(
                    experts.scale,
                    scale_off + expert * experts.scale_stride,
                    experts.scale_stride,
                    dest_off=slot * experts.scale_stride,
                )
        self._staged = {layer.index: key, -1: (layer.index,)}
