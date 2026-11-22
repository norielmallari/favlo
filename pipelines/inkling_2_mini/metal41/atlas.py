"""Resident expert images.

Expert bytes stay in the safetensors file. NVFP4 tensors are requantized to
Metal fp4 (E8M0 per 32) on the device, one tensor per dispatch. MXFP8
tensors are copied as stored. When every image does not fit beside the
dense weights, the half key/value cache, and the activation buffers, the
atlas keeps eight slots per projection and stages the experts a token
routes before they run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from favlo.memory import device_buffer, memory
from inkling_2_mini.metal41.layout import EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, stored_k
from inkling_2_mini.metal41.session import u32

# Largest stored tensor of one expert: MXFP8 fused gate/up, then the widest block scale.
_MAX_DATA = HIDDEN * HIDDEN
_MAX_SCALE = HIDDEN * (HIDDEN // 16)
CONTEXTS = (16384, 8192, 4096, 2048, 1024, 512)


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
    """Eight slots per projection plus the raw staging for eight NVFP4 images."""
    return 2 * TOP_K * (_MAX_DATA + _MAX_SCALE) + TOP_K * image_bytes()


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
        if complete:
            self._load_all()
            return
        self._raw_data = [self._span(TOP_K * _MAX_DATA) for _ in range(2)]
        self._raw_scale = [self._span(TOP_K * _MAX_SCALE) for _ in range(2)]
        self._global = [self._span(TOP_K * 4) for _ in range(2)]
        stage_data = [self._span(TOP_K * _MAX_DATA) for _ in range(2)]
        stage_scale = [self._span(TOP_K * _MAX_SCALE) for _ in range(2)]
        for layer in weights.layers:
            for which, experts in enumerate((layer.w13, layer.w2)):
                experts.data = stage_data[which]
                experts.scale = stage_scale[which]
                if experts.nvfp4:
                    _converted(experts)

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
        weights = self.weights
        largest_data = 0
        largest_scale = 0
        for layer in weights.layers:
            for experts in (layer.w13, layer.w2):
                if experts.nvfp4:
                    largest_data = max(largest_data, experts.file_data[1])
                    largest_scale = max(largest_scale, experts.file_scale[1])
        raw_data = device_buffer(self.owner, largest_data) if largest_data else None
        raw_scale = device_buffer(self.owner, largest_scale) if largest_scale else None
        try:
            for layer in weights.layers:
                for experts in (layer.w13, layer.w2):
                    data_off, data_n = experts.file_data
                    scale_off, scale_n = experts.file_scale
                    if not experts.nvfp4:
                        experts.data = self.session.upload(weights.blob(data_off, data_n))
                        experts.scale = self.session.upload(weights.blob(scale_off, scale_n))
                        continue
                    memory.store(raw_data, 0, weights.blob(data_off, data_n))
                    memory.store(raw_scale, 0, weights.blob(scale_off, scale_n))
                    self._convert(experts, raw_data, raw_scale, experts.global_scale, EXPERTS, None)
        finally:
            for handle in (raw_data, raw_scale):
                if handle is not None:
                    memory.release(handle)

    def _convert(
        self,
        experts: Any,
        raw_data: Any,
        raw_scale: Any,
        global_scale: np.ndarray,
        count: int,
        global_handle: Any,
    ) -> None:
        """Requant ``count`` NVFP4 images in ``raw_*`` to Metal fp4 slots 0..count-1."""
        src_k = experts.cols
        if global_handle is None:
            data = self.session.zeros(EXPERTS * experts.rows * (stored_k(src_k) // 2))
            scale = self.session.zeros(EXPERTS * experts.rows * (stored_k(src_k) // 32))
            gs = device_buffer(self.owner, EXPERTS * 4)
            memory.store(gs, 0, np.ascontiguousarray(global_scale, dtype=np.float32).tobytes())
            experts.data = data
            experts.scale = scale
            _converted(experts)
        else:
            gs = global_handle
            memory.store(gs, 0, np.ascontiguousarray(global_scale, dtype=np.float32).tobytes())
        blocks = count * experts.rows * (experts.inner // 32)
        self.session.launch(
            "pack_nvfp4_fp4",
            [raw_data, raw_scale, gs, experts.data, experts.scale],
            [u32(count), u32(experts.rows), u32(src_k), u32(experts.inner)],
            blocks,
        )
        if global_handle is None:
            memory.release(gs)

    def stage(self, layer: Any, chosen: list[int]) -> None:
        """Copy up to eight routed experts into slots 0..n-1 of both projections.

        The device must be idle. NVFP4 slots are requantized by a dispatch in
        the open command buffer, ahead of the kernels that read them.
        """
        if len(chosen) > TOP_K:
            message = f"{len(chosen)} experts do not fit {TOP_K} slots"
            raise RuntimeError(message)
        key = tuple(chosen)
        if self._staged.get(layer.index) == key and self._staged.get(-1) == (layer.index,):
            return
        weights = self.weights
        for which, experts in enumerate((layer.w13, layer.w2)):
            data_off, data_n = experts.file_data
            scale_off, scale_n = experts.file_scale
            src_data = data_n // EXPERTS
            src_scale = scale_n // EXPERTS
            if experts.nvfp4:
                for slot, expert in enumerate(chosen):
                    memory.store(
                        self._raw_data[which],
                        slot * src_data,
                        weights.blob(data_off + expert * src_data, src_data),
                    )
                    memory.store(
                        self._raw_scale[which],
                        slot * src_scale,
                        weights.blob(scale_off + expert * src_scale, src_scale),
                    )
                scales = experts.global_scale[np.asarray(chosen, dtype=np.int64)]
                self._convert(
                    experts,
                    self._raw_data[which],
                    self._raw_scale[which],
                    scales,
                    len(chosen),
                    self._global[which],
                )
                continue
            for slot, expert in enumerate(chosen):
                memory.store(
                    experts.data,
                    slot * src_data,
                    weights.blob(data_off + expert * src_data, src_data),
                )
                memory.store(
                    experts.scale,
                    slot * src_scale,
                    weights.blob(scale_off + expert * src_scale, src_scale),
                )
        self._staged = {layer.index: key, -1: (layer.index,)}
