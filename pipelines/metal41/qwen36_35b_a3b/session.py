"""Device buffers and the shader library for this pipeline."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import memory

KERNELS = (
    "prefill_fp4_e8m0",
    "prefill_fp8_e8m0",
    "prefill_fp8_n64",
    "prefill_fp8_n64_stride",
    "prefill_fp8_n64_sweep",
    "prefill_fp8_m8",
    "gemv_fp4",
    "gemv_f16",
    "rmsnorm_f32",
    "layernorm_f32",
    "silu_f32",
    "sigmoid_f32",
    "softplus_f32",
    "gelu_tanh_f32",
    "l2_heads_f32",
    "residual_add_f32",
    "swiglu_f32",
    "rope_partial",
    "gdn_conv",
    "gdn_state",
    "attend_gqa",
    "attend_head72",
    "topk8",
    "gather_row",
    "dequant_e4m3",
    "cast_f32_f16",
    "shift_history",
    "store_row",
    "split_q_gate",
    "axpy_f32",
    "zero_f32",
    "mul_f32",
    "softmax256",
    "gather_fp4_row",
    "gather_fp8_row",
    "gemv_fp4_batch",
    "gemv_fp4_x4",
    "gemv_fp4_vec",
    "gemv_fp8_vec",
    "gemv_fp4_vec_slots",
    "gemv_fp4_vec_acc",
    "gemv_fp8_vec_acc",
    "gemv_f16_f32",
    "gdn_in_proj",
    "gdn_in_proj_fp8",
    "attn_qkv",
    "attn_qkv_fp8",
    "router_coeff",
    "gdn_conv_shift",
    "qk_norm_rope",
    "store_kv",
    "attend_gqa_heads",
    "kv_plane_k",
    "kv_tail_store",
    "kv_plane_v",
    "kv_cast_q",
    "kv_row_init",
    "kv_softmax_tile",
    "kv_softmax_span",
    "kv_softmax_max",
    "kv_softmax_prob",
    "kv_softmax_close",
    "kv_softmax_max_rows",
    "kv_softmax_prob_rows",
    "kv_cast_probs",
    "kv_scale_partial",
    "kv_pv_split",
    "kv_pv_split8",
    "kv_cast_blocks",
    "kv_restore_sum",
    "kv_part_sum",
    "kv_acc_add",
    "kv_attend_write",
    "moe_gate_silu",
    "moe_down_mix",
    "moe_gate_silu_w",
    "moe_down_mix_w",
    "moe_gate_silu_fp8",
    "moe_down_mix_fp8",
    "rmsnorm_rows",
    "embed_fp4_rows",
    "embed_fp8_rows",
    "residual_rows",
    "gdn_conv_rows",
    "gdn_conv_keep",
    "gdn_state_rows",
    "gdn_l2_rows",
    "gdn_chunk_solve",
    "gdn_chunk_out",
    "gdn_chunk_state",
    "gdn_y_norm",
    "split_q_gate_rows",
    "qk_norm_rope_rows",
    "store_kv_rows",
    "attend_prefill",
    "attend_prefill2",
    "gemv_f16_rows",
    "gemv_f16_grid",
    "coeff_rows",
    "router8_rows",
    "silu_mul_rows",
    "gather_half_rows",
    "gather_half_slots",
    "gather_half_grouped",
    "moe_group",
    "silu_mul_grouped",
    "cast_f32_f16_n",
    "scatter_add_grouped",
    "silu_wave",
    "prefill_fp4_expert_tile",
    "prefill_fp4_expert_tile_at",
    "prefill_fp4_expert_tile_w",
    "prefill_fp8_expert_tile_at",
    "prefill_f16",
    "prefill_fp4_n64",
    "prefill_fp4_n64_stride",
    "restore_fp4_i8",
    "restore_fp8_i8",
    "quantize_half_i8",
    "prefill_i8_n64",
    "scale_i8_f32",
    "scatter_add_rows",
    "silu_mul_slots",
    "router8",
    "copy_u32",
    "remap_expert_ids",
    "mix_experts",
    "layernorm_rows_f32",
    "cast_pad_f16",
    "pack_bias_f32",
    "gemv_fp4_tail",
    "gemv_fp8_tail",
    "gelu_tanh_n",
    "gelu_erf_n",
    "split_qkv72",
    "vision_rope72",
    "vision_attend72",
    "dequant_fp4_rows",
    "dequant_fp8_rows",
)

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_ZERO = b"\x00" * (8 << 20)


def _clear(buffer: Any, nbytes: int, note: Any = None) -> None:
    view = memoryview(buffer.contents().as_buffer(nbytes))
    offset = 0
    while offset < nbytes:
        end = min(offset + len(_ZERO), nbytes)
        view[offset:end] = _ZERO[: end - offset]
        step = end - offset
        offset = end
        if note is not None:
            note(step)


def _metal() -> Any:
    import Metal

    return Metal


def _objc_pair(result: Any) -> tuple[Any, str | None]:
    if isinstance(result, tuple) and len(result) == 2:
        value, error = result
        if error is not None:
            return None, str(error)
        return value, None
    return result, None


class Session:
    """One Metal device. Compiling the library fails if a kernel has no pipeline."""

    def __init__(self, owner: str | None = None) -> None:
        metal = _metal()
        self._owner = owner or f"qwen36-35b-a3b:{id(self)}"
        device = memory.device
        options = metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        source = _SHADER.read_text(encoding="utf-8")
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(source, options, None)
        )
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        self.queue = memory.make_queue()
        self.shared = metal.MTLResourceStorageModeShared
        self.pipes: dict[str, Any] = {}
        self._recording = False
        self._redirect: Any = None
        self._inflight: list[tuple[Any, list[Any]]] = []
        self._sizes: dict[tuple[int, int, int], Any] = {}
        for name in KERNELS:
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"{name} is not in the shader library"
                raise RuntimeError(message)
            pipeline, pipe_error = _objc_pair(
                device.newComputePipelineStateWithFunction_error_(function, None)
            )
            if pipeline is None:
                message = f"{name} pipeline failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipeline
        self._held: list[Any] = []
        self._owned: list[Any] = []
        self._load_note = None

    @property
    def device_name(self) -> str:
        name = self.device.name
        return str(name() if callable(name) else name)

    def upload(self, blob: bytes | np.ndarray) -> Any:
        if isinstance(blob, np.ndarray):
            raw = np.ascontiguousarray(blob).tobytes()
        else:
            raw = bytes(blob)
        if raw:
            buffer = memory.upload(raw, owner=self._owner)
        else:
            buffer = memory.allocate(16, owner=self._owner, zero=True)
        return self._keep(buffer)

    def allocate(self, nbytes: int) -> Any | None:
        """A shared buffer, or None when the device refuses the length."""
        return self._keep(memory.allocate(max(int(nbytes), 16), owner=self._owner))

    def close(self) -> None:
        """Wait for the queue and drop every buffer this session allocated."""
        try:
            self.end()
        except Exception:
            abort = getattr(self, "abort", None)
            if callable(abort):
                abort()
        self._inflight.clear()
        self._held.clear()
        self._owned.clear()
        self.pipes.clear()
        self._load_note = None
        memory.sweep(self._owner)
        self.queue = None
        self.device = None
        self.metal = None

    def _keep(self, buffer: Any) -> Any:
        if buffer is not None:
            self._owned.append(buffer)
        return buffer

    def zeros(self, nbytes: int) -> Any:
        buffer = self.allocate(nbytes)
        if buffer is None:
            message = f"device refused a {nbytes} byte buffer"
            raise RuntimeError(message)
        _clear(buffer, nbytes, self._load_note)
        return buffer

    def download(self, buffer: Any, nbytes: int) -> bytes:
        return bytes(memoryview(buffer.contents().as_buffer(nbytes)))

    def download_f32(self, buffer: Any, count: int) -> np.ndarray:
        return np.frombuffer(self.download(buffer, count * 4), dtype=np.float32).copy()

    def dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        self._held.extend(buffers)
        self._held.extend(constants)
        command = memory.command_buffer(self.queue)
        encoder = command.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipes[name])
        for index, buffer in enumerate(buffers):
            memory.bind(encoder, index, buffer, 0)
        base = len(buffers)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._size(groups),
            self._size(threads),
        )
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        status = int(command.status())
        error = command.error()
        memory.retire(command)
        self._held.clear()
        if status != 4:
            message = f"{name} command status {status} {error}"
            raise RuntimeError(message)

    def begin(self) -> None:
        self._recording = True
        self._pass_held: list[Any] = []
        self._pass_command = memory.command_buffer(self.queue)
        self._pass_encoder = self._pass_command.computeCommandEncoder()

    def encode(
        self,
        name: str,
        bindings: list[tuple[Any, int]],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
    ) -> None:
        redirect = self._redirect
        if redirect is None:
            encoder = self._pass_encoder
            held = self._pass_held
        else:
            encoder = redirect.encoder
            held = redirect.held
        encoder.setComputePipelineState_(self.pipes[name])
        for index, (buffer, offset) in enumerate(bindings):
            held.append(buffer)
            memory.bind(encoder, index, buffer, int(offset))
        base = len(bindings)
        for index, blob in enumerate(constants):
            held.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self._size(groups),
            self._size(threads),
        )

    def _size(self, value: tuple[int, int, int]) -> Any:
        cached = self._sizes.get(value)
        if cached is None:
            cached = self.metal.MTLSize(*value)
            self._sizes[value] = cached
        return cached

    def split(self) -> None:
        """End this encoder and open another on the same command buffer.

        A kernel can fill an indirect-dispatch buffer, and the next encoder
        reads that buffer when the command buffer runs.
        """
        memory.handoff(self._pass_encoder, self._pass_command)
        self._pass_encoder.endEncoding()
        self._pass_encoder = self._pass_command.computeCommandEncoder()

    def encode_indirect(
        self,
        name: str,
        bindings: list[tuple[Any, int]],
        constants: list[bytes],
        indirect: Any,
        indirect_offset: int,
        threads: tuple[int, int, int],
    ) -> None:
        """Dispatch ``name``. The grid is three uints at ``indirect_offset``."""
        encoder = self._pass_encoder
        encoder.setComputePipelineState_(self.pipes[name])
        for index, (buffer, offset) in enumerate(bindings):
            self._pass_held.append(buffer)
            memory.bind(encoder, index, buffer, int(offset))
        base = len(bindings)
        for index, blob in enumerate(constants):
            self._pass_held.append(blob)
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        self._pass_held.append(indirect)
        memory.bind_indirect(encoder, indirect, int(indirect_offset), self._size(threads))

    def commit(self) -> None:
        """Commit the open encoder and leave it in flight."""
        if not self._recording:
            message = "commit without an open command buffer"
            raise RuntimeError(message)
        self._recording = False
        memory.handoff(self._pass_encoder, self._pass_command)
        self._pass_encoder.endEncoding()
        command = self._pass_command
        command.commit()
        self._inflight.append((command, self._pass_held))
        self._pass_held = []
        self._pass_encoder = None
        self._pass_command = None

    def abort(self) -> None:
        """Drop the open encoder. A buffer already in flight stays in flight."""
        if not self._recording:
            return
        self._recording = False
        self._pass_encoder.endEncoding()
        self._pass_encoder = None
        self._pass_command = None
        self._pass_held = []

    def inflight(self) -> int:
        return len(self._inflight)

    def wait_oldest(self) -> None:
        if not self._inflight:
            return
        command, held = self._inflight.pop(0)
        self._finish(command, held)

    def wait_inflight(self) -> None:
        while self._inflight:
            self.wait_oldest()

    def end(self) -> None:
        if self._recording:
            self.commit()
        self.wait_inflight()

    def _finish(self, command: Any, held: list[Any]) -> None:
        command.waitUntilCompleted()
        status = int(command.status())
        error = command.error()
        memory.retire(command)
        del held
        if status != 4:
            message = f"command status {status} {error}"
            raise RuntimeError(message)

    def prefill(
        self,
        name: str,
        activation: np.ndarray,
        data: bytes,
        scales: bytes,
        rows: int,
        cols: int,
        inner: int,
    ) -> np.ndarray:
        if rows % 64 or cols % 32 or inner % 32:
            message = "prefill tile requires rows multiple of 64 and cols, inner multiple of 32"
            raise ValueError(message)
        destination = self.zeros(rows * cols * 4)
        width = int(self.pipes[name].threadExecutionWidth())
        self.dispatch(
            name,
            [self.upload(activation), self.upload(data), self.upload(scales), destination],
            [struct.pack("<I", rows), struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols // 32, rows // 64, 1),
            (width * 4, 1, 1),
        )
        return self.download_f32(destination, rows * cols).reshape(rows, cols)

    def gemv_fp4(
        self, activation: np.ndarray, data: bytes, scales: bytes, cols: int, inner: int
    ) -> np.ndarray:
        destination = self.zeros(cols * 4)
        self.dispatch(
            "gemv_fp4",
            [self.upload(activation), self.upload(data), self.upload(scales), destination],
            [struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols, 1, 1),
            (32, 1, 1),
        )
        return self.download_f32(destination, cols)

    def gemv_f16(self, activation: np.ndarray, weights: np.ndarray) -> np.ndarray:
        cols, inner = weights.shape
        destination = self.zeros(cols * 4)
        self.dispatch(
            "gemv_f16",
            [
                self.upload(activation),
                self.upload(np.ascontiguousarray(weights, dtype=np.float16)),
                destination,
            ],
            [struct.pack("<I", cols), struct.pack("<I", inner)],
            (cols, 1, 1),
            (32, 1, 1),
        )
        return self.download_f32(destination, cols)
