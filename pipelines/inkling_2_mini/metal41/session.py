"""One Metal device and the Inkling shader library."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from favlo.memory import device_buffer, memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"

# Threads per threadgroup for each kernel.
KERNELS: dict[str, int] = {
    "matmul_f16_m64": 128,
    "matmul_fp4_m32": 128,
    "expert_fp4_tile": 128,
    "expert_fp8_tile": 128,
    "gemv_fp4": 32,
    "gemv_fp4_logits": 32,
    "gemv_fp4_conv_add": 32,
    "gemv_f16": 32,
    "gemv_attn": 32,
    "route_gemv": 32,
    "moe_gate_fp4": 32,
    "moe_gate_fp8": 32,
    "moe_down_fp4": 32,
    "moe_down_fp8": 32,
    "route_rows": 128,
    "rmsnorm": 256,
    "mix_blocks": 256,
    "ctl_down": 256,
    "ctl_blend": 256,
    "ctl_pair_rows": 256,
    "ctl_blend_rows": 256,
    "attn_prep": 128,
    "kv_window": 256,
    "conv_tail": 256,
    "conv_add_rows": 256,
    "attend_group": 256,
    "attend_flash": 128,
    "moe_group": 128,
    "gather_grouped": 256,
    "swiglu_grouped": 256,
    "scatter_grouped": 256,
    "zero_f32": 256,
    "copy_words": 256,
    "cast_f16": 256,
    "embed_gather": 256,
    "argmax_span": 256,
    "argmax_index": 256,
    "pack_nvfp4_fp4": 256,
    "dequant_nvfp4": 256,
    "vision_dequant": 256,
    "vision_pad": 256,
    "vision_bias_cast": 256,
    "vision_add": 256,
    "vision_ln": 256,
    "vision_gelu": 256,
    "vision_residual": 256,
    "vision_rope": 256,
    "vision_attend": 32,
    "vision_merge": 256,
    "vision_scatter": 256,
}


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


def u32(value: int) -> bytes:
    return struct.pack("<I", int(value))


def f32(value: float) -> bytes:
    return struct.pack("<f", float(value))


class Session:
    """Compile the library and dispatch Metal kernels.

    Between ``begin`` and ``end`` kernels go into one encoder. ``flush``
    commits the open command buffer without waiting and opens the next, so
    the host can encode while the device runs. ``end`` waits for every
    committed buffer.
    """

    def __init__(self, owner: str) -> None:
        metal = _metal()
        self._owner = owner
        device = memory.device
        options = metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _objc_pair(
            device.newLibraryWithSource_options_error_(
                _SHADER.read_text(encoding="utf-8"), options, None
            )
        )
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self.metal = metal
        self.device = device
        self.queue = memory.make_queue()
        self.pipes: dict[str, Any] = {}
        self._threads: dict[str, Any] = {}
        self._sizes: dict[tuple[int, int, int], Any] = {}
        self._owned: list[Any] = []
        for name, threads in KERNELS.items():
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
            self._threads[name] = self._size((threads, 1, 1))
        self._recording = False
        self._pass_held: list[Any] = []
        self._pass_command: Any = None
        self._pass_encoder: Any = None
        self._inflight: list[tuple[Any, list[Any]]] = []

    def close(self) -> None:
        if self._recording:
            self.end()
        self._owned.clear()
        self.pipes.clear()
        memory.sweep(self._owner)
        self.queue = None
        self.device = None
        self.metal = None

    def _keep(self, handle: Any) -> Any:
        if handle is not None:
            self._owned.append(handle)
        return handle

    def upload(self, blob: bytes | np.ndarray) -> Any:
        if isinstance(blob, np.ndarray):
            raw = np.ascontiguousarray(blob).tobytes()
        else:
            raw = bytes(blob)
        if not raw:
            return self._keep(device_buffer(self._owner, 16, zero=True))
        return self._keep(memory.upload(raw, owner=self._owner))

    def zeros(self, nbytes: int) -> Any:
        return self._keep(device_buffer(self._owner, max(int(nbytes), 16), zero=True))

    def download(self, handle: Any, nbytes: int, offset: int = 0) -> bytes:
        return memory.load(handle, int(offset), int(nbytes))

    def download_f32(self, handle: Any, count: int, offset: int = 0) -> np.ndarray:
        return np.frombuffer(self.download(handle, count * 4, offset), dtype=np.float32).copy()

    def _size(self, value: tuple[int, int, int]) -> Any:
        cached = self._sizes.get(value)
        if cached is None:
            cached = self.metal.MTLSize(*value)
            self._sizes[value] = cached
        return cached

    @property
    def recording(self) -> bool:
        return self._recording

    def begin(self) -> None:
        """Open one command buffer for the kernels that follow."""
        if self._recording:
            message = "a command buffer is already open"
            raise RuntimeError(message)
        self._recording = True
        self._open()

    def _open(self) -> None:
        self._pass_held = []
        self._pass_command = memory.command_buffer(self.queue)
        self._pass_encoder = self._pass_command.computeCommandEncoder()

    def _commit(self) -> None:
        memory.handoff(self._pass_encoder, self._pass_command)
        self._pass_encoder.endEncoding()
        command = self._pass_command
        command.commit()
        self._inflight.append((command, self._pass_held))
        self._pass_encoder = None
        self._pass_command = None
        self._pass_held = []

    def flush(self) -> None:
        """Commit the open buffer without waiting and open the next one."""
        if not self._recording:
            return
        self._commit()
        self._open()

    def commit(self) -> None:
        """Commit the open buffer without waiting. Nothing stays open."""
        if not self._recording:
            return
        self._recording = False
        self._commit()

    def pending(self) -> bool:
        return bool(self._inflight)

    def wait_oldest(self) -> None:
        """Wait for the oldest committed buffer."""
        if not self._inflight:
            return
        command, held = self._inflight.pop(0)
        status, error = memory.retire(command)
        del held
        if status != 4:
            message = f"command status {status} {error}"
            raise RuntimeError(message)

    def drain(self) -> None:
        """Commit anything open and wait for every committed buffer."""
        self.commit()
        failure = None
        while self._inflight:
            try:
                self.wait_oldest()
            except RuntimeError as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    def end(self) -> None:
        """Commit the open buffer and wait for every committed buffer."""
        if not self._recording and not self._inflight:
            return
        self.drain()

    def dispatch(
        self,
        name: str,
        buffers: list[Any],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int] | None = None,
    ) -> None:
        """Encode one kernel. A buffer entry is a handle or (handle, byte offset)."""
        single = not self._recording
        if single:
            self.begin()
        encoder = self._pass_encoder
        held = self._pass_held
        encoder.setComputePipelineState_(self.pipes[name])
        bind = memory.bind
        for index, entry in enumerate(buffers):
            if type(entry) is tuple:
                handle, offset = entry
            else:
                handle, offset = entry, 0
            held.append(handle)
            bind(encoder, index, handle, offset)
        base = len(buffers)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        grid = self._threads[name] if threads is None else self._size(threads)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(self._size(groups), grid)
        if single:
            self.end()

    def launch(self, name: str, buffers: list[Any], constants: list[bytes], count: int) -> None:
        """One thread per element, in groups of 256."""
        groups = (max((int(count) + 255) // 256, 1), 1, 1)
        self.dispatch(name, buffers, constants, groups, (256, 1, 1))
