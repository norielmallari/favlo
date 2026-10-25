"""Metal session. GEMMs are ``matmul2d``. Elementwise work is a kernel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from favlo.memory import bind, memory
from wan22_ti2v_5b.spec import FP16_MAX, TILE_M, TILE_N, TILE_N_I8

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_THREADS = 128
_HEAD_BUDGET = 512 * 1024 * 1024


class Gpu:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.queue = memory.make_queue()
        self._lib = _library()
        self._pipes: dict[str, object] = {}
        self._resident_handles: dict[object, object] = {}
        self._resident_alive: list[np.ndarray] = []
        self._scratch: dict[str, tuple[int, object]] = {}

    def close(self) -> None:
        memory.sweep(self.owner)
        self._resident_handles.clear()
        self._resident_alive.clear()
        self._scratch.clear()

    def upload(self, array: np.ndarray):
        raw = np.ascontiguousarray(array)
        return memory.upload(raw.tobytes(), owner=self.owner)

    def zeros(self, nbytes: int):
        return memory.allocate(nbytes, owner=self.owner, zero=True)

    def download(self, handle, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
        count = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = memory.load(handle, 0, count)
        return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()

    def release(self, handle) -> None:
        memory.release(handle)

    def _resident(self, array: np.ndarray, view: np.ndarray | None = None, factory=None):
        """Upload ``array`` once. ``factory`` builds the bytes on a miss."""
        key = id(array)
        found = self._resident_handles.get(key)
        if found is None:
            if factory is not None:
                payload = factory()
            elif view is not None:
                payload = view
            else:
                payload = array
            found = self.upload(payload)
            self._resident_handles[key] = found
            self._resident_alive.append(array)
        return found

    def _keep(self, key: object, payload: np.ndarray):
        """Upload ``payload`` once under an explicit key."""
        found = self._resident_handles.get(key)
        if found is None:
            found = self.upload(payload)
            self._resident_handles[key] = found
            self._resident_alive.append(payload)
        return found

    def _resident_f16(self, packed: np.ndarray, padded_n: int, cols: int, out_n: int):
        key = (id(packed), padded_n, cols, out_n)
        found = self._resident_handles.get(key)
        if found is None:
            weight = np.zeros((padded_n, cols), dtype=np.float16)
            weight[:out_n] = np.ascontiguousarray(packed, dtype=np.float16)
            found = self.upload(weight)
            self._resident_handles[key] = found
            self._resident_alive.append(packed)
        return found

    def _resident_conv(
        self,
        source: np.ndarray,
        shaped: np.ndarray,
        padded_n: int,
        k_pad: int,
        out_c: int,
        taps: int,
    ):
        key = (id(source), padded_n, k_pad, out_c, taps)
        found = self._resident_handles.get(key)
        if found is None:
            flat = np.ascontiguousarray(shaped, dtype=np.float16).reshape(out_c, taps)
            pad = np.zeros((padded_n, k_pad), dtype=np.float16)
            pad[:out_c, :taps] = flat
            found = self.upload(pad)
            self._resident_handles[key] = found
            self._resident_alive.append(source)
        return found

    def linear(
        self,
        left: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray | None,
        *,
        kind: str,
        bias: np.ndarray | None = None,
        stride: int | None = None,
    ) -> np.ndarray:
        """``left @ weight.T``. ``left`` is ``[M, K]``. int8 stores float32."""
        if kind == "int8":
            return self._linear_i8(left, packed, scales, bias)
        rows, cols = int(left.shape[0]), int(left.shape[1])
        out_n = int(packed.shape[0]) if kind != "f16" else int(packed.shape[0])
        if kind == "f16":
            out_n = int(packed.shape[0])
            k_weight = int(packed.shape[1])
        else:
            k_weight = int(cols)
        if k_weight != cols:
            message = f"K {cols} does not match the weight"
            raise RuntimeError(message)
        padded_m = _pad(rows, TILE_M)
        padded_n = _pad(out_n, TILE_N)
        host = np.zeros((padded_m, cols), dtype=np.float16)
        host[:rows] = np.ascontiguousarray(left, dtype=np.float16)
        left_h = self.upload(host)
        if kind == "f16":
            weight_h = self._resident_f16(packed, padded_n, cols, out_n)
        else:
            weight = np.ascontiguousarray(packed)
            if int(weight.shape[0]) != padded_n:
                message = f"{kind} N {weight.shape[0]} is not padded to {padded_n}"
                raise RuntimeError(message)
            weight_h = self.upload(weight)
        scale_h = self.upload(scales) if scales is not None else self.zeros(16)
        dest = self.zeros(padded_m * padded_n * 4)
        stride_a = cols
        if kind == "f16":
            dims = np.array([padded_m, padded_n, cols, stride_a, cols], dtype=np.int32)
            self._gemm("gemm_f16_tn", (left_h, weight_h, dest), dims, scales=False)
        elif kind == "fp4":
            dims = np.array([padded_m, padded_n, cols, stride_a, 0], dtype=np.int32)
            self._gemm("gemm_fp4", (left_h, weight_h, scale_h, dest), dims, scales=True)
        elif kind == "fp8":
            dims = np.array([padded_m, padded_n, cols, stride_a, 0], dtype=np.int32)
            self._gemm("gemm_fp8", (left_h, weight_h, scale_h, dest), dims, scales=True)
        else:
            message = f"unknown kind {kind}"
            raise RuntimeError(message)
        values = self.download(dest, np.float32, (padded_m, padded_n))[:rows, :out_n]
        self.release(left_h)
        if kind != "f16":
            self.release(weight_h)
        self.release(scale_h)
        self.release(dest)
        if bias is not None:
            values = values + bias.astype(np.float32).reshape(1, -1)
        return _store_half(values)

    def _linear_i8(
        self,
        left: np.ndarray,
        packed: np.ndarray,
        scales: np.ndarray | None,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        from wan22_ti2v_5b.quant import quantize_activation

        rows, cols = int(left.shape[0]), int(left.shape[1])
        out_n = int(packed.shape[0])
        stored_k = int(packed.shape[1])
        if stored_k != cols:
            message = f"K {cols} does not match the int8 weight"
            raise RuntimeError(message)
        if out_n % TILE_N_I8 != 0:
            message = f"int8 N {out_n} is not a multiple of {TILE_N_I8}"
            raise RuntimeError(message)
        if scales is None:
            message = "int8 weight is missing its row scales"
            raise RuntimeError(message)
        logical = int(bias.shape[0]) if bias is not None else out_n
        padded_m = _pad(rows, TILE_M)
        host = np.zeros((padded_m, cols), dtype=np.float32)
        host[:rows] = np.ascontiguousarray(left, dtype=np.float32)
        codes, row_scale = quantize_activation(host)
        bias_row = np.zeros(out_n, dtype=np.float16)
        if bias is not None:
            bias_row[:logical] = np.ascontiguousarray(bias, dtype=np.float16).reshape(-1)[:out_n]
        left_h = self.upload(codes)
        weight_h = self._resident(packed)
        row_h = self.upload(row_scale)
        scale_h = self._resident(scales)
        bias_h = self.upload(bias_row)
        dest = self.zeros(padded_m * out_n * 4)
        dims = np.array([padded_m, out_n, cols, cols, cols], dtype=np.int32)
        shape = np.array([padded_m, out_n, logical], dtype=np.int32)
        group = _Dispatch(self)
        group.gemm("gemm_i8", (left_h, weight_h, dest), dims, tile_n=TILE_N_I8)
        group.launch("scale_i8", (dest, row_h, scale_h, bias_h), shape.tobytes(), padded_m * out_n)
        group.finish("linear_i8")
        values = self.download(dest, np.float32, (padded_m, out_n))[:rows, :logical]
        for handle in (left_h, row_h, bias_h, dest):
            self.release(handle)
        return values

    def gemm_half(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        transpose_right: bool,
        stride_a: int | None = None,
        stride_b: int | None = None,
    ) -> np.ndarray:
        """Float16 times float16. The destination is the real ``M x N`` rows."""
        rows = int(left.shape[0])
        if transpose_right:
            cols = int(right.shape[0])
            k_dim = int(right.shape[1])
            name = "gemm_f16_tn"
            stride_right = int(right.shape[1] if stride_b is None else stride_b)
        else:
            cols = int(right.shape[1])
            k_dim = int(right.shape[0])
            name = "gemm_f16_nn"
            stride_right = int(right.shape[1] if stride_b is None else stride_b)
        if int(left.shape[1]) != k_dim and stride_a is None:
            message = "gemm K does not match"
            raise RuntimeError(message)
        k_left = int(left.shape[1])
        padded_m = _pad(rows, TILE_M)
        padded_n = _pad(cols, TILE_N)
        host = np.zeros((padded_m, k_left), dtype=np.float16)
        host[:rows] = left
        # The nn kernel reads right as [K, N]. Pad N, not K.
        if transpose_right:
            weight = np.zeros((padded_n, k_dim), dtype=np.float16)
            weight[:cols] = right
            stride_right = k_dim
        else:
            weight = np.zeros((k_dim, padded_n), dtype=np.float16)
            weight[:, :cols] = right
            stride_right = padded_n
        left_h = self.upload(np.ascontiguousarray(host))
        right_h = self.upload(np.ascontiguousarray(weight))
        dest = self.zeros(padded_m * padded_n * 4)
        dims = np.array(
            [padded_m, padded_n, k_left, k_left if stride_a is None else stride_a, stride_right],
            dtype=np.int32,
        )
        self._gemm(name, (left_h, right_h, dest), dims, scales=False)
        values = self.download(dest, np.float32, (padded_m, padded_n))[:rows, :cols]
        self.release(left_h)
        self.release(right_h)
        self.release(dest)
        return values

    def gemm_half_heads(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        transpose_right: bool,
    ) -> np.ndarray:
        """``left`` is ``[heads, M, K]``. One command buffer per chunk of heads."""
        heads = int(left.shape[0])
        rows = int(left.shape[1])
        if transpose_right:
            cols = int(right.shape[1])
        else:
            cols = int(right.shape[2])
        padded_m = _pad(rows, TILE_M)
        padded_n = _pad(cols, TILE_N)
        dest_bytes = padded_m * padded_n * 4
        chunk = max(1, _HEAD_BUDGET // max(dest_bytes, 1))
        parts = [
            self._gemm_half_chunk(
                left[start : min(heads, start + chunk)],
                right[start : min(heads, start + chunk)],
                transpose_right=transpose_right,
            )
            for start in range(0, heads, chunk)
        ]
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=0)

    def _gemm_half_chunk(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        transpose_right: bool,
    ) -> np.ndarray:
        count = int(left.shape[0])
        rows = int(left.shape[1])
        k_left = int(left.shape[2])
        if transpose_right:
            cols = int(right.shape[1])
            k_dim = int(right.shape[2])
            name = "gemm_f16_tn"
        else:
            cols = int(right.shape[2])
            k_dim = int(right.shape[1])
            name = "gemm_f16_nn"
        padded_m = _pad(rows, TILE_M)
        padded_n = _pad(cols, TILE_N)
        group = _Dispatch(self)
        staged = []
        for index in range(count):
            host = np.zeros((padded_m, k_left), dtype=np.float16)
            host[:rows] = left[index]
            if transpose_right:
                weight = np.zeros((padded_n, k_dim), dtype=np.float16)
                weight[:cols] = right[index]
                stride_right = k_dim
            else:
                weight = np.zeros((k_dim, padded_n), dtype=np.float16)
                weight[:, :cols] = right[index]
                stride_right = padded_n
            left_h = self.upload(host)
            right_h = self.upload(weight)
            dest = self.zeros(padded_m * padded_n * 4)
            dims = np.array([padded_m, padded_n, k_left, k_left, stride_right], dtype=np.int32)
            group.gemm(name, (left_h, right_h, dest), dims, tile_n=TILE_N)
            staged.append((left_h, right_h, dest))
        group.finish(name)
        out = np.empty((count, rows, cols), dtype=np.float32)
        for index, (left_h, right_h, dest) in enumerate(staged):
            out[index] = self.download(dest, np.float32, (padded_m, padded_n))[:rows, :cols]
            self.release(left_h)
            self.release(right_h)
            self.release(dest)
        return out

    def _gemm(
        self,
        name: str,
        handles,
        dims: np.ndarray,
        *,
        scales: bool,
        tile_n: int = TILE_N,
    ) -> None:
        group = _Dispatch(self)
        group.gemm(name, handles, dims, tile_n=tile_n)
        group.finish(name)
        del scales

    def conv3d(
        self,
        source: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        *,
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        out_size: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """``source`` is ``[C, T, H, W]``. A wide convolution is im2col plus ``matmul2d``."""
        in_c, frames, height, width = (int(v) for v in source.shape)
        origin = weight
        if weight.ndim == 4:
            weight = weight.reshape(
                weight.shape[0], weight.shape[1], 1, weight.shape[2], weight.shape[3]
            )
            stride = (1, stride[1], stride[2])
            padding = (0, padding[1], padding[2])
        _out_c, _ic, kt, kh, kw = (int(v) for v in weight.shape)
        if out_size is None:
            out_t = _conv_out(frames, kt, stride[0], padding[0])
            out_h = _conv_out(height, kh, stride[1], padding[1])
            out_w = _conv_out(width, kw, stride[2], padding[2])
        else:
            out_t, out_h, out_w = out_size
        src_h = self._stage("conv-src", np.ascontiguousarray(source, dtype=np.float32))
        group = _Dispatch(self)
        result = self._enqueue_conv(
            group,
            src_h,
            origin,
            weight,
            bias,
            in_c,
            frames,
            height,
            width,
            stride,
            padding,
            out_t,
            out_h,
            out_w,
        )
        group.finish("conv3d")
        return self.download(result, np.float32, (int(weight.shape[0]), out_t, out_h, out_w))

    def activated_conv(
        self,
        source: np.ndarray,
        gamma: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray | None,
        previous: np.ndarray | None,
        *,
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        out_size: tuple[int, int, int],
        tail: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """RMS, SiLU, the causal prepend, and the convolution share one submission."""
        data = np.ascontiguousarray(source, dtype=np.float32)
        channels, frames, height, width = (int(v) for v in data.shape)
        src_h = self._stage("conv-src", data)
        group = _Dispatch(self)
        sites = frames * height * width
        scale = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
        gamma_h = self._keep(("channel-rms", id(gamma), channels), scale)
        shape = np.array([channels, sites], dtype=np.int32)
        group.launch("rms_nchw_f32", (src_h, gamma_h), shape.tobytes(), sites)
        group.launch("silu_f32", (src_h,), np.int32(data.size).tobytes(), int(data.size))
        take = min(int(tail), frames)
        tail_h = self._buffer("conv-tail", channels * take * height * width * 4)
        grid = np.array([channels, frames, height, width, take], dtype=np.int32)
        group.launch(
            "pack_time_tail",
            (src_h, tail_h),
            grid.tobytes(),
            channels * take * height * width,
        )
        conv_src = src_h
        conv_frames = frames
        if previous is not None:
            prev = np.ascontiguousarray(previous, dtype=np.float32)
            prev_t = int(prev.shape[1])
            prev_h = self._stage("conv-prev", prev)
            joined = self._buffer("conv-join", channels * (prev_t + frames) * sites // frames * 4)
            spec = np.array([channels, prev_t, frames, height * width], dtype=np.int32)
            group.launch(
                "nchw_prepend",
                (prev_h, src_h, joined),
                spec.tobytes(),
                channels * (prev_t + frames) * height * width,
            )
            conv_src = joined
            conv_frames = prev_t + frames
        origin = weight
        if weight.ndim == 4:
            weight = weight.reshape(
                weight.shape[0], weight.shape[1], 1, weight.shape[2], weight.shape[3]
            )
            stride = (1, stride[1], stride[2])
            padding = (0, padding[1], padding[2])
        out_t, out_h, out_w = out_size
        out = self._enqueue_conv(
            group,
            conv_src,
            origin,
            weight,
            bias,
            channels,
            conv_frames,
            height,
            width,
            stride,
            padding,
            out_t,
            out_h,
            out_w,
        )
        group.finish("activated-conv")
        values = self.download(out, np.float32, (int(weight.shape[0]), out_t, out_h, out_w))
        packed = self.download(tail_h, np.float32, (channels, take, height, width))
        return values, packed

    def channel_rms(self, values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """RMS over axis 0, then the per-channel scale. ``values`` stays contiguous."""
        return self._channel(values, gamma, silu=False)

    def rms_silu(self, values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Channel RMS, the scale, then SiLU. One command buffer."""
        return self._channel(values, gamma, silu=True)

    def _channel(self, values: np.ndarray, gamma: np.ndarray, *, silu: bool) -> np.ndarray:
        data = np.ascontiguousarray(values, dtype=np.float32)
        channels = int(data.shape[0])
        sites = int(data.size // channels)
        scale = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
        handle = self.upload(data)
        gamma_h = self._keep(("channel-rms", id(gamma), channels), scale)
        shape = np.array([channels, sites], dtype=np.int32)
        group = _Dispatch(self)
        group.launch("rms_nchw_f32", (handle, gamma_h), shape.tobytes(), sites)
        if silu:
            group.launch("silu_f32", (handle,), np.int32(data.size).tobytes(), int(data.size))
        group.finish("channel_rms")
        out = self.download(handle, np.float32, data.shape)
        self.release(handle)
        return out

    def _enqueue_conv(
        self,
        group: _Dispatch,
        src_h,
        origin,
        weight: np.ndarray,
        bias: np.ndarray | None,
        in_c: int,
        frames: int,
        height: int,
        width: int,
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        out_t: int,
        out_h: int,
        out_w: int,
    ):
        out_c, _ic, kt, kh, kw = (int(v) for v in weight.shape)
        taps = in_c * kt * kh * kw
        outputs = out_t * out_h * out_w
        if out_c < 32 or outputs < 1024 or taps < 64:
            return self._enqueue_small(
                group,
                src_h,
                origin,
                weight,
                bias,
                out_c,
                in_c,
                frames,
                height,
                width,
                kt,
                kh,
                kw,
                stride,
                padding,
                out_t,
                out_h,
                out_w,
            )
        return self._enqueue_wide(
            group,
            src_h,
            origin,
            weight,
            bias,
            out_c,
            in_c,
            frames,
            height,
            width,
            kt,
            kh,
            kw,
            taps,
            stride,
            padding,
            out_t,
            out_h,
            out_w,
            outputs,
        )

    def _enqueue_small(
        self,
        group,
        src_h,
        origin,
        weight,
        bias,
        out_c,
        in_c,
        frames,
        height,
        width,
        kt,
        kh,
        kw,
        stride,
        padding,
        out_t,
        out_h,
        out_w,
    ):
        args = _conv_args(
            out_c, in_c, frames, height, width, kt, kh, kw, padding, stride, out_t, out_h, out_w, 0
        )
        packed = np.ascontiguousarray(weight, dtype=np.float32)
        held = self._resident(origin, factory=lambda: packed)
        dest = self._buffer("conv-nchw", out_c * out_t * out_h * out_w * 4)
        group.launch(
            "conv3d_f32",
            (src_h, held, self._bias_row(bias, out_c), dest),
            args.tobytes(),
            out_c * out_t * out_h * out_w,
        )
        return dest

    def _enqueue_wide(
        self,
        group,
        src_h,
        origin,
        weight,
        bias,
        out_c,
        in_c,
        frames,
        height,
        width,
        kt,
        kh,
        kw,
        taps,
        stride,
        padding,
        out_t,
        out_h,
        out_w,
        outputs,
    ):
        k_pad = _pad(taps, 64)
        padded_n = _pad(out_c, 64)
        weight_h = self._resident_conv(origin, weight, padded_n, k_pad, out_c, taps)
        col_m = _pad(min(_CONV_ROWS, outputs), TILE_M)
        col_h = self._buffer("conv-col", col_m * k_pad * 2)
        if k_pad != taps:
            memory.zero(col_h, 0, col_m * k_pad * 2)
        dest_rows = _pad(outputs, TILE_M)
        dest = self._buffer("conv-dest", dest_rows * padded_n * 4)
        row0 = 0
        while row0 < outputs:
            valid = min(_CONV_ROWS, outputs - row0)
            padded_m = _pad(valid, TILE_M)
            args = _conv_args(
                out_c,
                in_c,
                frames,
                height,
                width,
                kt,
                kh,
                kw,
                padding,
                stride,
                out_t,
                out_h,
                out_w,
                k_pad,
                row0,
            )
            dims = np.array([padded_m, padded_n, k_pad, k_pad, k_pad], dtype=np.int32)
            group.launch("im2col_f16", (src_h, col_h), args.tobytes(), valid * taps)
            group.gemm(
                "gemm_f16_tn64",
                (col_h, weight_h, dest),
                dims,
                tile_n=64,
                offsets=(0, 0, row0 * padded_n * 4),
            )
            row0 += valid
        nchw = self._buffer("conv-nchw", out_c * outputs * 4)
        epi = np.zeros(1, dtype=_EPI)
        epi["out_c"] = out_c
        epi["padded_n"] = padded_n
        epi["out_t"] = out_t
        epi["out_h"] = out_h
        epi["out_w"] = out_w
        epi["limit"] = np.float32(FP16_MAX)
        group.launch(
            "conv_epilogue_nchw",
            (dest, self._bias_row(bias, out_c), nchw),
            epi.tobytes(),
            outputs,
        )
        return nchw

    def _bias_row(self, bias: np.ndarray | None, out_c: int):
        if bias is None:
            return self._keep(("conv-bias-zero", out_c), np.zeros(out_c, dtype=np.float32))
        flat = np.zeros(out_c, dtype=np.float32)
        src = np.ascontiguousarray(bias, dtype=np.float32).reshape(-1)
        flat[:out_c] = src[:out_c]
        return self._keep(("conv-bias", id(bias), out_c), flat)

    def _buffer(self, key: str, nbytes: int):
        found = self._scratch.get(key)
        if found is not None and found[0] >= nbytes:
            return found[1]
        if found is not None:
            self.release(found[1])
        handle = memory.allocate(int(nbytes), owner=self.owner, zero=False)
        self._scratch[key] = (int(nbytes), handle)
        return handle

    def _stage(self, key: str, array: np.ndarray):
        raw = np.ascontiguousarray(array)
        handle = self._buffer(key, int(raw.nbytes))
        memory.store(handle, 0, raw)
        return handle

    def _launch(self, name: str, handles, constants: bytes, threads: int) -> None:
        group = _Dispatch(self)
        group.launch(name, handles, constants, threads)
        group.finish(name)

    def _pipe(self, name: str):
        found = self._pipes.get(name)
        if found is not None:
            return found
        function = self._lib.newFunctionWithName_(name)
        if function is None:
            message = f"shader is missing {name}"
            raise RuntimeError(message)
        pipe, error = memory.device.newComputePipelineStateWithFunction_error_(function, None)
        if pipe is None:
            message = f"pipeline {name} failed: {error}"
            raise RuntimeError(message)
        self._pipes[name] = pipe
        return pipe


class _Dispatch:
    """Several kernels, one commit, one wait."""

    def __init__(self, gpu: Gpu) -> None:
        self.gpu = gpu
        self.command = memory.command_buffer(gpu.queue)
        self.encoder = self.command.computeCommandEncoder()
        self._blobs: list = []

    def gemm(
        self,
        name: str,
        handles,
        dims: np.ndarray,
        *,
        tile_n: int,
        offsets: tuple[int, ...] | None = None,
    ) -> None:
        dims_h = self.gpu.upload(np.ascontiguousarray(dims))
        self._blobs.append(dims_h)
        groups_x = int(dims[1]) // tile_n
        groups_y = int(dims[0]) // TILE_M
        encoder = self.encoder
        encoder.setComputePipelineState_(self.gpu._pipe(name))
        self._bind(handles, dims_h, offsets)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            _size(groups_x, groups_y, 1),
            _size(_THREADS, 1, 1),
        )

    def launch(
        self,
        name: str,
        handles,
        constants: bytes,
        threads: int,
        offsets: tuple[int, ...] | None = None,
    ) -> None:
        blob = self.gpu.upload(np.frombuffer(constants, dtype=np.uint8))
        self._blobs.append(blob)
        encoder = self.encoder
        encoder.setComputePipelineState_(self.gpu._pipe(name))
        self._bind(handles, blob, offsets)
        width = 256
        groups = max(1, (threads + width - 1) // width)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            _size(groups, 1, 1),
            _size(width, 1, 1),
        )

    def rows(self, name: str, handles, constants: bytes, count: int) -> None:
        """One threadgroup of 256 lanes per row."""
        blob = self.gpu.upload(np.frombuffer(constants, dtype=np.uint8))
        self._blobs.append(blob)
        encoder = self.encoder
        encoder.setComputePipelineState_(self.gpu._pipe(name))
        self._bind(handles, blob, None)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            _size(max(int(count), 1), 1, 1),
            _size(256, 1, 1),
        )

    def _bind(self, handles, extra, offsets: tuple[int, ...] | None) -> None:
        shifts = offsets or (0,) * len(handles)
        for index, handle in enumerate(handles):
            bind(self.encoder, index, handle, int(shifts[index]) if index < len(shifts) else 0)
        bind(self.encoder, len(handles), extra)

    def finish(self, label: str) -> None:
        self.encoder.endEncoding()
        self.command.commit()
        status, error = memory.retire(self.command)
        for blob in self._blobs:
            self.gpu.release(blob)
        if int(status) != 4:
            message = f"{label} status {status} {error}"
            raise RuntimeError(message)


def _library():
    import Metal

    source = _SHADER.read_text(encoding="utf-8")
    compile_options = Metal.MTLCompileOptions.alloc().init()
    compile_options.setLanguageVersion_((4 << 16) + 1)
    library, error = memory.device.newLibraryWithSource_options_error_(
        source, compile_options, None
    )
    if library is None:
        message = f"shader failed: {error}"
        raise RuntimeError(message)
    return library


def _conv_out(length: int, kernel: int, stride: int, padding: int) -> int:
    return (length + 2 * padding - kernel) // stride + 1


_CONV_ROWS = 4096


def _conv_args(
    out_c,
    in_c,
    frames,
    height,
    width,
    kt,
    kh,
    kw,
    padding,
    stride,
    out_t,
    out_h,
    out_w,
    k_pad,
    row0=0,
):
    return np.array(
        [
            out_c,
            in_c,
            frames,
            height,
            width,
            kt,
            kh,
            kw,
            padding[0],
            padding[1],
            padding[2],
            stride[0],
            stride[1],
            stride[2],
            out_t,
            out_h,
            out_w,
            k_pad,
            row0,
        ],
        dtype=np.int32,
    )


_EPI = np.dtype(
    [
        ("out_c", np.int32),
        ("padded_n", np.int32),
        ("out_t", np.int32),
        ("out_h", np.int32),
        ("out_w", np.int32),
        ("limit", np.float32),
    ]
)


def _pad(count: int, multiple: int) -> int:
    return max(multiple, ((count + multiple - 1) // multiple) * multiple)


def _store_half(values: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(values), axis=1, keepdims=True)
    factor = np.where(peak > FP16_MAX, peak / np.float32(FP16_MAX), np.float32(1.0))
    stored = (values / factor).astype(np.float16)
    return (stored.astype(np.float32) * factor).astype(np.float32)


def _size(width: int, height: int, depth: int):
    import Metal

    return Metal.MTLSize(width, height, depth)
