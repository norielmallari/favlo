"""Metal device session for Gemma 4 matmul and the block kernels."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from favlo.memory import memory

_SHADER = Path(__file__).resolve().parent / "shaders" / "kernels.metal"
_KERNELS = (
    "matmul_fp8",
    "matmul_fp4",
    "matmul_f16",
    "matmul_f16_64",
    "matmul_f16_64_batch",
    "matmul_fp4_64",
    "matmul_fp4_64_batch",
    "matmul_fp4_batch",
    "dequant_q4_0",
    "dequant_q4_0_batch",
    "repack_q4_fp4",
    "gather_half_slots",
    "pad_half_rows",
    "scatter_slot_rows",
    "group_expert_slots",
    "gemv_fp4_routed",
    "write_rows",
    "write_rows_tight",
    "gemv_q4_0",
    "gemv_q4_0_routed",
    "mix_experts",
    "gemm_q4_0_n",
    "qmm_q4_0",
    "gemm_q4_0",
    "write_row",
    "rmsnorm",
    "rmsnorm_rows",
    "rmsnorm_heads",
    "zero_buffer",
    "expand_fp16",
    "embed_q4_0",
    "presence_penalty",
    "router_topk",
    "topk_groups",
    "sample_topk",
    "rope_pairs",
    "rope_rows",
    "rope_axial",
    "attend",
    "attend_span",
    "attend_decode",
    "attend_span4",
    "attend_vision",
    "store_kv",
    "store_kv_rows",
    "geglu",
    "geglu_rows",
    "gather_rows",
    "scatter_axpy",
    "geglu_pair",
    "residual_scale",
    "add_scaled",
    "scale_copy",
    "vision_standardize",
    "vision_positions",
    "pool_patches",
    "compact_rows",
    "softcap",
    "router_norm",
)
_TILE_M = 8
_TILE_N = 32


def _pair(result: object) -> tuple[object, str | None]:
    if isinstance(result, tuple) and len(result) == 2:
        value, error = result
        if error is not None:
            return None, str(error)
        return value, None
    return result, None


class Session:
    """One compiled library. Weights and the cache stay with the runtime."""

    def __init__(self, owner: str) -> None:
        import Metal

        self.owner = owner
        self.metal = Metal
        device = memory.device
        options = Metal.MTLCompileOptions.alloc().init()
        options.setLanguageVersion_((4 << 16) + 1)
        library, error = _pair(
            device.newLibraryWithSource_options_error_(
                _SHADER.read_text(encoding="utf-8"),
                options,
                None,
            )
        )
        if library is None:
            message = f"shader library failed to compile: {error}"
            raise RuntimeError(message)
        self.device = device
        self.queue = memory.make_queue()
        self._fence = device.newFence()
        self._fence_armed = False
        self.pipes: dict[str, object] = {}
        for name in _KERNELS:
            function = library.newFunctionWithName_(name)
            if function is None:
                message = f"{name} is not in the shader library"
                raise RuntimeError(message)
            pipeline, pipe_error = _pair(
                device.newComputePipelineStateWithFunction_error_(function, None)
            )
            if pipeline is None:
                message = f"{name} pipeline failed: {pipe_error}"
                raise RuntimeError(message)
            self.pipes[name] = pipeline
        self._command = None
        self._encoder = None
        self._inflight: list[object] = []

    def zeros(self, nbytes: int) -> object:
        return memory.allocate(nbytes, owner=self.owner, zero=True)

    def upload(self, data: bytes | np.ndarray) -> object:
        return memory.upload(data, owner=self.owner)

    def upload_planes(self, data: bytes, scales: bytes) -> tuple[object, object]:
        handles = memory.upload_group([data, scales], owner=self.owner)
        return handles[0], handles[1]

    def store(self, handle: object, data: bytes) -> None:
        memory.store(handle, 0, data)

    def download_f32(self, handle: object, count: int) -> np.ndarray:
        self.flush()
        raw = memory.load(handle, 0, count * 4)
        return np.frombuffer(raw, dtype="<f4").copy()

    def encode(
        self,
        name: str,
        handles: list[object],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        origins: list[int] | None = None,
    ) -> None:
        """Record one dispatch. The device runs it when :meth:`flush` commits."""
        self._open()
        encoder = self._encoder
        encoder.setComputePipelineState_(self.pipes[name])
        for index, handle in enumerate(handles):
            origin = 0 if origins is None else origins[index]
            memory.bind(encoder, index, handle, origin)
        base = len(handles)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            self.metal.MTLSizeMake(groups[0], groups[1], groups[2]),
            self.metal.MTLSizeMake(threads[0], threads[1], threads[2]),
        )

    def split(self) -> None:
        """End this encoder so the next one can indirect-dispatch a buffer it wrote."""
        self._open()
        encoder = self._encoder
        encoder.updateFence_(self._fence)
        self._fence_armed = True
        encoder.endEncoding()
        encoder = self._command.computeCommandEncoder()
        encoder.waitForFence_(self._fence)
        self._encoder = encoder

    def encode_indirect(
        self,
        name: str,
        handles: list[object],
        constants: list[bytes],
        indirect: object,
        indirect_offset: int,
        threads: tuple[int, int, int],
        origins: list[int] | None = None,
    ) -> None:
        """Dispatch ``name``. Three uints at ``indirect_offset`` are the grid."""
        self._open()
        encoder = self._encoder
        encoder.setComputePipelineState_(self.pipes[name])
        for index, handle in enumerate(handles):
            origin = 0 if origins is None else origins[index]
            memory.bind(encoder, index, handle, origin)
        base = len(handles)
        for index, blob in enumerate(constants):
            encoder.setBytes_length_atIndex_(blob, len(blob), base + index)
        memory.bind_indirect(
            encoder,
            indirect,
            indirect_offset,
            self.metal.MTLSizeMake(threads[0], threads[1], threads[2]),
        )

    def commit(self) -> None:
        """Submit the open buffer. The next buffer waits for this one before it touches memory."""
        encoder = self._encoder
        command = self._command
        if encoder is None or command is None:
            return
        encoder.updateFence_(self._fence)
        self._fence_armed = True
        encoder.endEncoding()
        command.commit()
        self._inflight.append(command)
        self._encoder = None
        self._command = None

    def flush(self) -> None:
        """Submit the open buffer and wait until every submitted buffer has finished."""
        self.commit()
        inflight = self._inflight
        self._inflight = []
        for command in inflight:
            command.waitUntilCompleted()
            failed = getattr(command, "error", None)
            error = failed() if callable(failed) else None
            if error is not None:
                message = f"command buffer failed: {error}"
                raise RuntimeError(message)

    def _open(self) -> None:
        if self._encoder is not None:
            return
        command = self.queue.commandBuffer()
        encoder = command.computeCommandEncoder()
        if self._fence_armed:
            encoder.waitForFence_(self._fence)
        self._command = command
        self._encoder = encoder

    def dispatch(
        self,
        name: str,
        handles: list[object],
        constants: list[bytes],
        groups: tuple[int, int, int],
        threads: tuple[int, int, int],
        origins: list[int] | None = None,
    ) -> None:
        self.encode(name, handles, constants, groups, threads, origins)
        self.flush()

    def _u32(self, value: int) -> bytes:
        return struct.pack("<I", int(value))

    def _f32(self, value: float) -> bytes:
        return struct.pack("<f", float(value))

    def matmul(
        self,
        kind: str,
        activation: object,
        weight: object,
        scales: object | None,
        destination: object,
        rows: int,
        cols: int,
        inner: int,
        *,
        commit: bool = True,
    ) -> None:
        if rows % _TILE_M or cols % _TILE_N or inner % 32:
            message = f"matmul tile rows {rows} cols {cols} inner {inner}"
            raise ValueError(message)
        pipe = self.pipes[kind]
        width = int(pipe.threadExecutionWidth())
        handles = [activation, weight]
        if scales is not None:
            handles.append(scales)
        handles.append(destination)
        self.encode(
            kind,
            handles,
            [self._u32(rows), self._u32(cols), self._u32(inner)],
            (cols // _TILE_N, rows // _TILE_M, 1),
            (width * 4, 1, 1),
        )
        if commit:
            self.commit()

    def close(self) -> None:
        self.flush()
        memory.sweep(self.owner)


def _check_fp4(session: Session) -> None:
    from gemma4_26b_a4b.planes import pack_matrix, restore_matrix

    rows, cols, inner = 8, 32, 256
    activation = np.zeros((rows, inner), dtype=np.float32)
    activation[0, :8] = np.array([1, -1, 0.5, 2, -3, 4, 0, 6], dtype=np.float32)
    weight = np.zeros((cols, inner), dtype=np.float32)
    weight[0, 0] = 1
    weight[1, 1] = 2
    weight[2, 2] = -1
    weight[3, :] = 0.25
    blob = pack_matrix(weight, "fp4")
    restored = restore_matrix(blob, cols, inner, "fp4")
    data_bytes = cols * (inner // 2)
    data = blob[:data_bytes]
    scales = blob[data_bytes:]
    act = session.upload(np.ascontiguousarray(activation.astype(np.float16)))
    weight_h, scale_h = session.upload_planes(data, scales)
    dest = session.zeros(rows * cols * 4)
    session.matmul("matmul_fp4", act, weight_h, scale_h, dest, rows, cols, inner)
    got = session.download_f32(dest, rows * cols).reshape(rows, cols)
    expect = activation @ restored.T
    if not np.allclose(got, expect, rtol=2e-2, atol=2e-2):
        message = f"fp4 matmul mismatch max {np.max(np.abs(got - expect))}"
        raise RuntimeError(message)
    print("kernels fp4 matmul ok", flush=True)


def _check_f16(session: Session) -> None:
    rows, cols, inner = 8, 32, 64
    activation = np.zeros((rows, inner), dtype=np.float32)
    activation[0, :4] = np.array([1, -1, 0.5, 2], dtype=np.float32)
    weight = np.zeros((cols, inner), dtype=np.float32)
    weight[0, 0] = 1
    weight[1, 1] = -2
    weight[2, :] = 0.5
    data = np.ascontiguousarray(weight.astype(np.float16)).tobytes()
    act = session.upload(np.ascontiguousarray(activation.astype(np.float16)))
    weight_h = session.upload(data)
    dest = session.zeros(rows * cols * 4)
    session.matmul("matmul_f16", act, weight_h, None, dest, rows, cols, inner)
    got = session.download_f32(dest, rows * cols).reshape(rows, cols)
    restored = weight.astype(np.float16).astype(np.float32)
    expect = activation.astype(np.float16).astype(np.float32) @ restored.T
    if not np.allclose(got, expect, rtol=1e-2, atol=1e-2):
        message = f"fp16 matmul mismatch max {np.max(np.abs(got - expect))}"
        raise RuntimeError(message)
    print("kernels fp16 matmul ok", flush=True)


def _check_qmm(session: Session) -> None:
    from gemma4_26b_a4b.planes import align_k, matrix_nbytes, pack_matrix, restore_matrix

    rows, inner = 32, 64
    pad = align_k(inner, "q4_0")
    rng = np.random.default_rng(1)
    left = rng.normal(0.0, 0.05, size=(rows, inner)).astype(np.float32)
    right = rng.normal(0.0, 0.05, size=(rows, inner)).astype(np.float32)
    blob = pack_matrix(left, "q4_0") + pack_matrix(right, "q4_0")
    packed = session.upload(blob)
    origins = np.zeros(2, dtype=np.uint64)
    origins[1] = matrix_nbytes(rows, inner, "q4_0")
    origin_h = session.upload(origins.tobytes())
    n0, n1 = 20, 6
    act = rng.normal(0.0, 0.5, size=(n0 + n1, pad)).astype(np.float32)
    act_h = session.upload(np.ascontiguousarray(act).tobytes())
    jobs = np.asarray([[0, 0, n0, 0], [1, 0, n1, n0]], dtype=np.uint32)
    job_h = session.upload(np.ascontiguousarray(jobs).tobytes())
    dest = session.zeros((n0 + n1) * rows * 4)
    session.dispatch(
        "qmm_q4_0",
        [packed, origin_h, job_h, act_h, dest],
        [session._u32(pad), session._u32(rows), session._u32(inner)],
        ((rows + 3) // 4, int(jobs.shape[0]), 1),
        (32, 1, 1),
    )
    got = np.frombuffer(memory.load(dest, 0, (n0 + n1) * rows * 4), dtype=np.float32)
    got = got.reshape(n0 + n1, rows)
    split = int(origins[1])
    restored_left = restore_matrix(blob[:split], rows, inner, "q4_0")
    restored_right = restore_matrix(blob[split:], rows, inner, "q4_0")
    expect = np.vstack([act[:n0, :inner] @ restored_left.T, act[n0:, :inner] @ restored_right.T])
    if not np.allclose(got, expect, rtol=1e-3, atol=1e-3):
        message = f"q4_0 routed gemm mismatch max {np.max(np.abs(got - expect))}"
        raise RuntimeError(message)
    print("kernels q4_0 routed gemm ok", flush=True)


def _check_q4_0(session: Session) -> None:
    from gemma4_26b_a4b.planes import align_k, pack_matrix, restore_matrix

    cols, inner = 32, 64
    weight = np.zeros((cols, inner), dtype=np.float32)
    weight[0, 0] = 8.0
    weight[0, 16] = -4.0
    weight[1, 1] = 3.0
    weight[3, :] = 0.25
    blob = pack_matrix(weight, "q4_0")
    restored = restore_matrix(blob, cols, inner, "q4_0")
    k_pad = align_k(inner, "q4_0")
    packed = session.upload(blob)
    half = session.zeros(cols * k_pad * 2)
    blocks = cols * (k_pad // 32)
    session.dispatch(
        "dequant_q4_0",
        [packed, half],
        [session._u32(k_pad), session._u32(cols)],
        (int(np.ceil(blocks / 256)), 1, 1),
        (256, 1, 1),
    )
    raw = memory.load(half, 0, cols * k_pad * 2)
    got = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(cols, k_pad)
    if not np.allclose(got[:, :inner], restored, rtol=1e-3, atol=1e-3):
        message = f"q4_0 dequant mismatch max {np.max(np.abs(got[:, :inner] - restored))}"
        raise RuntimeError(message)
    print("kernels q4_0 dequant ok", flush=True)
    activation = np.zeros(k_pad, dtype=np.float32)
    activation[0] = 1.0
    activation[16] = 0.5
    act = session.upload(np.ascontiguousarray(activation).tobytes())
    out = session.zeros(cols * 4)
    session.dispatch(
        "gemv_q4_0",
        [packed, act, out],
        [session._u32(k_pad), session._u32(cols), session._u32(inner)],
        ((cols + 7) // 8, 1, 1),
        (64, 1, 1),
    )
    got_out = np.frombuffer(memory.load(out, 0, cols * 4), dtype=np.float32)
    expect = restored @ activation[:inner]
    if not np.allclose(got_out, expect, rtol=1e-3, atol=1e-3):
        message = f"q4_0 gemv mismatch max {np.max(np.abs(got_out - expect))}"
        raise RuntimeError(message)
    wide_inner = 128
    wide_pad = align_k(wide_inner, "q4_0")
    wide = np.zeros((128, wide_inner), dtype=np.float32)
    wide[:, 0] = 1.0
    wide[:, 16] = -0.5
    wide[:, 80] = 0.5
    wide[40, :] = 0.25
    wide_blob = pack_matrix(wide, "q4_0")
    wide_restored = restore_matrix(wide_blob, 128, wide_inner, "q4_0")
    wide_pack = session.upload(wide_blob)
    wide_activation = np.zeros(wide_pad, dtype=np.float32)
    wide_activation[0] = 1.0
    wide_activation[16] = 0.5
    wide_activation[80] = -1.0
    wide_act = session.upload(np.ascontiguousarray(wide_activation).tobytes())
    wide_out = session.zeros(128 * 4)
    session.dispatch(
        "gemv_q4_0",
        [wide_pack, wide_act, wide_out],
        [session._u32(wide_pad), session._u32(128), session._u32(wide_inner)],
        ((128 + 7) // 8, 1, 1),
        (64, 1, 1),
    )
    got_wide = np.frombuffer(memory.load(wide_out, 0, 128 * 4), dtype=np.float32)
    expect_wide = wide_restored @ wide_activation[:wide_inner]
    if not np.allclose(got_wide, expect_wide, rtol=1e-3, atol=1e-3):
        message = f"q4_0 gemv mismatch max {np.max(np.abs(got_wide - expect_wide))}"
        raise RuntimeError(message)
    short_tokens = 4
    short_act = np.zeros((short_tokens, wide_pad), dtype=np.float32)
    short_act[:, 0] = 1.0
    short_act[:, 16] = 0.5
    short_act[:, 80] = -1.0
    short_src = session.upload(np.ascontiguousarray(short_act).tobytes())
    short_out = session.zeros(short_tokens * 128 * 4)
    session.dispatch(
        "gemm_q4_0_n",
        [wide_pack, short_src, short_out],
        [
            session._u32(wide_pad),
            session._u32(128),
            session._u32(short_tokens),
            session._u32(wide_inner),
        ],
        ((128 + 7) // 8, 1, 1),
        (64, 1, 1),
    )
    got_short = np.frombuffer(memory.load(short_out, 0, short_tokens * 128 * 4), dtype=np.float32)
    got_short = got_short.reshape(short_tokens, 128)
    expect_short = short_act[:, :wide_inner] @ wide_restored.T
    if not np.allclose(got_short, expect_short, rtol=1e-3, atol=1e-3):
        message = f"q4_0 short gemm mismatch max {np.max(np.abs(got_short - expect_short))}"
        raise RuntimeError(message)
    print("kernels q4_0 gemv ok", flush=True)
    _check_qmm(session)
    tokens = 40
    wide_inner = 256
    wide_pad = align_k(wide_inner, "q4_0")
    wide = np.zeros((128, wide_inner), dtype=np.float32)
    rng = np.random.default_rng(0)
    wide[:, :] = rng.normal(0.0, 0.05, size=wide.shape).astype(np.float32)
    wide_blob = pack_matrix(wide, "q4_0")
    wide_restored = restore_matrix(wide_blob, 128, wide_inner, "q4_0")
    wide_pack = session.upload(wide_blob)
    act_rows = rng.normal(0.0, 0.5, size=(tokens, wide_pad)).astype(np.float32)
    many = session.upload(np.ascontiguousarray(act_rows).tobytes())
    gemm_out = session.zeros(tokens * 128 * 4)
    session.dispatch(
        "gemm_q4_0",
        [wide_pack, many, gemm_out],
        [
            session._u32(wide_pad),
            session._u32(128),
            session._u32(tokens),
            session._u32(wide_inner),
        ],
        ((128 + 31) // 32, (tokens + 31) // 32, 1),
        (128, 1, 1),
    )
    got_gemm = np.frombuffer(memory.load(gemm_out, 0, tokens * 128 * 4), dtype=np.float32)
    got_gemm = got_gemm.reshape(tokens, 128)
    expect_gemm = act_rows[:, :wide_inner] @ wide_restored.T
    if not np.allclose(got_gemm, expect_gemm, rtol=1e-3, atol=1e-3):
        message = f"q4_0 gemm mismatch max {np.max(np.abs(got_gemm - expect_gemm))}"
        raise RuntimeError(message)
    print("kernels q4_0 gemm ok", flush=True)
    tile_m, tile_n, tile_k = 64, 64, 128
    tile_pad = align_k(tile_k, "q4_0")
    tile_w = rng.normal(0.0, 0.05, size=(tile_n, tile_k)).astype(np.float32)
    tile_blob = pack_matrix(tile_w, "q4_0")
    tile_restored = restore_matrix(tile_blob, tile_n, tile_k, "q4_0")
    tile_pack = session.upload(tile_blob)
    tile_half = session.zeros(tile_n * tile_pad * 2)
    tile_blocks = tile_n * (tile_pad // 32)
    session.dispatch(
        "dequant_q4_0",
        [tile_pack, tile_half],
        [session._u32(tile_pad), session._u32(tile_n)],
        (int(np.ceil(tile_blocks / 256)), 1, 1),
        (256, 1, 1),
    )
    tile_act = rng.normal(0.0, 0.5, size=(tile_m, tile_pad)).astype(np.float32)
    tile_src = session.upload(np.ascontiguousarray(tile_act).tobytes())
    tile_ah = session.zeros(tile_m * tile_pad * 2)
    session.dispatch(
        "write_rows",
        [tile_src, tile_ah],
        [session._u32(tile_k), session._u32(tile_pad), session._u32(tile_m)],
        ((tile_pad + 255) // 256, tile_m, 1),
        (256, 1, 1),
    )
    tile_dest = session.zeros(tile_m * tile_n * 4)
    width = int(session.pipes["matmul_f16_64"].threadExecutionWidth())
    session.dispatch(
        "matmul_f16_64",
        [tile_ah, tile_half, tile_dest],
        [session._u32(tile_m), session._u32(tile_n), session._u32(tile_pad)],
        (tile_n // 64, tile_m // 64, 1),
        (width * 4, 1, 1),
    )
    got_tile = np.frombuffer(memory.load(tile_dest, 0, tile_m * tile_n * 4), dtype=np.float32)
    got_tile = got_tile.reshape(tile_m, tile_n)
    expect_tile = tile_act[:, :tile_k] @ tile_restored.T
    if not np.allclose(got_tile, expect_tile, rtol=1e-2, atol=1e-2):
        message = f"q4_0 matmul2d mismatch max {np.max(np.abs(got_tile - expect_tile))}"
        raise RuntimeError(message)
    print("kernels q4_0 matmul2d ok", flush=True)


def check_matmul() -> None:
    """One fp8 tile against the packed restore. This is the kernels acceptance."""
    from gemma4_26b_a4b.planes import pack_matrix, restore_matrix

    session = Session("gemma4-26b-a4b:kernels")
    try:
        rows, cols, inner = 8, 64, 128
        activation = np.zeros((rows, inner), dtype=np.float32)
        activation[0, :8] = np.array([1, -1, 0.5, 2, -3, 4, 0, 6], dtype=np.float32)
        weight = np.zeros((cols, inner), dtype=np.float32)
        weight[0, 0] = 1
        weight[1, 1] = 2
        weight[2, 2] = -1
        weight[3, :] = 0.25
        blob = pack_matrix(weight, "fp8")
        restored = restore_matrix(blob, cols, inner, "fp8")
        packed = np.frombuffer(blob, dtype=np.uint8)
        data_bytes = cols * inner
        data = bytes(packed[:data_bytes])
        scales = bytes(packed[data_bytes:])
        half_act = np.ascontiguousarray(activation.astype(np.float16))
        act = session.upload(half_act)
        weight_h, scale_h = session.upload_planes(data, scales)
        dest = session.zeros(rows * cols * 4)
        session.matmul("matmul_fp8", act, weight_h, scale_h, dest, rows, cols, inner)
        got = session.download_f32(dest, rows * cols).reshape(rows, cols)
        expect = activation @ restored.T
        if not np.allclose(got, expect, rtol=1e-2, atol=1e-2):
            message = (
                f"fp8 matmul mismatch max {np.max(np.abs(got - expect))} "
                f"got {got[0, :8].tolist()} expect {expect[0, :8].tolist()}"
            )
            raise RuntimeError(message)
        print("kernels fp8 matmul ok", flush=True)
        _check_fp4(session)
        _check_f16(session)
        _check_q4_0(session)
        _check_select(session)
    finally:
        session.close()


def _check_select(session: Session) -> None:
    scores = np.linspace(-3.0, 5.0, 256, dtype=np.float32)
    scores[10] = 9.0
    scores[11] = 9.0
    uploaded = session.upload(np.ascontiguousarray(scores))
    dummy = session.zeros(4)
    out_scores = session.zeros(4 * 4)
    out_ids = session.zeros(4 * 4)
    session.dispatch(
        "topk_groups",
        [uploaded, dummy, out_scores, out_ids],
        [session._u32(256), session._u32(4), session._f32(1.0), session._u32(1)],
        (1, 1, 1),
        (256, 1, 1),
    )
    got = struct.unpack("<4I", memory.load(out_ids, 0, 16))
    if got != (10, 11, 255, 254):
        message = f"topk order {got}"
        raise RuntimeError(message)
    logits = np.arange(128, dtype=np.float32)
    scale = np.ones(128, dtype=np.float32)
    indices = session.zeros(8 * 4)
    weights = session.zeros(8 * 4)
    session.dispatch(
        "router_topk",
        [
            session.upload(np.ascontiguousarray(logits)),
            session.upload(np.ascontiguousarray(scale)),
            indices,
            weights,
        ],
        [session._u32(128), session._u32(8)],
        (1, 1, 1),
        (128, 1, 1),
    )
    chosen = struct.unpack("<8I", memory.load(indices, 0, 32))
    if chosen != tuple(range(127, 119, -1)):
        message = f"router order {chosen}"
        raise RuntimeError(message)
    print("kernels topk ok", flush=True)
