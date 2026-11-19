"""One knot of diffusion blocks. The token stream stays half until the knot ends."""

from __future__ import annotations

import numpy as np

from wan22_ti2v_5b.metal41.activations import apply_rope, layernorm, softmax
from wan22_ti2v_5b.metal41.gpu import Gpu, _Dispatch, _pad
from wan22_ti2v_5b.metal41.quant import quantize_activation
from wan22_ti2v_5b.metal41.schedule import rope_tables
from wan22_ti2v_5b.metal41.spec import DIM, EPS, HEAD, HEADS, TILE_N_I8
from wan22_ti2v_5b.metal41.weights import Matrix

_ROW = np.dtype(
    [
        ("rows", np.int32),
        ("cols", np.int32),
        ("has_weight", np.int32),
        ("has_bias", np.int32),
        ("eps", np.float32),
    ]
)
_SOFT = np.dtype(
    [
        ("rows", np.int32),
        ("stride", np.int32),
        ("valid", np.int32),
        ("scale", np.float32),
    ]
)
_SCALE = np.float32(HEAD**-0.5)
_FLASH_Q = 16
_FLASH_K = 48
_FLASH_THREADS = 128
_FLASH = np.dtype(
    [
        ("q_len", np.uint32),
        ("kv_len", np.uint32),
        ("heads", np.uint32),
        ("origin", np.uint32),
        ("scale", np.float32),
        ("tiles", np.uint32),
        ("q_cap", np.uint32),
        ("kv_cap", np.uint32),
    ]
)


class Stream:
    """Thirty blocks share one command buffer. The residual stays half on device."""

    def __init__(self, gpu: Gpu) -> None:
        self.gpu = gpu
        self._rows = 0
        self._text_rows = 0
        self._handles: list = []
        self._cross: list[tuple | None] = []
        self._cross_id: int | None = None
        self._tables: tuple | None = None

    def run(
        self,
        tokens: np.ndarray,
        proj: np.ndarray,
        blocks: list,
        context: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        rows, cols = int(tokens.shape[0]), int(tokens.shape[1])
        text_rows = int(context.shape[0])
        if cols != DIM or int(context.shape[1]) != DIM:
            message = "device block expects the transformer width"
            raise RuntimeError(message)
        if int(proj.shape[0]) != rows or int(proj.shape[1]) != DIM * 6:
            message = "time projection does not match the token rows"
            raise RuntimeError(message)
        self._fit(rows, text_rows, blocks)
        if self._cross_id != id(context):
            self._drop_cross()
            self._cross_id = id(context)
            self._cross = [None] * len(blocks)
            self._text_src = self._upload_rows(context, self._text_pm)
        slots = np.ascontiguousarray(
            proj.reshape(rows, 6, DIM).transpose(1, 0, 2), dtype=np.float16
        )
        proj_h = self.gpu.upload(slots)
        host = np.zeros((self._pm, DIM), dtype=np.float16)
        host[:rows] = tokens.astype(np.float16)
        self.gpu.release(self._stream)
        self._stream = self.gpu.upload(host)
        self._handles[0] = self._stream
        group = _Dispatch(self.gpu)
        for index, block in enumerate(blocks):
            self._block(group, block, index, rows, text_rows, height, width, proj_h)
        group.finish("dit")
        self.gpu.release(proj_h)
        out = self.gpu.download(self._stream, np.float16, (self._pm, DIM))[:rows]
        return out.astype(np.float32)

    def _block(
        self,
        group: _Dispatch,
        block,
        index: int,
        rows: int,
        text_rows: int,
        height: int,
        width: int,
        proj_h,
    ) -> None:
        self._norm(group, self._stream, self._normed, self._pm, DIM, None, None)
        self._modulate(group, rows, proj_h, block, scale=1, shift=0)
        self._qkv(group, block, rows)
        self._attend(group, rows, height, width)
        self._project(group, self._update, self._pm, DIM, block.self_o, self._acc)
        self._gate(group, rows, self._acc, proj_h, block, slot=2)
        self._norm(group, self._stream, self._normed, self._pm, DIM, block.norm3, block.norm3_bias)
        self._project(group, self._normed, self._pm, DIM, block.cross_q, self._q)
        self._rms(group, self._q, rows, DIM, block.cross_norm_q)
        self._cross_pair(group, block, index, text_rows)
        key_h, value_h = self._cross[index]
        self._attend_pair(group, rows, text_rows, self._q, key_h, value_h, self._pm, self._text_cap)
        self._project(group, self._update, self._pm, DIM, block.cross_o, self._acc)
        count = rows * DIM
        group.launch("add_h", (self._stream, self._acc), np.int32(count).tobytes(), count)
        self._norm(group, self._stream, self._normed, self._pm, DIM, None, None)
        self._modulate(group, rows, proj_h, block, scale=4, shift=3)
        self._project(group, self._normed, self._pm, DIM, block.ffn0, self._acc)
        group.launch(
            "gelu_tanh_h",
            (self._acc,),
            np.int32(self._pm * int(block.ffn0.logical_n)).tobytes(),
            self._pm * int(block.ffn0.logical_n),
        )
        wide = int(block.ffn0.logical_n)
        self._project(group, self._acc, self._pm, wide, block.ffn2, self._update)
        self._gate(group, rows, self._update, proj_h, block, slot=5)

    def _qkv(self, group: _Dispatch, block, rows: int) -> None:
        self._project(group, self._normed, self._pm, DIM, block.self_q, self._q)
        self._project(group, self._normed, self._pm, DIM, block.self_k, self._k)
        self._project(group, self._normed, self._pm, DIM, block.self_v, self._v)
        self._rms(group, self._q, rows, DIM, block.self_norm_q)
        self._rms(group, self._k, rows, DIM, block.self_norm_k)

    def _attend(self, group: _Dispatch, rows: int, height: int, width: int) -> None:
        self._rope(group, self._q, rows, height, width)
        self._rope(group, self._k, rows, height, width)
        self._attend_pair(group, rows, rows, self._q, self._k, self._v, self._pm, self._kv_cap)

    def _attend_pair(
        self, group: _Dispatch, rows: int, keys: int, query, key, value, q_cap: int, kv_cap: int
    ) -> None:
        launch_flash(group, query, key, value, self._update, rows, keys, q_cap, kv_cap)

    def _cross_pair(self, group: _Dispatch, block, index: int, text_rows: int) -> None:
        cached = self._cross[index]
        if cached is not None:
            return
        key_h = self.gpu.zeros(self._text_cap * DIM * 2)
        value_h = self.gpu.zeros(self._text_cap * DIM * 2)
        self._handles.extend((key_h, value_h))
        self._project(group, self._text_src, self._text_pm, DIM, block.cross_k, key_h)
        self._rms(group, key_h, text_rows, DIM, block.cross_norm_k)
        self._project(group, self._text_src, self._text_pm, DIM, block.cross_v, value_h)
        self._cross[index] = (key_h, value_h)

    def _project(
        self, group: _Dispatch, source, rows: int, cols: int, matrix: Matrix, dest
    ) -> None:
        out_n = matrix.stored_n()
        logical = int(matrix.logical_n)
        group.rows(
            "quantize_i8_h",
            (source, self._codes, self._row_scale),
            np.array([rows, cols], dtype=np.int32).tobytes(),
            rows,
        )
        weight_h, weight_off = self._plane(matrix, "packed")
        scale_h, scale_off = self._plane(matrix, "scales")
        bias_h, bias_off = self._bias_plane(matrix)
        dims = np.array([rows, out_n, cols, cols, cols], dtype=np.int32)
        group.gemm(
            "gemm_i8",
            (self._codes, weight_h, self._iacc),
            dims,
            tile_n=TILE_N_I8,
            offsets=(0, weight_off, 0),
        )
        shape = np.array([rows, out_n, logical], dtype=np.int32)
        group.launch(
            "scale_i8_h",
            (self._iacc, self._row_scale, scale_h, bias_h, dest),
            shape.tobytes(),
            rows * out_n,
            offsets=(0, 0, scale_off, bias_off, 0),
        )

    def _norm(self, group, source, dest, rows: int, cols: int, weight, bias) -> None:
        args = np.zeros(1, dtype=_ROW)
        args["rows"] = rows
        args["cols"] = cols
        args["has_weight"] = 0 if weight is None else 1
        args["has_bias"] = 0 if bias is None else 1
        args["eps"] = np.float32(EPS)
        group.rows(
            "layernorm_h",
            (source, self._vector(weight), self._vector(bias), dest),
            args.tobytes(),
            rows,
        )

    def _rms(self, group, values, rows: int, cols: int, weight) -> None:
        args = np.zeros(1, dtype=_ROW)
        args["rows"] = rows
        args["cols"] = cols
        args["has_weight"] = 1
        args["has_bias"] = 0
        args["eps"] = np.float32(EPS)
        group.rows(
            "rmsnorm_h",
            (values, self._vector(weight), values),
            args.tobytes(),
            rows,
        )

    def _modulate(self, group, rows: int, proj_h, block, *, scale: int, shift: int) -> None:
        mod = self._vector(block.modulation)
        shape = np.array([rows, DIM], dtype=np.int32)
        group.launch(
            "modulate_h",
            (self._normed, proj_h, proj_h, mod, mod, self._normed),
            shape.tobytes(),
            rows * DIM,
            offsets=(
                0,
                self._proj_offset(scale, rows),
                self._proj_offset(shift, rows),
                scale * DIM * 2,
                shift * DIM * 2,
                0,
            ),
        )

    def _gate(self, group, rows: int, update, proj_h, block, *, slot: int) -> None:
        mod = self._vector(block.modulation)
        shape = np.array([rows, DIM], dtype=np.int32)
        group.launch(
            "residual_h",
            (self._stream, update, proj_h, mod, self._stream),
            shape.tobytes(),
            rows * DIM,
            offsets=(0, 0, self._proj_offset(slot, rows), slot * DIM * 2, 0),
        )

    def _rope(self, group, values, rows: int, height: int, width: int) -> None:
        grid = np.array([rows, HEADS, height, width], dtype=np.int32)
        group.launch(
            "rope_h",
            (values, self._tables[0], self._tables[1], self._tables[2]),
            grid.tobytes(),
            rows * HEADS * 64,
        )

    def _fit(self, rows: int, text_rows: int, blocks: list) -> None:
        if self._rows == rows and self._text_rows == text_rows and self._tables is not None:
            return
        self._drop_cross()
        for handle in self._handles:
            self.gpu.release(handle)
        self._handles = []
        self._rows = rows
        self._text_rows = text_rows
        self._pm = _pad(rows, 64)
        self._text_pm = _pad(text_rows, 64)
        self._kv_cap = max(self._pm, _pad(rows, _FLASH_K))
        self._text_cap = max(self._text_pm, _pad(text_rows, _FLASH_K))
        wide = max(self._pm, self._text_pm)
        pairs = (
            ("_stream", self._pm * DIM * 2),
            ("_normed", self._pm * DIM * 2),
            ("_acc", wide * 14336 * 2),
            ("_iacc", wide * 14336 * 4),
            ("_update", self._pm * DIM * 2),
            ("_q", self._pm * DIM * 2),
            ("_k", self._kv_cap * DIM * 2),
            ("_v", self._kv_cap * DIM * 2),
            ("_codes", wide * 14336),
            ("_row_scale", wide * 4),
            ("_text_src", self._text_pm * DIM * 2),
        )
        for name, nbytes in pairs:
            handle = self.gpu.zeros(nbytes)
            setattr(self, name, handle)
            self._handles.append(handle)
        tables = rope_tables()
        self._tables = tuple(
            self.gpu._keep(("rope", index), table) for index, table in enumerate(tables)
        )
        self._cross = [None] * len(blocks)
        self._cross_id = None
        self._zero = self.gpu._keep(("zero", DIM, "half"), np.zeros(DIM, dtype=np.float16))

    def _upload_rows(self, rows: np.ndarray, padded: int):
        host = np.zeros((padded, DIM), dtype=np.float16)
        host[: int(rows.shape[0])] = rows.astype(np.float16)
        handle = self.gpu.upload(host)
        self._handles.append(handle)
        return handle

    def _drop_cross(self) -> None:
        self._cross = []
        self._cross_id = None

    def _vector(self, values: np.ndarray | None):
        if values is None:
            return self._zero
        return self.gpu._keep(
            ("vec", id(values)),
            np.ascontiguousarray(values, dtype=np.float16).reshape(-1),
        )

    def _plane(self, matrix: Matrix, part: str):
        if part == "packed" and matrix.packed_at is not None:
            return matrix.span, int(matrix.packed_at)
        if part == "scales" and matrix.scales_at is not None:
            return matrix.span, int(matrix.scales_at)
        values = matrix.packed if part == "packed" else matrix.scales
        return self.gpu._resident(values), 0

    def _bias_plane(self, matrix: Matrix):
        if matrix.bias_at is not None:
            return matrix.span, int(matrix.bias_at)
        out_n = matrix.stored_n()
        row = np.zeros(out_n, dtype=np.float16)
        if matrix.bias is not None:
            flat = np.ascontiguousarray(matrix.bias, dtype=np.float16).reshape(-1)
            row[: int(matrix.logical_n)] = flat[:out_n]
        packed = matrix.packed
        key = id(packed) if packed is not None else (id(matrix.span), matrix.packed_at)
        return self.gpu._keep((key, "bias", out_n), row), 0

    def _proj_offset(self, slot: int, rows: int) -> int:
        return slot * rows * DIM * 2


def check_stream(gpu: Gpu) -> None:
    """The device norm, quantize, and RoPE match the host formulas."""
    rng = np.random.default_rng(4)
    rows = rng.standard_normal((8, 32)).astype(np.float32)
    weight = rng.standard_normal(32).astype(np.float32)
    bias = rng.standard_normal(32).astype(np.float32)
    _expect_close(gpu, "layernorm_f32", rows, layernorm(rows, weight, bias), weight, bias)
    coded, scale = quantize_activation(rows)
    got_codes, got_scale = _quantize(gpu, rows)
    if not np.array_equal(got_codes, coded) or float(np.max(np.abs(got_scale - scale))) > 1e-5:
        message = "device quantize drifted"
        raise RuntimeError(message)
    rounded = np.ascontiguousarray(rows, dtype=np.float16)
    coded_h, scale_h = quantize_activation(rounded.astype(np.float32))
    got_codes, got_scale = _quantize_half(gpu, rounded)
    if not np.array_equal(got_codes, coded_h) or float(np.max(np.abs(got_scale - scale_h))) > 1e-5:
        message = "half quantize drifted"
        raise RuntimeError(message)
    tokens = rng.standard_normal((4, HEAD)).astype(np.float32)
    expect = apply_rope(tokens, 2, 2)
    got = _rope(gpu, tokens)
    if float(np.max(np.abs(got - expect))) > 1e-5:
        message = "device rope drifted"
        raise RuntimeError(message)
    scores = rng.standard_normal((4, 6)).astype(np.float32)
    expect = softmax(scores * _SCALE).astype(np.float16).astype(np.float32)
    got = _softmax(gpu, scores)
    if float(np.max(np.abs(got - expect))) > 1e-3:
        message = "device softmax drifted"
        raise RuntimeError(message)
    error = _flash(gpu)
    print(f"stream attention {error:.6e}", flush=True)
    if error > 5e-2:
        message = "device attention drifted"
        raise RuntimeError(message)
    print("stream max abs ok", flush=True)


def launch_flash(
    group: _Dispatch,
    query,
    key,
    value,
    dest,
    q_len: int,
    kv_len: int,
    q_cap: int,
    kv_cap: int,
) -> None:
    """One flash attention. The half rows are ``[cap, heads, 128]``."""
    if q_cap < _pad(q_len, _FLASH_Q) or kv_cap < _pad(max(kv_len, 1), _FLASH_K):
        message = "flash tile does not fit"
        raise RuntimeError(message)
    tiles = (q_len + _FLASH_Q - 1) // _FLASH_Q
    args = np.zeros(1, dtype=_FLASH)
    args["q_len"] = q_len
    args["kv_len"] = kv_len
    args["heads"] = HEADS
    args["origin"] = 0
    args["scale"] = _SCALE
    args["tiles"] = tiles
    args["q_cap"] = q_cap
    args["kv_cap"] = kv_cap
    group.grid(
        "attend_flash",
        (query, key, value, dest),
        args.tobytes(),
        tiles * HEADS,
        _FLASH_THREADS,
    )


def _expect_close(gpu: Gpu, name: str, rows, expect, weight, bias) -> None:
    del name
    src = gpu.upload(rows)
    dst = gpu.zeros(rows.size * 4)
    args = np.zeros(1, dtype=_ROW)
    args["rows"] = rows.shape[0]
    args["cols"] = rows.shape[1]
    args["has_weight"] = 1
    args["has_bias"] = 1
    args["eps"] = np.float32(EPS)
    group = _Dispatch(gpu)
    group.rows(
        "layernorm_f32",
        (src, gpu.upload(weight), gpu.upload(bias), dst),
        args.tobytes(),
        int(rows.shape[0]),
    )
    group.finish("layernorm_f32")
    got = gpu.download(dst, np.float32, rows.shape)
    gpu.release(src)
    gpu.release(dst)
    if float(np.max(np.abs(got - expect))) > 1e-4:
        message = "device layernorm drifted"
        raise RuntimeError(message)


def _quantize_half(gpu: Gpu, rows: np.ndarray):
    src = gpu.upload(np.ascontiguousarray(rows, dtype=np.float16))
    codes = gpu.zeros(rows.size)
    scale = gpu.zeros(rows.shape[0] * 4)
    group = _Dispatch(gpu)
    group.rows(
        "quantize_i8_h",
        (src, codes, scale),
        np.array(rows.shape, dtype=np.int32).tobytes(),
        int(rows.shape[0]),
    )
    group.finish("quantize_i8_h")
    got_codes = gpu.download(codes, np.int8, rows.shape)
    got_scale = gpu.download(scale, np.float32, (rows.shape[0],))
    gpu.release(src)
    gpu.release(codes)
    gpu.release(scale)
    return got_codes, got_scale


def _quantize(gpu: Gpu, rows: np.ndarray):
    src = gpu.upload(rows)
    codes = gpu.zeros(rows.size)
    scale = gpu.zeros(rows.shape[0] * 4)
    group = _Dispatch(gpu)
    group.rows(
        "quantize_i8",
        (src, codes, scale),
        np.array(rows.shape, dtype=np.int32).tobytes(),
        int(rows.shape[0]),
    )
    group.finish("quantize_i8")
    got_codes = gpu.download(codes, np.int8, rows.shape)
    got_scale = gpu.download(scale, np.float32, (rows.shape[0],))
    gpu.release(src)
    gpu.release(codes)
    gpu.release(scale)
    return got_codes, got_scale


def _rope(gpu: Gpu, tokens: np.ndarray) -> np.ndarray:
    tables = rope_tables()
    handles = [gpu.upload(tokens)]
    handles.extend(gpu.upload(table) for table in tables)
    grid = np.array([tokens.shape[0], 1, 2, 2], dtype=np.int32)
    group = _Dispatch(gpu)
    group.launch("rope_f32", tuple(handles), grid.tobytes(), int(tokens.shape[0]) * 64)
    group.finish("rope_f32")
    got = gpu.download(handles[0], np.float32, tokens.shape)
    for handle in handles:
        gpu.release(handle)
    return got


def _flash(gpu: Gpu) -> float:
    """Live rows that do not fill a tile, with poison past the live edge."""
    rng = np.random.default_rng(5)
    q_len, kv_len = 20, 13
    q_cap, kv_cap = _pad(q_len, _FLASH_Q), _pad(kv_len, _FLASH_K)
    query = np.zeros((q_cap, HEADS, HEAD), dtype=np.float32)
    key = np.zeros((kv_cap, HEADS, HEAD), dtype=np.float32)
    value = np.zeros((kv_cap, HEADS, HEAD), dtype=np.float32)
    noise = np.float32(0.05)
    query[:q_len] = rng.standard_normal((q_len, HEADS, HEAD)).astype(np.float32) * noise
    key[:kv_len] = rng.standard_normal((kv_len, HEADS, HEAD)).astype(np.float32) * noise
    value[:kv_len] = rng.standard_normal((kv_len, HEADS, HEAD)).astype(np.float32) * noise
    query[q_len:] = np.float32(3)
    key[kv_len:] = np.float32(3)
    value[kv_len:] = np.float32(3)
    expect = _attend_reference(query[:q_len], key[:kv_len], value[:kv_len])
    q_h = gpu.upload(np.ascontiguousarray(query.reshape(q_cap, DIM), dtype=np.float16))
    k_h = gpu.upload(np.ascontiguousarray(key.reshape(kv_cap, DIM), dtype=np.float16))
    v_h = gpu.upload(np.ascontiguousarray(value.reshape(kv_cap, DIM), dtype=np.float16))
    dest = gpu.zeros(q_cap * DIM * 4)
    group = _Dispatch(gpu)
    launch_flash(group, q_h, k_h, v_h, dest, q_len, kv_len, q_cap, kv_cap)
    group.finish("attend_flash")
    got = gpu.download(dest, np.float16, (q_cap, DIM))[:q_len].astype(np.float32)
    for handle in (q_h, k_h, v_h, dest):
        gpu.release(handle)
    return float(np.max(np.abs(got - expect.reshape(q_len, DIM))))


def _attend_reference(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    q = query.astype(np.float16).astype(np.float32)
    k = key.astype(np.float16).astype(np.float32)
    v = value.astype(np.float16).astype(np.float32)
    dots = np.einsum("thd,khd->htk", q, k) * _SCALE
    peak = np.max(dots, axis=-1, keepdims=True)
    weights = np.exp(dots - peak)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("htk,khd->thd", weights, v)


def _softmax(gpu: Gpu, scores: np.ndarray) -> np.ndarray:
    src = gpu.upload(scores)
    dst = gpu.zeros(scores.size * 2)
    soft = np.zeros(1, dtype=_SOFT)
    soft["rows"] = scores.shape[0]
    soft["stride"] = scores.shape[1]
    soft["valid"] = scores.shape[1]
    soft["scale"] = _SCALE
    group = _Dispatch(gpu)
    group.launch("softmax_f16", (src, dst), soft.tobytes(), int(scores.shape[0]))
    group.finish("softmax_f16")
    got = gpu.download(dst, np.float16, scores.shape).astype(np.float32)
    gpu.release(src)
    gpu.release(dst)
    return got
