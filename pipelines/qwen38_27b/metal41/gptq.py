"""GPTQ rounding on the fp4 grid, with the FOEM column update.

Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", 2022, Algorithm 1. Zheng et al., "First-Order Error
Matters", AAAI 2026, arXiv:2507.11017, Algorithm 1. Columns are rounded in
order. Each residual is spread over the columns not yet rounded, weighted by
the upper Cholesky factor of the inverse input Gram. FOEM shrinks that
residual by ``beta`` times the column's drift from the original weights, then
pulls the next column toward those original weights by the same factor.

The grid is the one the kernels read: E2M1, ties to even, one round-up E8M0
scale per 32 along K. The scale group stays 32. The lazy block stays 128.
``beta`` 0 is GPTQ. The published FOEM tables use ``beta`` 0.1.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from qwen38_27b.metal41.formats import BLOCK, FP4_MAGNITUDE, FP4_QMAX, biased_e8m0

_ROW_CHUNK = 4096

_GRID = FP4_MAGNITUDE.astype(np.float32)
_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def _round_column(values: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest E2M1 codes and the restored column. Past 6 saturates."""
    scaled = values / scale
    magnitude = np.abs(scaled)
    choice = np.searchsorted(_EDGE, magnitude, side="left").astype(np.uint8)
    on_edge = (choice < 7) & (magnitude == _EDGE[np.minimum(choice, 6)])
    choice = np.where(on_edge & ((choice & 1) == 1), choice + 1, choice).astype(np.uint8)
    sign = np.where(np.signbit(scaled), np.float32(-1.0), np.float32(1.0))
    restored = sign * _GRID[choice] * scale
    code = choice | (np.signbit(scaled).astype(np.uint8) << np.uint8(3))
    return code, restored


def _pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << np.uint8(4))


def _inverse_factor(hessian: np.ndarray, damp: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper factor of ``(H + damp)^-1``, and the dead-column mask."""
    base = np.array(hessian, dtype=np.float64, copy=True)
    last = None
    for factor in (1.0, 10.0, 100.0):
        hessian = np.array(base, copy=True)
        diagonal = np.diag(hessian).copy()
        dead = diagonal <= 0.0
        hessian[dead, dead] = 1.0
        diagonal[dead] = 1.0
        hessian[np.diag_indices_from(hessian)] += (damp * factor) * float(np.mean(diagonal))
        try:
            lower = np.linalg.cholesky(hessian)
        except np.linalg.LinAlgError as exc:
            last = exc
            continue
        eye = np.eye(hessian.shape[0], dtype=np.float64)
        inverse_lower = np.linalg.solve(lower, eye)
        inverse = inverse_lower.T @ inverse_lower
        upper = np.linalg.cholesky(inverse).T
        return np.ascontiguousarray(upper, dtype=np.float32), dead
    message = "input Gram is not positive definite"
    raise np.linalg.LinAlgError(message) from last


def foem_fp4(
    matrix: np.ndarray,
    hessian: np.ndarray,
    *,
    beta: float = 0.1,
    damp: float = 0.01,
    block: int = 128,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``[rows, inner]`` as fp4. ``hessian`` is the Gram along K.

    Returns packed nibbles ``[rows, inner / 2]`` and E8M0 scales ``[rows, inner / 32]``.
    """
    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    if inner % BLOCK or block % BLOCK or hessian.shape != (inner, inner):
        message = f"inner {inner} block {block} does not match the Gram {hessian.shape}"
        raise ValueError(message)
    upper, dead = _inverse_factor(hessian, damp)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales = np.empty((rows, inner // BLOCK), dtype=np.uint8)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        work = np.array(weight[start:stop], dtype=np.float32)
        original = np.array(work, dtype=np.float32)
        work[:, dead] = 0.0
        nibbles = np.empty(work.shape, dtype=np.uint8)
        for b0 in range(0, inner, block):
            b1 = min(b0 + block, inner)
            width = b1 - b0
            part = work[:, b0:b1]
            orig = original[:, b0:b1]
            errors = np.zeros((stop - start, width), dtype=np.float32)
            factor = upper[b0:b1, b0:b1]
            scale = np.ones(stop - start, dtype=np.float32)
            for i in range(width):
                column = b0 + i
                if dead[column]:
                    nibbles[:, column] = 0
                    if column % BLOCK == 0:
                        scales[start:stop, column // BLOCK] = 0
                    continue
                if column % BLOCK == 0:
                    amax = np.max(np.abs(part[:, i : i + BLOCK]), axis=1)
                    biased, scale = biased_e8m0(amax, FP4_QMAX)
                    scales[start:stop, column // BLOCK] = biased
                code, restored = _round_column(part[:, i], scale)
                nibbles[:, column] = code
                diagonal = float(factor[i, i])
                if diagonal == 0.0:
                    message = f"column {column} has a zero inverse diagonal"
                    raise RuntimeError(message)
                drift = part[:, i] - orig[:, i]
                error = ((part[:, i] - restored) - np.float32(beta) * drift) / np.float32(diagonal)
                errors[:, i] = error
                if i + 1 < width:
                    coeff = np.array(factor[i, i + 1 :], dtype=np.float32, copy=True)
                    coeff[dead[column + 1 : b1]] = 0.0
                    part[:, i + 1 :] -= error[:, None] * coeff
                    nxt = column + 1
                    if not dead[nxt]:
                        part[:, i + 1] -= np.float32(beta) * (part[:, i + 1] - orig[:, i + 1])
            if b1 < inner:
                coeff = np.array(upper[b0:b1, b1:], dtype=np.float32, copy=True)
                coeff[:, dead[b1:]] = 0.0
                work[:, b1:] -= errors @ coeff
                if not dead[b1]:
                    work[:, b1] -= np.float32(beta) * (work[:, b1] - original[:, b1])
            if not np.isfinite(work).all():
                message = "FOEM update left a non-finite weight"
                raise RuntimeError(message)
        packed[start:stop] = _pack_nibbles(nibbles)
    return packed, scales


def _align64(value: int) -> int:
    return (int(value) + 63) & ~63


def gram_bytes(count: int, width: int) -> tuple[int, int]:
    """Transpose bytes and scratch bytes for one half ``XᵀX``."""
    token_pad = _align64(count)
    width_pad = _align64(width)
    return width_pad * token_pad * 2, width_pad * width_pad * 4


def encode_gram(
    session: Any,
    source: Any,
    acc: Any,
    count: int,
    span: int,
    width: int,
    transposed: Any,
    scratch: Any,
    *,
    half: bool,
) -> None:
    """Add half-precision ``XᵀX`` for the first ``count`` rows into the device Gram.

    ``transposed`` and ``scratch`` are the sizes from ``gram_bytes``. The reduction
    is one ``matmul2d``.
    """
    if count < 1 or width < 1:
        return
    token_pad = _align64(count)
    width_pad = _align64(width)
    kernel = "gram_transpose_f16" if half else "gram_transpose_f32"
    session.encode(
        kernel,
        [(source, 0), (transposed, 0)],
        [struct.pack("<I", int(value)) for value in (count, span, width, token_pad, width_pad)],
        (token_pad // 64, width_pad, 1),
        (64, 1, 1),
    )
    session.encode(
        "gram_xtx",
        [(transposed, 0), (scratch, 0), (acc, 0)],
        [struct.pack("<I", int(value)) for value in (width, width_pad, token_pad)],
        (width_pad // 64, width_pad // 64, 1),
        (128, 1, 1),
    )


class DeviceFactor:
    """Upper Cholesky factor of one inverse Gram, resident for every matrix on that stream."""

    def __init__(
        self,
        session: Any,
        upper: np.ndarray | Any,
        dead: np.ndarray,
        *,
        resident_dead: Any = None,
    ) -> None:
        self._dead = np.asarray(dead, dtype=bool)
        if resident_dead is None:
            self.inner = int(np.asarray(upper).shape[0])
            self.upper = session.upload(np.ascontiguousarray(upper, dtype=np.float32))
            self.dead = session.upload(np.ascontiguousarray(self._dead, dtype=np.uint8))
        else:
            self.inner = int(self._dead.shape[0])
            self.upper = upper
            self.dead = resident_dead

    def release(self) -> None:
        from qwen38_27b.metal41.session import release_buffer

        release_buffer(self.upper)
        release_buffer(self.dead)
        self.upper = None
        self.dead = None


def _flag_value(buffer: Any) -> int:
    raw = memoryview(buffer.contents().as_buffer(4))[:4]
    return int(struct.unpack("<I", bytes(raw))[0])


def _set_flag(buffer: Any, value: int) -> None:
    memoryview(buffer.contents().as_buffer(4))[:4] = struct.pack("<I", int(value))


def _panel_count(n: int, k: int) -> int:
    trailing = n - k - 32
    tiles = trailing // 32
    return tiles * (tiles + 1) // 2


def _encode_cholesky(session: Any, work: Any, n: int, flag: Any) -> None:
    _set_flag(flag, 0)
    session.begin()
    for k in range(0, n, 32):
        session.encode(
            "chol_panel",
            [(work, 0), (flag, 0)],
            [struct.pack("<I", n), struct.pack("<I", k)],
            (1, 1, 1),
            (32, 1, 1),
        )
        trailing = n - k - 32
        if trailing <= 0:
            continue
        session.encode(
            "chol_trsm",
            [(work, 0), (flag, 0)],
            [struct.pack("<I", n), struct.pack("<I", k)],
            ((trailing + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        session.encode(
            "chol_syrk",
            [(work, 0), (flag, 0)],
            [struct.pack("<I", n), struct.pack("<I", k)],
            (_panel_count(n, k), 1, 1),
            (256, 1, 1),
        )
    session.end()


def _encode_inverse(session: Any, work: Any, temp: Any, n: int) -> None:
    session.begin()
    for k in range(n - 32, -1, -32):
        session.encode(
            "trinv_panel",
            [(work, 0)],
            [struct.pack("<I", n), struct.pack("<I", k)],
            (1, 1, 1),
            (32, 1, 1),
        )
        trailing = n - k - 32
        if trailing <= 0:
            continue
        session.encode(
            "trinv_mul",
            [(work, 0), (temp, 0)],
            [struct.pack("<I", n), struct.pack("<I", k), struct.pack("<I", trailing)],
            (trailing, 1, 1),
            (1, 1, 1),
        )
        session.encode(
            "trinv_apply",
            [(work, 0), (temp, 0)],
            [struct.pack("<I", n), struct.pack("<I", k), struct.pack("<I", trailing)],
            (trailing // 32, 1, 1),
            (256, 1, 1),
        )
    session.end()


def factor_gram(
    session: Any,
    gram: Any,
    n: int,
    damp: float = 0.01,
    workspace: Any = None,
) -> DeviceFactor:
    """Upper factor of ``(H + damp)^-1`` on device. The Gram stays put."""
    from qwen38_27b.metal41.session import release_buffer

    if n < 32 or n % 32:
        message = f"Gram width {n} is not a multiple of 32"
        raise ValueError(message)
    raw = np.frombuffer(gram.contents().as_buffer(n * n * 4), dtype=np.float32)
    diag = np.array(raw.reshape(n, n).diagonal(), dtype=np.float32, copy=True)
    dead = diag <= 0.0
    live = diag.astype(np.float64, copy=True)
    live[dead] = 1.0
    mean = float(live.mean())
    held: list[Any] = []
    dead_buf = session.upload(np.ascontiguousarray(dead, dtype=np.uint8))
    held.append(dead_buf)
    try:
        work = workspace
        if work is None:
            work = session.zeros(n * n * 4)
            held.append(work)
        hinv = session.zeros(n * n * 4)
        held.append(hinv)
        temp = session.zeros(n * 32 * 4)
        held.append(temp)
        flag = session.zeros(4)
        held.append(flag)
        upper = session.zeros(n * n * 4)
        held.append(upper)
    except RuntimeError as exc:
        for buffer in held:
            release_buffer(buffer)
        message = "capture ran out of memory for the Gram factor"
        raise RuntimeError(message) from exc
    try:
        built = False
        for mult in (1.0, 10.0, 100.0):
            lam = np.float32(damp * mult * mean)
            session.begin()
            session.encode(
                "gram_prepare",
                [(gram, 0), (work, 0), (dead_buf, 0)],
                [struct.pack("<I", n), struct.pack("<f", lam)],
                ((n * n + 255) // 256, 1, 1),
                (256, 1, 1),
            )
            session.end()
            _encode_cholesky(session, work, n, flag)
            if _flag_value(flag) != 0:
                continue
            _encode_inverse(session, work, temp, n)
            tiles = (n // 32) * (n // 32 + 1) // 2
            session.begin()
            session.encode(
                "xtx_lower",
                [(work, 0), (hinv, 0)],
                [struct.pack("<I", n)],
                (tiles, 1, 1),
                (256, 1, 1),
            )
            session.end()
            _encode_cholesky(session, hinv, n, flag)
            if _flag_value(flag) != 0:
                continue
            session.begin()
            session.encode(
                "transpose_upper",
                [(hinv, 0), (upper, 0)],
                [struct.pack("<I", n)],
                (n, 1, 1),
                (1, 1, 1),
            )
            session.end()
            built = True
            break
        if not built:
            message = "input Gram is not positive definite"
            raise np.linalg.LinAlgError(message)
        factor = DeviceFactor(session, upper, dead, resident_dead=dead_buf)
        held = [item for item in held if item is not upper and item is not dead_buf]
        return factor
    finally:
        for buffer in held:
            release_buffer(buffer)


def foem_fp4_device(
    session: Any,
    matrix: np.ndarray,
    factor: DeviceFactor,
    *,
    beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack one resident matrix. A streamed checkout uses ``foem_fp4_rows``."""
    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    return foem_fp4_rows(
        session,
        rows,
        inner,
        lambda start, stop: weight[start:stop],
        factor,
        beta=beta,
    )


def foem_fp4_rows(
    session: Any,
    rows: int,
    inner: int,
    load,
    factor: DeviceFactor,
    *,
    beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``rows`` by ``inner``. ``load(start, stop)`` returns that contiguous row block."""
    if inner % BLOCK or factor.inner != inner:
        message = f"inner {inner} does not match the factor {factor.inner}"
        raise ValueError(message)
    if factor._dead.shape != (inner,):
        message = f"dead mask {factor._dead.shape} does not match inner {inner}"
        raise ValueError(message)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales = np.empty((rows, inner // BLOCK), dtype=np.uint8)
    mask = factor._dead
    staged: list[tuple[int, dict[str, Any]]] = []
    from qwen38_27b.metal41.session import release_buffer

    try:
        session.begin()
        try:
            for start in range(0, rows, _ROW_CHUNK):
                stop = min(start + _ROW_CHUNK, rows)
                tile = np.ascontiguousarray(load(start, stop), dtype=np.float32)
                if tile.shape != (stop - start, inner):
                    message = f"row block {tile.shape} is not {(stop - start, inner)}"
                    raise ValueError(message)
                work = np.array(tile, copy=True)
                work[:, mask] = 0.0
                staged.append((start, _stage_chunk(session, factor, tile, work, beta)))
                session.submit()
            session.end()
        finally:
            if getattr(session, "_pass_encoder", None) is not None:
                session.finish()
        for start, buffers in staged:
            chunk = rows - start if start + _ROW_CHUNK > rows else _ROW_CHUNK
            packed[start : start + chunk] = np.frombuffer(
                session.download(buffers["packed"], chunk * (inner // 2)), dtype=np.uint8
            ).reshape(chunk, inner // 2)
            scales[start : start + chunk] = np.frombuffer(
                session.download(buffers["scales"], chunk * (inner // BLOCK)), dtype=np.uint8
            ).reshape(chunk, inner // BLOCK)
    finally:
        for _start, buffers in staged:
            for buffer in buffers.values():
                release_buffer(buffer)
    return packed, scales


def _stage_chunk(
    session: Any,
    factor: DeviceFactor,
    tile: np.ndarray,
    work: np.ndarray,
    beta: float,
) -> dict[str, Any]:
    """Upload one row block and encode it on the open command buffer."""
    chunk, inner = tile.shape
    original = session.upload(tile)
    device_work = session.upload(np.ascontiguousarray(work))
    codes = session.zeros(chunk * inner)
    scale_buf = session.zeros(chunk * (inner // BLOCK))
    errors = session.zeros(chunk * 128 * 4)
    packed_buf = session.zeros(chunk * (inner // 2))
    beta_blob = struct.pack("<f", np.float32(beta))
    _encode_chunk(
        session,
        factor,
        original,
        device_work,
        codes,
        scale_buf,
        errors,
        packed_buf,
        chunk,
        inner,
        beta_blob,
    )
    return {
        "original": original,
        "work": device_work,
        "codes": codes,
        "scales": scale_buf,
        "errors": errors,
        "packed": packed_buf,
    }


def _encode_chunk(
    session: Any,
    factor: DeviceFactor,
    original: Any,
    device_work: Any,
    codes: Any,
    scale_buf: Any,
    errors: Any,
    packed_buf: Any,
    chunk: int,
    inner: int,
    beta_blob: bytes,
) -> None:
    for b0 in range(0, inner, 128):
        width = min(128, inner - b0)
        session.encode(
            "foem_block",
            [
                (device_work, 0),
                (original, 0),
                (factor.upper, 0),
                (factor.dead, 0),
                (codes, 0),
                (scale_buf, 0),
                (errors, 0),
            ],
            [
                struct.pack("<I", chunk),
                struct.pack("<I", inner),
                struct.pack("<I", b0),
                struct.pack("<I", width),
                beta_blob,
            ],
            ((chunk + 255) // 256, 1, 1),
            (256, 1, 1),
        )
        remain = inner - (b0 + width)
        if remain:
            session.encode(
                "foem_tail",
                [
                    (device_work, 0),
                    (original, 0),
                    (errors, 0),
                    (factor.upper, 0),
                    (factor.dead, 0),
                ],
                [
                    struct.pack("<I", chunk),
                    struct.pack("<I", inner),
                    struct.pack("<I", b0),
                    struct.pack("<I", width),
                    beta_blob,
                ],
                ((remain + 63) // 64, (chunk + 3) // 4, 1),
                (64, 4, 1),
            )
    session.encode(
        "foem_pack",
        [(codes, 0), (packed_buf, 0)],
        [struct.pack("<I", chunk), struct.pack("<I", inner)],
        ((chunk * (inner // 2) + 255) // 256, 1, 1),
        (256, 1, 1),
    )


def pad_fp4(
    packed: np.ndarray,
    scales: np.ndarray,
    stored_rows: int,
    stored_inner: int,
) -> tuple[bytes, bytes]:
    """Zero-pad a logical fp4 pack out to the stored tile shape."""
    rows, half = packed.shape
    inner = half * 2
    if stored_rows < rows or stored_inner < inner or scales.shape != (rows, inner // BLOCK):
        message = "stored shape is smaller than the FOEM pack"
        raise ValueError(message)
    data = np.zeros((stored_rows, stored_inner // 2), dtype=np.uint8)
    plane = np.zeros((stored_rows, stored_inner // BLOCK), dtype=np.uint8)
    data[:rows, :half] = packed
    plane[:rows, : scales.shape[1]] = scales
    return data.tobytes(), plane.tobytes()


def restore_fp4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Float matrix from packed nibbles and E8M0 scales."""
    rows, half = packed.shape
    inner = half * 2
    codes = np.empty((rows, inner), dtype=np.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    sign = np.where(codes & 0x8, np.float32(-1.0), np.float32(1.0))
    values = sign * _GRID[codes & 0x7]
    factors = scales.astype(np.int32)
    restored = values.reshape(rows, inner // BLOCK, BLOCK)
    restored = restored * np.ldexp(np.ones(factors.shape, dtype=np.float32), factors - 127)[..., None]
    return restored.reshape(rows, inner)


def quadratic_form(session: Any, gram: Any, n: int, rows: np.ndarray) -> float:
    """Sum of ``v H vᵀ`` for each row of ``rows``. ``gram`` stays on device."""
    from qwen38_27b.metal41.session import release_buffer

    matrix = np.ascontiguousarray(rows, dtype=np.float32)
    sample, width = matrix.shape
    if width != n or sample < 1:
        message = f"quadratic rows {matrix.shape} do not match Gram {n}"
        raise ValueError(message)
    uploaded = session.upload(matrix)
    out = session.zeros(sample * 4)
    try:
        session.begin()
        session.encode(
            "gram_quadratic",
            [(gram, 0), (uploaded, 0), (out, 0)],
            [struct.pack("<I", n), struct.pack("<I", sample)],
            (sample, 1, 1),
            (256, 1, 1),
        )
        session.end()
        return float(session.download_f32(out, sample).sum())
    finally:
        release_buffer(uploaded)
        release_buffer(out)


def device_relative_error(
    session: Any,
    gram: Any,
    n: int,
    weight: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
) -> float:
    """``tr(E H Eᵀ) / tr(W H Wᵀ)`` for the sampled rows. ``H`` stays on device."""
    restored = restore_fp4(packed, scales).astype(np.float32)
    error = np.ascontiguousarray(weight, dtype=np.float32) - restored
    denom = quadratic_form(session, gram, n, weight)
    if denom == 0.0:
        return 0.0
    return quadratic_form(session, gram, n, error) / denom


def relative_error(
    weight: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
    hessian: np.ndarray,
    *,
    sample: int = 256,
) -> float:
    """``tr(E H E^T) / tr(W H W^T)`` on at most ``sample`` rows."""
    step = max(1, weight.shape[0] // sample)
    rows = slice(0, weight.shape[0], step)
    got = weight[rows].astype(np.float64)
    error = got - restore_fp4(packed[rows], scales[rows]).astype(np.float64)
    gram = np.asarray(hessian, dtype=np.float64)
    denom = float(np.sum((got @ gram) * got))
    if denom == 0.0:
        return 0.0
    return float(np.sum((error @ gram) * error) / denom)
