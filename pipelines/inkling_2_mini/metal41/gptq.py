"""Error-compensating fp4 rounding of the dense planes.

Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", 2022, Algorithm 1. Zheng et al., "First-Order
Error Matters", AAAI 2026, arXiv:2507.11017, Equation 19. Columns are rounded
in order. Each residual is spread over the columns not yet rounded, weighted
by the upper Cholesky factor of the inverse input Gram. FOEM shrinks that
residual by ``beta`` times the column's drift from the original weights, then
pulls the next column toward those original weights by the same factor. The
first column of the next block gets that pull too. ``beta`` 0 is GPTQ. The
published tables use ``beta`` 0.1.

The format is the one the kernels read: E2M1 codes with one E8M0 scale per
32, the power of two rounded up so the group peak fits when the group starts.
Hadamard rotations are not applied. The forward kernels would have to change.

A calibration run sums ``X^T X`` per input stream on the host. The column
update runs on the device. The packed planes go to one safetensors file
beside the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from inkling_2_mini.metal41.codec import e4m3_to_float

PACK_NAME = "dense-fp4.safetensors"
PACK_FORMAT = "favlo-fp4-gptq-foem-1"
BETA = 0.1
_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


_E4M3_POS = np.array([e4m3_to_float(i) for i in range(127)], dtype=np.float64)
_E4M3_EDGE = (_E4M3_POS[:-1] + _E4M3_POS[1:]) * 0.5


def e8m0_fp8(amax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Biased E8M0 exponent and scale whose fp8 peak of 448 covers ``amax``."""
    ratio = np.maximum(amax.astype(np.float64) / 448.0, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


def e4m3(scaled: np.ndarray) -> np.ndarray:
    """Nearest E4M3 byte. 0x7F and 0xFF stay unused."""
    choice = np.searchsorted(_E4M3_EDGE, np.abs(scaled), side="left").astype(np.uint8)
    choice = np.minimum(choice, np.uint8(126))
    return np.where(np.signbit(scaled), choice | np.uint8(0x80), choice)


def e8m0(amax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Biased E8M0 exponent and scale whose fp4 peak of 6 covers ``amax``."""
    ratio = np.maximum(amax / 6.0, np.finfo(np.float64).tiny)
    mantissa, exponent = np.frexp(ratio)
    power = exponent.astype(np.int32) - (mantissa == 0.5).astype(np.int32)
    power = np.where(amax <= 0.0, np.int32(-127), power)
    biased = np.clip(power + 127, 0, 254).astype(np.uint8)
    scale = np.ldexp(np.ones(amax.shape, dtype=np.float32), biased.astype(np.int32) - 127)
    return biased, scale


def e2m1(scaled: np.ndarray) -> np.ndarray:
    """Nearest E2M1 nibble, ties to the even code. Past 6 saturates."""
    magnitude = np.abs(scaled)
    choice = np.searchsorted(_EDGE, magnitude, side="left").astype(np.uint8)
    on_edge = (choice < 7) & (magnitude == _EDGE[np.minimum(choice, 6)])
    choice = np.where(on_edge & ((choice & 1) == 1), choice + 1, choice).astype(np.uint8)
    return choice | (np.signbit(scaled).astype(np.uint8) << np.uint8(3))


def dequant(nibbles: np.ndarray) -> np.ndarray:
    values = _GRID[nibbles & 7]
    return np.where(nibbles & 8, -values, values)


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << np.uint8(4))


class Capture:
    """Sums of ``X^T X`` per (layer, stream). Layer -1 is the head."""

    def __init__(self) -> None:
        self.sums: dict[tuple[int, str], np.ndarray] = {}
        self.tokens = 0
        self.done = 0
        self.source = ""

    def add(self, layer: int, stream: str, rows: np.ndarray) -> None:
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        gram = rows.T @ rows
        held = self.sums.get((layer, stream))
        if held is None:
            self.sums[(layer, stream)] = gram
        else:
            held += gram

    def absorb(self, other: Capture) -> None:
        """Add a finished sequence. A failed sequence is discarded instead."""
        for key, gram in other.sums.items():
            held = self.sums.get(key)
            if held is None:
                self.sums[key] = gram
            else:
                held += gram
        self.tokens += other.tokens

    def save(self, path: Path) -> None:
        arrays = {f"{layer}:{stream}": gram for (layer, stream), gram in self.sums.items()}
        partial = path.with_suffix(".partial.npz")
        np.savez(
            partial,
            tokens=np.int64(self.tokens),
            done=np.int64(self.done),
            source=np.array(self.source),
            **arrays,
        )
        partial.replace(path)

    @classmethod
    def load(cls, path: Path) -> Capture:
        capture = cls()
        with np.load(path) as held:
            for key in held.files:
                if key == "tokens":
                    capture.tokens = int(held[key])
                elif key == "done":
                    capture.done = int(held[key])
                elif key == "source":
                    capture.source = str(held[key])
                else:
                    layer, stream = key.split(":")
                    capture.sums[(int(layer), stream)] = held[key]
            if "done" not in held.files and capture.tokens:
                capture.done = 2**31
        return capture

    def hessian(self, layer: int, stream: str) -> np.ndarray:
        """A query, key, or value stream adds the inputs the three streams shared."""
        names = ("shared", stream) if stream in ("q", "k", "v") else (stream,)
        parts = [self.sums.get((layer, name)) for name in names]
        found = [part for part in parts if part is not None]
        if not found:
            message = f"no calibration inputs for layer {layer} {stream}"
            raise KeyError(message)
        total = found[0].astype(np.float64)
        for part in found[1:]:
            total += part
        return total


def _inverse_factor(hessian: np.ndarray, damp: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper factor U with U^T U = (H + damp)^-1, and the dead-column mask."""
    h = np.array(hessian, dtype=np.float64)
    diagonal = np.diag(h).copy()
    dead = diagonal <= 0.0
    h[dead, dead] = 1.0
    diagonal[dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(diagonal))
    lower = np.linalg.cholesky(h)
    eye = np.eye(h.shape[0])
    inverse_lower = np.linalg.solve(lower, eye)
    inverse = inverse_lower.T @ inverse_lower
    upper = np.linalg.cholesky(inverse).T
    return np.ascontiguousarray(upper, dtype=np.float32), dead


def gptq_fp4(
    matrix: np.ndarray,
    hessian: np.ndarray,
    *,
    damp: float = 0.01,
    block: int = 128,
    chunk: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack [rows, inner] like the round-to-nearest packer, with GPTQ rounding."""
    rows, inner = matrix.shape
    if inner % block or block % 32:
        message = f"inner {inner} and block {block} must be multiples of 32"
        raise ValueError(message)
    upper, dead = _inverse_factor(hessian, damp)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales = np.empty((rows, inner // 32), dtype=np.uint8)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        work = np.array(matrix[start:stop], dtype=np.float32)
        work[:, dead] = 0.0
        nibbles = np.empty(work.shape, dtype=np.uint8)
        scale = np.ones(stop - start, dtype=np.float32)
        for b0 in range(0, inner, block):
            b1 = b0 + block
            part = work[:, b0:b1]
            errors = np.empty((stop - start, block), dtype=np.float32)
            factor = upper[b0:b1, b0:b1]
            for i in range(block):
                column = b0 + i
                if column % 32 == 0:
                    amax = np.max(np.abs(part[:, i : i + 32]), axis=1)
                    biased, scale = e8m0(amax)
                    scales[start:stop, column // 32] = biased
                code = e2m1(part[:, i] / scale)
                nibbles[:, column] = code
                error = (part[:, i] - dequant(code) * scale) / factor[i, i]
                if i + 1 < block:
                    part[:, i + 1 :] -= np.outer(error, factor[i, i + 1 :])
                errors[:, i] = error
            if b1 < inner:
                work[:, b1:] -= errors @ upper[b0:b1, b1:]
        packed[start:stop] = pack_nibbles(nibbles)
    return packed, scales


def _inverse_factor_stable(hessian: np.ndarray, damp: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper factor of ``(H + damp)^-1``. The damp grows if the Gram is indefinite."""
    base = np.array(hessian, dtype=np.float64, copy=True)
    last: BaseException | None = None
    for factor in (1.0, 10.0, 100.0):
        gram = np.array(base, copy=True)
        diagonal = np.diag(gram).copy()
        dead = diagonal <= 0.0
        gram[dead, dead] = 1.0
        diagonal[dead] = 1.0
        gram[np.diag_indices_from(gram)] += (damp * factor) * float(np.mean(diagonal))
        try:
            lower = np.linalg.cholesky(gram)
        except np.linalg.LinAlgError as exc:
            last = exc
            continue
        eye = np.eye(gram.shape[0], dtype=np.float64)
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
    beta: float = BETA,
    damp: float = 0.01,
    block: int = 128,
    chunk: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``[rows, inner]`` with the FOEM column update. ``beta`` 0 is GPTQ."""
    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    if inner % block or block % 32 or hessian.shape != (inner, inner):
        message = f"inner {inner} block {block} does not match the Gram {hessian.shape}"
        raise ValueError(message)
    upper, dead = _inverse_factor_stable(hessian, damp)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales = np.empty((rows, inner // 32), dtype=np.uint8)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        work = np.array(weight[start:stop], dtype=np.float32)
        original = np.array(work, dtype=np.float32)
        work[:, dead] = 0.0
        nibbles = np.empty(work.shape, dtype=np.uint8)
        scale = np.ones(stop - start, dtype=np.float32)
        for b0 in range(0, inner, block):
            b1 = b0 + block
            part = work[:, b0:b1]
            orig = original[:, b0:b1]
            errors = np.zeros((stop - start, block), dtype=np.float32)
            factor = upper[b0:b1, b0:b1]
            for i in range(block):
                column = b0 + i
                if dead[column]:
                    nibbles[:, column] = 0
                    if column % 32 == 0:
                        scales[start:stop, column // 32] = 0
                    continue
                if column % 32 == 0:
                    amax = np.max(np.abs(part[:, i : i + 32]), axis=1)
                    biased, scale = e8m0(amax)
                    scales[start:stop, column // 32] = biased
                code = e2m1(part[:, i] / scale)
                nibbles[:, column] = code
                diagonal = float(factor[i, i])
                if diagonal == 0.0:
                    message = f"column {column} has a zero inverse diagonal"
                    raise RuntimeError(message)
                drift = part[:, i] - orig[:, i]
                residual = part[:, i] - dequant(code) * scale
                error = (residual - np.float32(beta) * drift) / np.float32(diagonal)
                errors[:, i] = error
                if i + 1 < block:
                    coeff = np.array(factor[i, i + 1 :], dtype=np.float32, copy=True)
                    coeff[dead[column + 1 : b1]] = 0.0
                    part[:, i + 1 :] -= error[:, None] * coeff
                    if not dead[column + 1]:
                        nxt = i + 1
                        part[:, nxt] -= np.float32(beta) * (part[:, nxt] - orig[:, nxt])
            if b1 < inner:
                coeff = np.array(upper[b0:b1, b1:], dtype=np.float32, copy=True)
                coeff[:, dead[b1:]] = 0.0
                work[:, b1:] -= errors @ coeff
                if not dead[b1]:
                    work[:, b1] -= np.float32(beta) * (work[:, b1] - original[:, b1])
            if not np.isfinite(work).all():
                message = "FOEM update left a non-finite weight"
                raise RuntimeError(message)
        packed[start:stop] = pack_nibbles(nibbles)
    return packed, scales


def foem_fp4_device(
    session: Any,
    matrix: np.ndarray,
    hessian: np.ndarray,
    *,
    beta: float = BETA,
    damp: float = 0.01,
    block: int = 128,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """The same pack as ``foem_fp4``, with the column update on the device."""
    from inkling_2_mini.metal41.session import f32, u32

    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    if inner % block or block % 32 or hessian.shape != (inner, inner):
        message = f"inner {inner} block {block} does not match the Gram {hessian.shape}"
        raise ValueError(message)
    upper, dead = _inverse_factor_stable(hessian, damp)
    dead_bytes = np.ascontiguousarray(dead, dtype=np.uint8)
    packed = np.empty((rows, inner // 2), dtype=np.uint8)
    scales_out = np.empty((rows, inner // 32), dtype=np.uint8)
    threads = 256
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        count = stop - start
        work = np.array(weight[start:stop], dtype=np.float32)
        original = np.array(work, dtype=np.float32)
        work[:, dead] = 0.0
        work_buf = session.upload(work)
        original_buf = session.upload(original)
        nibble_buf = session.zeros(count * inner)
        scale_buf = session.zeros(count * (inner // 32))
        error_buf = session.zeros(count * block * 4)
        flag = session.zeros(4)
        session.begin()
        for b0 in range(0, inner, block):
            b1 = b0 + block
            factor = np.ascontiguousarray(upper[b0:b1, b0:b1])
            session.dispatch(
                "foem_block",
                [
                    work_buf,
                    original_buf,
                    session.upload(factor),
                    session.upload(dead_bytes[b0:b1]),
                    nibble_buf,
                    scale_buf,
                    error_buf,
                    flag,
                ],
                [u32(count), u32(inner), u32(b0), u32(block), f32(beta)],
                ((count + threads - 1) // threads, 1, 1),
            )
            if b1 < inner:
                tail = inner - b1
                coeff = np.array(upper[b0:b1, b1:], dtype=np.float32, copy=True)
                coeff[:, dead[b1:]] = 0.0
                pull = 0 if dead[b1] else 1
                session.dispatch(
                    "foem_tail",
                    [
                        work_buf,
                        original_buf,
                        error_buf,
                        session.upload(np.ascontiguousarray(coeff)),
                        flag,
                    ],
                    [u32(count), u32(inner), u32(b1), u32(block), u32(tail), u32(pull), f32(beta)],
                    ((count * tail + threads - 1) // threads, 1, 1),
                )
        session.end()
        if int(np.frombuffer(session.download(flag, 4), dtype=np.uint32)[0]) != 0:
            message = "FOEM update left a non-finite weight"
            raise RuntimeError(message)
        nibbles = np.frombuffer(session.download(nibble_buf, count * inner), dtype=np.uint8)
        packed[start:stop] = pack_nibbles(nibbles.reshape(count, inner))
        scales_out[start:stop] = np.frombuffer(
            session.download(scale_buf, count * (inner // 32)), dtype=np.uint8
        ).reshape(count, inner // 32)
    return packed, scales_out


def foem_fp8(
    matrix: np.ndarray,
    hessian: np.ndarray,
    *,
    beta: float = BETA,
    damp: float = 0.01,
    block: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Host E4M3 FOEM. The bake uses ``foem_fp8_device`` on the full planes."""
    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    if inner % block or block % 32 or hessian.shape != (inner, inner):
        message = f"inner {inner} block {block} does not match the Gram {hessian.shape}"
        raise ValueError(message)
    upper, dead = _inverse_factor_stable(hessian, damp)
    codes = np.empty((rows, inner), dtype=np.uint8)
    scales = np.empty((rows, inner // 32), dtype=np.uint8)
    work = np.array(weight, dtype=np.float32)
    original = np.array(work, dtype=np.float32)
    work[:, dead] = 0.0
    for b0 in range(0, inner, block):
        b1 = b0 + block
        part = work[:, b0:b1]
        orig = original[:, b0:b1]
        errors = np.zeros((rows, block), dtype=np.float32)
        factor = upper[b0:b1, b0:b1]
        scale = np.ones(rows, dtype=np.float32)
        for i in range(block):
            column = b0 + i
            if dead[column]:
                codes[:, column] = 0
                if column % 32 == 0:
                    scales[:, column // 32] = 0
                continue
            if column % 32 == 0:
                amax = np.max(np.abs(part[:, i : i + 32]), axis=1)
                biased, scale = e8m0_fp8(amax)
                scales[:, column // 32] = biased
            code = e4m3(part[:, i] / scale)
            codes[:, column] = code
            grid = _E4M3_POS[code & np.uint8(0x7F)].astype(np.float32) * scale
            restored = np.where(code & np.uint8(0x80), -grid, grid).astype(np.float32)
            diagonal = float(factor[i, i])
            if diagonal == 0.0:
                message = f"column {column} has a zero inverse diagonal"
                raise RuntimeError(message)
            drift = part[:, i] - orig[:, i]
            error = (part[:, i] - restored - np.float32(beta) * drift) / np.float32(diagonal)
            errors[:, i] = error
            if i + 1 < block:
                coeff = np.array(factor[i, i + 1 :], dtype=np.float32, copy=True)
                coeff[dead[column + 1 : b1]] = 0.0
                part[:, i + 1 :] -= error[:, None] * coeff
                if not dead[column + 1]:
                    nxt = i + 1
                    part[:, nxt] -= np.float32(beta) * (part[:, nxt] - orig[:, nxt])
        if b1 < inner:
            coeff = np.array(upper[b0:b1, b1:], dtype=np.float32, copy=True)
            coeff[:, dead[b1:]] = 0.0
            work[:, b1:] -= errors @ coeff
            if not dead[b1]:
                work[:, b1] -= np.float32(beta) * (work[:, b1] - original[:, b1])
        if not np.isfinite(work).all():
            message = "FOEM update left a non-finite weight"
            raise RuntimeError(message)
    return codes, scales


def foem_fp8_device(
    session: Any,
    matrix: np.ndarray,
    hessian: np.ndarray,
    *,
    beta: float = BETA,
    damp: float = 0.01,
    block: int = 128,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 codes with the FOEM column update. ``beta`` 0 is GPTQ."""
    from inkling_2_mini.metal41.session import f32, u32

    weight = np.ascontiguousarray(matrix, dtype=np.float32)
    rows, inner = weight.shape
    if inner % block or block % 32 or hessian.shape != (inner, inner):
        message = f"inner {inner} block {block} does not match the Gram {hessian.shape}"
        raise ValueError(message)
    upper, dead = _inverse_factor_stable(hessian, damp)
    dead_bytes = np.ascontiguousarray(dead, dtype=np.uint8)
    codes_out = np.empty((rows, inner), dtype=np.uint8)
    scales_out = np.empty((rows, inner // 32), dtype=np.uint8)
    threads = 256
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        count = stop - start
        work = np.array(weight[start:stop], dtype=np.float32)
        original = np.array(work, dtype=np.float32)
        work[:, dead] = 0.0
        work_buf = session.upload(work)
        original_buf = session.upload(original)
        code_buf = session.zeros(count * inner)
        scale_buf = session.zeros(count * (inner // 32))
        error_buf = session.zeros(count * block * 4)
        flag = session.zeros(4)
        session.begin()
        for b0 in range(0, inner, block):
            b1 = b0 + block
            factor = np.ascontiguousarray(upper[b0:b1, b0:b1])
            session.dispatch(
                "foem_block_fp8",
                [
                    work_buf,
                    original_buf,
                    session.upload(factor),
                    session.upload(dead_bytes[b0:b1]),
                    code_buf,
                    scale_buf,
                    error_buf,
                    flag,
                ],
                [u32(count), u32(inner), u32(b0), u32(block), f32(beta)],
                ((count + threads - 1) // threads, 1, 1),
            )
            if b1 < inner:
                tail = inner - b1
                coeff = np.array(upper[b0:b1, b1:], dtype=np.float32, copy=True)
                coeff[:, dead[b1:]] = 0.0
                pull = 0 if dead[b1] else 1
                session.dispatch(
                    "foem_tail",
                    [
                        work_buf,
                        original_buf,
                        error_buf,
                        session.upload(np.ascontiguousarray(coeff)),
                        flag,
                    ],
                    [u32(count), u32(inner), u32(b1), u32(block), u32(tail), u32(pull), f32(beta)],
                    ((count * tail + threads - 1) // threads, 1, 1),
                )
        session.end()
        if int(np.frombuffer(session.download(flag, 4), dtype=np.uint32)[0]) != 0:
            message = "FOEM update left a non-finite weight"
            raise RuntimeError(message)
        codes_out[start:stop] = np.frombuffer(
            session.download(code_buf, count * inner), dtype=np.uint8
        ).reshape(count, inner)
        scales_out[start:stop] = np.frombuffer(
            session.download(scale_buf, count * (inner // 32)), dtype=np.uint8
        ).reshape(count, inner // 32)
    return codes_out, scales_out


def foem_device_matches(session: Any) -> None:
    """The device pack agrees with ``foem_fp4`` on one small Gram."""
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((40, 256)).astype(np.float32)
    draw = rng.standard_normal((256, 64)).astype(np.float64)
    hessian = draw @ draw.T + np.eye(256)
    host_packed, host_scales = foem_fp4(matrix, hessian)
    device_packed, device_scales = foem_fp4_device(session, matrix, hessian)
    if np.array_equal(host_packed, device_packed) and np.array_equal(host_scales, device_scales):
        return
    differ = int(np.sum(host_packed != device_packed))
    message = f"FOEM device pack does not match the host ({differ} packed bytes)"
    raise RuntimeError(message)


def config_digest(root: Path) -> str:
    return hashlib.sha256((Path(root) / "config.json").read_bytes()).hexdigest()


def write_pack(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """One safetensors file of uint8 tensors, written beside and then renamed over."""
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for name, array in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    partial = path.with_suffix(".partial")
    with partial.open("wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        for array in tensors.values():
            handle.write(np.ascontiguousarray(array).tobytes())
    partial.replace(path)


__all__ = [
    "PACK_FORMAT",
    "PACK_NAME",
    "Capture",
    "config_digest",
    "dequant",
    "e2m1",
    "e8m0",
    "foem_device_matches",
    "e4m3",
    "e8m0_fp8",
    "foem_fp4",
    "foem_fp4_device",
    "foem_fp8_device",
    "gptq_fp4",
    "pack_nibbles",
    "write_pack",
]
