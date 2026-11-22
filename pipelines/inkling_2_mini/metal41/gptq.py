"""Error-compensating fp4 rounding of the dense planes (GPTQ).

Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers", 2022, Algorithm 1. Columns are rounded in order.
Each rounding error is spread over the columns not yet rounded, weighted by
the inverse Hessian ``2 X^T X`` of that plane's calibration inputs. The format
is the one the kernels read: E2M1 codes with one E8M0 scale per 32, the power
of two rounded up so the group peak fits when the group starts.

A calibration run sums ``X^T X`` per input stream on the host. The packed
planes go to one safetensors file beside the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PACK_NAME = "dense-fp4.safetensors"
PACK_FORMAT = "favlo-fp4-gptq-1"
_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_EDGE = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


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

    def add(self, layer: int, stream: str, rows: np.ndarray) -> None:
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        gram = rows.T @ rows
        held = self.sums.get((layer, stream))
        if held is None:
            self.sums[(layer, stream)] = gram
        else:
            held += gram

    def save(self, path: Path) -> None:
        arrays = {f"{layer}:{stream}": gram for (layer, stream), gram in self.sums.items()}
        partial = path.with_suffix(".partial.npz")
        np.savez(partial, tokens=np.int64(self.tokens), **arrays)
        partial.replace(path)

    @classmethod
    def load(cls, path: Path) -> Capture:
        capture = cls()
        with np.load(path) as held:
            for key in held.files:
                if key == "tokens":
                    capture.tokens = int(held[key])
                    continue
                layer, stream = key.split(":")
                capture.sums[(int(layer), stream)] = held[key]
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
    "gptq_fp4",
    "pack_nibbles",
    "write_pack",
]
