# int2 reconstruction miss

Measurement note for the `qwen38-flash-next` pipeline. The serving contract is [../architecture.md](../architecture.md). The element codes and the scale rule are in [../record.md](../record.md). This file records a format bound of the code points. It does not record the per-matrix errors of the Qwen3.8-Flash-Next checkout.

## Remainder

The round-up E8M0 scale in `qwen38_flash_next/formats.py` keeps each block peak inside the int2 range. Under that rule, a block whose scaled peak equals 1 misses by 1/2 of the peak, and a peak that has just rounded up misses by the whole peak. A search on 21 October 2026 did not find that pair of bounds for a two-bit grid whose positive endpoint is 1.

## Predecessors

- Method. The OCP MX specification stores one E8M0 scale for a block of 32. AMD aiter `RoundUp` and torchao `RCEIL` set that scale to the power of two at or above `amax / max_pos`. [Decomposing MXFP4 quantization error](https://arxiv.org/html/2605.20402) applies that scale to the E2M1 grid, whose positive endpoint is 6, and bounds the deadzone at 1/24 of an exact-scale peak.
- Format. The same OCP specification's integer element is MXINT8, with codes through 127. It does not define a two-bit element.
- Mechanism. Metal 4.1 `unpack` does not accept `int2b` on this SDK, so the shader shifts the stored pairs. The shift order and the signed endpoints are in [../record.md](../record.md).

## Rule

Positive codes are 0 and 1. `qmax` is 1. The scale rounds up, so a block peak becomes a scaled peak in `(1/2, 1]`. The miss is half the gap between the two codes that bracket the worst value in `[0, peak]`, divided by the peak. The only positive gap is 0 to 1, so that half-gap is 1/2. The table is that ratio at a scaled peak of 1, and the same ratio as the scaled peak approaches 1/2 from above.

A value whose scaled magnitude is below 1/2 restores as 0.

## Command

```text
uv run python -c "from qwen38_flash_next.formats import QMAX, quantize_int2_rows
import numpy as np
qmax = int(QMAX[2])
half = 1 / 2
print(half / qmax)
print(half / (qmax / 2))
src = np.zeros((1, 32), dtype=np.float32)
src[0, 0] = float(qmax)
src[0, 1] = half
packed, biased = quantize_int2_rows(src)
scale = float(np.ldexp(np.float32(1.0), int(np.asarray(biased).reshape(-1)[0]) - 127))
code = int(np.asarray(packed).reshape(-1)[0]) & 0x3
if code & 2:
    code -= 4
print(abs(code * scale - half), scale)"
```

The first line is the miss at a scaled peak of 1. The second is the miss as the scaled peak approaches 1/2. The third line is the quantizer's error on the midpoint of a block whose peak is 1, and that block's scale.

## Table

The miss is the largest absolute reconstruction error inside one block, divided by that block's peak.

| Scale landing | int2 miss / peak | Restores as 0 below |
|---|---:|---:|
| Scaled peak equals `qmax` (1) | 1/2 | 1/2 of the peak |
| Widest legal miss | 1 | the peak |

The widest row is the limit. A scaled peak of 1/2 is not itself legal, because the scale has just rounded up and the peak sits above 1/2. The midpoint 1/2 stays inside that block, so the ratio approaches 1.

The command prints `0.5`, `1.0`, and `0.5 1.0`.

## Search

21 October 2026. Queries: "int2 quantization signed codes -2 -1 0 1 E8M0 round up reconstruction error block of 32", "MXINT2 E8M0 qmax 1 deadzone round-up scale reconstruction error block 32". The predecessors in this note are the overlaps those queries returned. The search did not find the 1/2 and 1 bounds above for this grid.

## Corrections

Later finds are appended here. The sections above stay as committed.
