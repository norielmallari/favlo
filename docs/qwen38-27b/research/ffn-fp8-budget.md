# Feed-forward fp8 budget

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). This file records a format bound and the budget rule that spends fp8 on the worst feed-forward layers. It does not record the per-matrix errors of the [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) checkout.

## Remainder

The round-up E8M0 scale in `qwen38_27b/formats.py` keeps each block peak inside the element range. Under that rule, the widest fp4 miss on a block is 1/5 of the peak, and the widest fp8 E4M3 miss is 1/17 of the peak. Replacing fp4 with fp8 on a block shrinks that worst miss by 17/5. A search on 21 October 2026 did not find this bound used as a feed-forward budget for this checkpoint.

## Predecessors

- Method. MixQuant ([arXiv:2309.17341](https://arxiv.org/abs/2309.17341)) scores each layer by the mean squared gap between the full-precision weights and the restored quantized weights, then picks a bit width whose gap stays within a multiple of the 8-bit gap.
- Model. [mlx-community/Qwen3.8-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-OptiQ-4bit) keeps a tensor at 8-bit when a KL pass against a uniform 4-bit reference says the layer is sensitive. [l2dy/Qwen3.8-27B-MXFP4-mlx](https://huggingface.co/l2dy/Qwen3.8-27B-MXFP4-mlx) stores feed-forward layers 56 through 63 as MXFP8 and layers 0 through 55 as MXFP4, by a fixed recipe.
- Measurement. The element grids and the block of 32 are the OCP MX specification, as [REFERENCES.md](../../../REFERENCES.md) records them. fp4 magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`. fp8 E4M3FN ends at 448.
- Mechanism. Every language feed-forward layer is read on every token, so a wider matrix costs its extra bytes on every decode step.

## Rule

The sort key, once the checkout is scored, is the relative error of one feed-forward matrix: the root-mean-square of the BF16 row against the restored fp4 row, divided by the root-mean-square of the BF16 row. Language layers only. The MTP feed-forward stays fp4, because serial decode does not run it.

The budget example is the largest whole count of language layers that stays within 10% of 64, which is 6. Gate and up share `ffn_gate_up`, so a selected layer moves all three matrices. One fp4 feed-forward image is 47,349,760 bytes. Three images are 142,049,280 bytes. fp8 stores one byte per weight instead of a nibble, and the scale plane is unchanged, so a layer grows by 133,693,440 bytes. Six layers add 802,160,640 bytes to the record and to every decode read.

The format bound below is what that promotion buys on a block. It is not a substitute for the per-matrix sort. A flat list of relative errors does not earn the wider code.

## Command

The scale is `biased_e8m0` with the power of two rounded up, so a block peak `amax` becomes a scaled peak in `(qmax/2, qmax]`. `qmax` is 6 for fp4 and 448 for fp8. For each legal scaled peak, the miss is half the gap between the two codes that bracket the worst value in `[0, peak]`, divided by the peak. The table is the maximum of that ratio, and the same ratio when the scaled peak equals `qmax`.

## Table

The miss is the largest absolute reconstruction error inside one block, divided by that block's peak.

| Scale landing | fp4 miss / peak | fp8 miss / peak | fp4 / fp8 |
|---|---:|---:|---:|
| Scaled peak equals `qmax` | 1/6 | 1/28 | 14/3 |
| Widest legal miss | 1/5 | 1/17 | 17/5 |

The widest fp4 miss is a scaled peak of 5, halfway between the codes 4 and 6. The widest fp8 miss is a scaled peak of 272, halfway between the E4M3 codes 256 and 288. Those peaks are legal: each is `qmax` times a factor in `(1/2, 1]`.

A scaled fp4 magnitude of 0.25 or less restores as 0. That is 1/24 of an exact-scale peak and approaches 1/12 of a peak whose scale has just rounded up. A scaled fp8 magnitude below half the smallest subnormal, 1/1024, restores as 0. That is 1/458752 of an exact-scale peak and approaches 1/229376 when the scale has just rounded up.

## Search

21 October 2026. Queries: "mixed precision quantization rank layers by weight reconstruction error", "FP4 FP8 mixed precision feed-forward relative quantization error", "Qwen3.8-27B mixed precision FP4 FP8 Apple Silicon". The predecessors in this note are the overlaps those queries returned. The search did not find a per-matrix BF16-to-fp4 relative-error table for this checkpoint, and it did not find the round-up-scale miss bounds above used as an fp8 layer budget.

## Corrections

Quantize accepts `--ffn-fp8`. A mixed percentage writes `models/Qwen3.8-27B-fp8-e<P>` and records the chosen layers in that directory. The sections above stay as committed.

22 October 2026. The packed record keeps 12 language feed-forward layers at fp8, the largest whole count within 20% of 64. The harness sample of 100 on that date is in [../benchmarks.md](../benchmarks.md): IFEval prompt-strict 0.740, GSM8K 87/100, and HumanEval 98/100.

23 October 2026. A later sample of 100 on `Qwen3.8-27B-fp8-e20` is in [../benchmarks.md](../benchmarks.md): IFEval prompt-strict 0.750, GSM8K 93/100, and HumanEval 94/100.

31 October 2026. `quantize` stores MTP matrices as fp8. The language-layer ranking above does not include them. One-dimensional MTP parameters stay float16.
