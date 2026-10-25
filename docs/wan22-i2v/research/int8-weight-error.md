# Int8 weight error

Measurement note for the `wan22-i2v` pipeline. The serving contract is [../README.md](../README.md).

## Remainder

The worst int8 relative RMS against BF16 is 0.017063 on high-noise `blocks.0.self_attn.v.weight`, 0.013406 on the low-noise copy, and 0.016314 on umT5 `blocks.19.ffn.fc2.weight`. A search on 25 October 2026 did not find that table.

## Predecessors

- Method. Symmetric per-channel int8 stores one scale per output channel. [shraey/umt5-xxl-mlx](https://huggingface.co/shraey/umt5-xxl-mlx) uses that scheme for umT5 and reports a worst cosine of 0.998631. [unsloth/Wan2.2-I2V-A14B-FP8](https://huggingface.co/unsloth/Wan2.2-I2V-A14B-FP8) uses torchao per-channel int8 weights and reports a same-seed first-frame LPIPS of 0.15. [berryber09/Wan2.2-I2V-14B-INT8-W8A8](https://huggingface.co/berryber09/Wan2.2-I2V-14B-INT8-W8A8) uses learned tensorwise scales and calls the result near-lossless. [LightX2V](https://huggingface.co/lightx2v/Wan2.2-Distill-Models/tree/main) publishes int8 4-step expert files.
- Task. [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B). [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).
- Measurement. Those sources publish cosine, LPIPS, or a qualitative label. They do not publish this relative-RMS table.
- Mechanism. Each code is the nearest integer in −127…127. The scale is the channel peak divided by 127. Half a step is that peak divided by 254.

## Rule

Relative RMS is the RMS of the residual divided by the RMS of the BF16 matrix, scored in float32 after the int8 codes are multiplied by the channel scale. A zero source scores 0. The half-step bound is 1/254 of the channel peak. The relative RMS uses the matrix RMS as the denominator, so a matrix whose peak is set by a few weights can score above that bound.

## Command

`uv run favlo --wan22-i2v quantize --int8` packs each DiT linear and prints `int8 worst` with the matrix name. A matrix already stored as int8 is skipped, so a second run of the same directory does not reprint that table. `uv run favlo --wan22-i2v quantize-text` packs umT5 and prints `umt5 int8 worst` with the matrix name.

## Table

| Pack | Matrix | Relative RMS |
|---|---|---|
| high-noise expert | `blocks.0.self_attn.v.weight` | 0.017063 |
| low-noise expert | `blocks.0.self_attn.v.weight` | 0.013406 |
| umT5 | `blocks.19.ffn.fc2.weight` | 0.016314 |

## Search

25 October 2026. Queries: "Wan2.2-I2V-A14B int8 relative RMS self_attn.v.weight 0.017063" and "umt5-xxl int8 blocks.19.ffn.fc2.weight relative RMS per-channel amax 254". The returned notes are the predecessors above, plus fp8 umT5 checkpoints that list `blocks.19.ffn.fc2.weight` without a relative RMS. The table was not found in these sources on this date.

## Corrections

Later finds are appended here. The sections above stay as committed.
