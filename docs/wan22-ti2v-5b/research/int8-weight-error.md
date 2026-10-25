# Int8 weight error

Measurement note for the `wan22-ti2v-5b` pipeline. The serving contract is [../README.md](../README.md). This file records the worst relative RMS of this quantizer. It is a measurement of the checkout.

## Remainder

The worst int8 relative RMS is 0.014504 on `blocks.11.ffn.2`. A search on 26 October 2026 did not find that score.

## Predecessors

- Method. Symmetric per-channel int8 stores one scale per output row. [unsloth/Wan2.2-TI2V-5B-FP8](https://huggingface.co/unsloth/Wan2.2-TI2V-5B-FP8) uses torchao per-channel int8 weights with per-token activations and reports a same-seed first-frame LPIPS of 0.46. [Rin247/Wan2.2-TI2V-5B-INT8](https://huggingface.co/Rin247/Wan2.2-TI2V-5B-INT8) uses torchao `int8_weight_only`. [backpack-run/Wan2.2-TI2V-5B-Backpack-Video](https://huggingface.co/backpack-run/Wan2.2-TI2V-5B-Backpack-Video) uses `Int8WeightOnlyConfig`.
- Task. [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
- Measurement. Those sources publish LPIPS or a format name. They do not publish this relative RMS.
- Mechanism. Each code is the nearest integer in −127…127. The scale is the row peak divided by 127. Half a step is that peak divided by 254.

## Rule

Relative RMS is the RMS of the residual divided by the RMS of the source matrix, scored in float32 after the int8 codes are multiplied by the row scale. A zero source scores 0. The half-step bound is 1/254 of the row peak. The relative RMS uses the matrix RMS as the denominator, so a matrix whose peak is set by a few weights can score above that bound.

## Command

`uv run favlo --wan22-ti2v-5b quantize --weights <checkpoint>` packs each DiT linear and prints `worst` with the matrix name. The line in the serving note is that print.

## Table

| Matrix | Relative RMS |
|---|---|
| `blocks.11.ffn.2` | 0.014504 |

## Search

26 October 2026. Queries: "Wan2.2-TI2V-5B int8 relative RMS blocks.11.ffn.2 0.014504" and "Wan2.2-TI2V-5B-INT8 per-channel relative RMS ffn.2 torchao int8_weight_only". The returned notes are the predecessors above. The score was not found in these sources on this date.

## Corrections

Later finds are appended here. The sections above stay as committed.
