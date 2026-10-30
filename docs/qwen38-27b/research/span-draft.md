# Two-token draft

Measurement note for the `qwen38-27b` pipeline. The serving contract is [../README.md](../README.md). This file records the paired run that rejected a kept two-token state. A greedy request with `mtp` true verifies drafts from the parent DeltaNet state instead.

## Remainder

On one prompt, a two-token projection matched the serial argmax for each of the first six pairs, and the Gated DeltaNet input projection differed by at most 2e-6. Keeping the span state changed the completion by the tenth token. A search on 21 October 2026 did not find that paired count for this checkpoint.

## Predecessors

- Method. DeepSeek-V3 ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) predicts a second token with one sequential MTP block and reports a second-token acceptance of 85% to 90% on that model. Leviathan, Kalman, and Matias ([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)) verify a draft and keep the longest prefix the target would have emitted. Gloeckle et al. ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)) predict extra tokens from parallel heads. DeepSeek keeps the causal chain instead.
- Model. The language stack is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The gated delta step is Yang, Kautz, and Hatamizadeh ([arXiv:2412.06464](https://arxiv.org/abs/2412.06464)).
- Measurement. TreeWY ([arXiv:2608.20961](https://arxiv.org/abs/2608.20961)) shows that verifying a draft advances the Gated DeltaNet state through every drafted token before acceptance is known, and that the state cannot be rolled back by moving a pointer. GDN tree-scan ([arXiv:2609.23900](https://arxiv.org/abs/2609.23900)) requires each draft row to be computed from its parent state so the accepted chain matches serial decode.
- Mechanism. A span of two reads each fp4 weight once. The recurrent state is updated for both tokens. Attention keys and values can drop a rejected token by position. The DeltaNet state cannot.

## Rule

Prefill still fills the MTP plane. Decode does not keep the two-token state. A greedy request with `mtp` true emits the accepted prefix after restoring the parent state. `mtp` false stays the serial argmax. The span kernels remain in the shader library. `kernels` checks a synthetic two-row fp4 dot. That check is not the completion comparison.

## Command

`favlo --qwen38-27b kernels` dispatches `gemv_fp4_span2` on one random 8×256 tile and compares both rows with the dequantized product at absolute tolerance 1e-4. That is `check_fp4_span2` in `qwen38_27b/checks.py`.

The completion rows below are the recorded paired run in the pipeline note and in [../ceilings.md](../ceilings.md). No committed command regenerates them.

## Table

| Comparison | Recorded result |
|---|---|
| Argmax, first six pairs | Each pair matched two serial steps |
| Gated DeltaNet input projection | Differed by at most 2e-6 |
| Kept span state | Completion changed by the tenth token |
| `gemv_fp4_span2` beside `gemv_fp4_x4`, 17408×5120 | 0.50 ms beside 0.85 ms |

## Search

21 October 2026. Queries: "DeepSeek-V3 multi-token prediction speculative decoding draft acceptance recurrent state drift", and "two-token batched GEMV speculative draft recurrent state diverges serial decode Gated DeltaNet". The predecessors in this note are the overlaps those queries returned. The search did not find the six-pair, 2e-6, and tenth-token counts for this checkpoint.

## Corrections

Later finds are appended here. The sections above stay as committed.

30 October 2026. Decode verify now runs the span kernels for the first two feeds. A full accept keeps that state. A rejected tail still restores the snapshot and replays the accepted prefix. The tenth-token change in the rule above was a rejected token left in the DeltaNet state.
