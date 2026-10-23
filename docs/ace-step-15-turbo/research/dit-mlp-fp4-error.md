# DiT MLP fp4 error

Measurement note for the `ace-step-15-turbo` pipeline. The serving contract is [../README.md](../README.md). This file records the per-layer fp4 relative error of the [ACE-Step/acestep-v15-xl-turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo) DiT MLPs. It does not record a generation-time sum.

## Remainder

Each of the 32 DiT MLP layers is scored by the RMS of its BF16 matrix against the restored fp4 matrix, divided by the source RMS. Gate, up, and down share the worst of those three scores. The six highest layers are 0, 1, 2, 3, 4, and 21, and those layers are stored as fp8. A search on 24 October 2026 did not find this table.

## Predecessors

- Method. MixQuant ([arXiv:2309.17341](https://arxiv.org/abs/2309.17341)) scores a layer by the mean squared gap between the full-precision weights and the restored quantized weights, then keeps a wider code where that gap is large. The round-up E8M0 miss bound used as a feed-forward budget is recorded for a different checkpoint in [the 27B note](../../qwen38-27b/research/ffn-fp8-budget.md).
- Model. [sjoe1244/ACE-Step-v1.5-xl-turbo-nvfp4](https://huggingface.co/sjoe1244/ACE-Step-v1.5-xl-turbo-nvfp4) stores the large linear layers as NVFP4 and leaves norms, biases, embedders, and the output head in BF16. [scragnog/Ace-Step-1.5-MXFP4-Quants](https://huggingface.co/scragnog/Ace-Step-1.5-MXFP4-Quants) keeps embeddings, the output projection, and 1D tensors in float32. [ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit](https://huggingface.co/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-4bit) keeps the timestep-embedding family at 8-bit and the remaining large linears at 4-bit.
- Measurement. [arXiv:2603.08747](https://arxiv.org/abs/2603.08747) ranks transformer components by the perplexity change of keeping that component in float16, on Qwen2.5 models. MLP up and down projections rank above the gate. That paper does not score this DiT.
- Mechanism. Every Euler step reads every DiT MLP, so a layer stored as fp8 costs its extra bytes on every step.

## Rule

The sort key is the maximum of the three relative errors in one layer. The budget is the largest whole count that does not exceed 20% of 32, which is 6. An equal score keeps the lower layer index. The other 26 MLP layers stay fp4.

## Command

`uv run favlo --ace-step-15-turbo quantize` writes `models/acestep-v15-xl-turbo-fp8-fp4/mlp_error.json` while it packs `models/acestep-v15-xl-turbo`. A later run returns without rescoring when `manifest.json` is already `complete` and `mlp_fp8_percent` is 20. The table below is that file, rounded to six digits.

## Table

`layer` is the maximum of gate, up, and down. `store` is the width kept for all three matrices.

| Layer | Gate | Up | Down | Layer | Store |
|---:|---:|---:|---:|---:|---|
| 0 | 0.127553 | 0.126373 | 0.128395 | 0.128395 | fp8 |
| 1 | 0.120712 | 0.120248 | 0.125163 | 0.125163 | fp8 |
| 2 | 0.119511 | 0.119428 | 0.123922 | 0.123922 | fp8 |
| 3 | 0.119511 | 0.119606 | 0.122922 | 0.122922 | fp8 |
| 4 | 0.119434 | 0.119399 | 0.122465 | 0.122465 | fp8 |
| 5 | 0.119757 | 0.119772 | 0.122270 | 0.122270 | fp4 |
| 6 | 0.119922 | 0.119889 | 0.120749 | 0.120749 | fp4 |
| 7 | 0.120231 | 0.119998 | 0.120211 | 0.120231 | fp4 |
| 8 | 0.119799 | 0.119539 | 0.119694 | 0.119799 | fp4 |
| 9 | 0.120357 | 0.119807 | 0.119632 | 0.120357 | fp4 |
| 10 | 0.120221 | 0.119423 | 0.118621 | 0.120221 | fp4 |
| 11 | 0.120139 | 0.118877 | 0.119033 | 0.120139 | fp4 |
| 12 | 0.120546 | 0.118926 | 0.118650 | 0.120546 | fp4 |
| 13 | 0.121033 | 0.119732 | 0.119462 | 0.121033 | fp4 |
| 14 | 0.120648 | 0.119833 | 0.119652 | 0.120648 | fp4 |
| 15 | 0.120904 | 0.119961 | 0.121092 | 0.121092 | fp4 |
| 16 | 0.121243 | 0.119909 | 0.120340 | 0.121243 | fp4 |
| 17 | 0.121652 | 0.120134 | 0.119638 | 0.121652 | fp4 |
| 18 | 0.120946 | 0.119548 | 0.119834 | 0.120946 | fp4 |
| 19 | 0.121405 | 0.119570 | 0.119466 | 0.121405 | fp4 |
| 20 | 0.120878 | 0.119406 | 0.119761 | 0.120878 | fp4 |
| 21 | 0.122307 | 0.119692 | 0.118848 | 0.122307 | fp8 |
| 22 | 0.120784 | 0.119054 | 0.118945 | 0.120784 | fp4 |
| 23 | 0.120793 | 0.118993 | 0.119142 | 0.120793 | fp4 |
| 24 | 0.120278 | 0.118676 | 0.119398 | 0.120278 | fp4 |
| 25 | 0.119938 | 0.118479 | 0.118904 | 0.119938 | fp4 |
| 26 | 0.120052 | 0.118589 | 0.118956 | 0.120052 | fp4 |
| 27 | 0.119796 | 0.118665 | 0.118936 | 0.119796 | fp4 |
| 28 | 0.119990 | 0.118848 | 0.119312 | 0.119990 | fp4 |
| 29 | 0.119728 | 0.118746 | 0.119893 | 0.119893 | fp4 |
| 30 | 0.120089 | 0.119254 | 0.121488 | 0.121488 | fp4 |
| 31 | 0.119541 | 0.118811 | 0.121679 | 0.121679 | fp4 |

## Search

24 October 2026. Queries: "ACE-Step v1.5 XL Turbo DiT MLP fp4 fp8 mixed precision relative RMS quantization error layers", then "ACE-Step XL Turbo MXFP4 NVFP4 layer sensitivity cosine similarity DiT MLP promoted layers". The predecessors in this note are the overlaps those queries returned. The search did not find the 32-layer relative-error table above.

## Corrections
