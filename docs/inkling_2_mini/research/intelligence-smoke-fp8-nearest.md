# Intelligence smoke, nearest fp8 attention

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). The harness scores live in [../benchmarks.md](../benchmarks.md). This file records one intelligence smoke on nearest E4M3 attention with GPTQ-FOEM Metal fp4 experts and `half` keys and values. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Inkling-2-Mini` with nearest E4M3 attention and output projections, a float16 head, GPTQ-FOEM Metal fp4 experts in layers 0–19, MXFP8 experts in layers 20–23, `half` keys and values, MTP off, and thinking off scored the first 30 rows of each set.

- IFEval prompt strict was 0.833. Prompt loose was 0.800. Instruction strict was 0.889. Instruction loose was 0.867.
- GSM8K was 26/30, 0.867.
- HumanEval was 30/30, 1.000.

A search on 23 November 2026 did not find that table.

## Predecessors

- Method. [IFEval](https://arxiv.org/abs/2311.07911) scores verifiable instructions with a strict checker and a loose checker. [GSM8K](https://arxiv.org/abs/2110.14168) is grade-school math. [HumanEval](https://arxiv.org/abs/2107.03374) is 164 Python problems, scored by executing the tests.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and DeepSWE v1.1 48.2%. It does not report IFEval, GSM8K, or HumanEval. Those scores are not this smoke.
- Score. [An independent measurement of the original weights](https://github.com/ShinMegamiBoson/lookatmyquant) reports GSM8K 193/200 (96.5%) on the first 200 test rows, and HumanEval strict 152/164 (92.7%) on all 164 problems. Both are greedy, with `reasoning_effort` `none` and a 1,536-token cap.
- Score. [A GB10 measurement](https://carouter.ai/blog/thinking-machines-inkling-2-mini-model-benchmark-speed-and-quality) reports IFEval 498/541 at the default high effort.
- Score. The same harness on the GPTQ dense fp4 pack recorded IFEval prompt strict 0.867, GSM8K 24/30, and HumanEval 29/30 on 22 November 2026. That row is in [intelligence-smoke.md](intelligence-smoke.md).
- Score. None of them is the first 30 rows of this harness, at temperature 0, with thinking off, on this pack.

## Rule

- MTP is off. The registry row omits `mtp`, so the request does not draft.
- Thinking is off. Temperature is 0. The presence penalty is 0.
- The completion caps are 2048 tokens for IFEval and 1024 tokens for GSM8K and HumanEval.
- The smoke is the first 30 rows of each set, in file order.
- IFEval sends the dataset prompt. The published strict and loose checkers score `message.content`.
- GSM8K sends five training shots, then `Question` / `Answer`. A reply counts when its first `####` number matches.
- HumanEval sends the instruct prompt. The checker tests the continuation after the published signature.
- Attention and the output projection are nearest E4M3. The head stayed float16. Experts in layers 0–19 are Metal fp4 from GPTQ-FOEM. Experts in layers 20–23 stay MXFP8. Keys and values are `half`.

## Command

```text
uv run python -m harness.benchmarks --model inkling_2_mini --limit 30
```

The same command writes the smoke lines in [../benchmarks.md](../benchmarks.md).

## Table

| Suite | n | Score |
|---|---:|---|
| IFEval | 30 | prompt strict 0.833, prompt loose 0.800, instruction strict 0.889, instruction loose 0.867 |
| GSM8K | 30 | 26/30, 0.867 |
| HumanEval | 30 | 30/30, 1.000 |

## Search

23 November 2026. Queries: "Inkling-2-Mini IFEval prompt strict 0.833 GSM8K 26/30 HumanEval 30/30 first 30 rows 2026". The predecessors in this note are the overlaps that query returned. The search did not find prompt strict 0.833, GSM8K 26/30, and HumanEval 30/30 on the first 30 rows of this harness.

## Corrections

Later finds are appended here. The sections above stay as committed.
