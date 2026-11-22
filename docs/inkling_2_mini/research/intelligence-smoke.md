# Intelligence smoke

Measurement note for the `inkling_2_mini` pipeline. The serving contract is [../ARCHITECTURE.md](../ARCHITECTURE.md). The harness scores live in [../benchmarks.md](../benchmarks.md). This file records one intelligence smoke on `half`. It is a recorded run, not a format bound.

## Remainder

On an M5 Pro with 48 GB, `Inkling-2-Mini` with the GPTQ dense fp4 pack, `half` keys and values, MTP off, and thinking off scored the first 30 rows of each set.

- IFEval prompt strict was 0.867. Prompt loose was 0.833. Instruction strict was 0.911. Instruction loose was 0.889.
- GSM8K was 24/30, 0.800.
- HumanEval was 29/30, 0.967.

A search on 22 November 2026 did not find that table.

## Predecessors

- Method. [IFEval](https://arxiv.org/abs/2311.07911) scores verifiable instructions with a strict checker and a loose checker. [GSM8K](https://arxiv.org/abs/2110.14168) is grade-school math. [HumanEval](https://arxiv.org/abs/2107.03374) is 164 Python problems, scored by executing the tests.
- Model. [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), Apache-2.0. The [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/) reports SWE-bench Verified (bash only) 79.2% and DeepSWE v1.1 48.2%. It does not report IFEval, GSM8K, or HumanEval. Those scores are not this smoke.
- Score. [An independent measurement of the original weights](https://github.com/ShinMegamiBoson/lookatmyquant) reports GSM8K 193/200 (96.5%) on the first 200 test rows, and HumanEval strict 152/164 (92.7%) on all 164 problems. Both are greedy, with `reasoning_effort` `none` and a 1,536-token cap.
- Score. [Another independent measurement of the original weights](https://huggingface.co/nokeepai/Inkling-2-Mini-Unbound) reports IFEval 90.6 on all 541 prompts, GSM8K 96.6 on a 500-row subset at effort 0.9, and HumanEval+ 90.2 on all 164 problems.
- Score. None of them is the first 30 rows of this harness, at temperature 0, with thinking off.

## Rule

- MTP is off. The registry row omits `mtp`, so the request does not draft.
- Thinking is off. Temperature is 0. The presence penalty is 0.
- The completion caps are 2048 tokens for IFEval and 1024 tokens for GSM8K and HumanEval.
- The smoke is the first 30 rows of each set, in file order.
- IFEval sends the dataset prompt. The published strict and loose checkers score `message.content`.
- GSM8K sends five training shots, then `Question` / `Answer`. A reply counts when its first `####` number matches.
- HumanEval sends the instruct prompt. The checker tests the continuation after the published signature.
- The loaded line printed `kv half` and the GPTQ dense pack. A run on the nearest-code planes is a different row.

## Command

```text
uv run python -m harness.benchmarks --model inkling_2_mini --limit 30
```

The same command writes the smoke lines in [../benchmarks.md](../benchmarks.md).

## Table

The served folder is `Inkling-2-Mini`. The fused attention input plane, the attention output projection, and the head are Metal fp4 from `dense-fp4.safetensors`, rounded with GPTQ. Experts in layers 0–19 are NVFP4 requantized to Metal fp4 at open. Experts in layers 20–23 are MXFP8. Keys and values are `half`.

| Suite | n | Score |
|---|---:|---|
| IFEval | 30 | prompt strict 0.867, prompt loose 0.833, instruction strict 0.911, instruction loose 0.889 |
| GSM8K | 30 | 24/30, 0.800 |
| HumanEval | 30 | 29/30, 0.967 |

## Search

22 November 2026. Queries: "Inkling-2-Mini IFEval GSM8K HumanEval score thinkingmachines model card", and `"0.867" OR "24/30" OR "29/30" Inkling-2-Mini IFEval GSM8K HumanEval`. The predecessors in this note are the overlaps those queries returned. The model card's evaluation table does not name IFEval, GSM8K, or HumanEval. The search did not find prompt strict 0.867, GSM8K 24/30, or HumanEval 29/30 on the first 30 rows.

## Corrections

Later finds are appended here. The sections above stay as committed.
