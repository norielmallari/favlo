# Recorded forward

Measurement note for the `gemma4_26b_a4b` pipeline. The serving contract is [../README.md](../README.md). The request rows are in [../ceilings.md](../ceilings.md). This file records those runs. The isolated dispatch sum is [kernel-ceilings.md](kernel-ceilings.md).

## Remainder

On an M5 Pro with 48 GB, a varied prompt prefilled 1,024 tokens at 1,161.9 tok/s. The carrot smoke decoded at 46.64, 44.03, and 41.97 tok/s. The coding-agent runtime log decoded 820 tokens at 18.84 tok/s. A search on 31 October 2026 did not find that run.

## Predecessors

- Method. The forward is the shader library summed in [kernel-ceilings.md](kernel-ceilings.md). Overlap, the prefix store, and the expert repack are inside these times. The kernel sum leaves them out.
- Model. [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). [oMLX n50yftgp](https://omlx.ai/benchmarks/performance/n50yftgp) reports 1,289 prefill tok/s and 72.5 decode tok/s at 1,024 tokens for a 4-bit MLX copy on an M5 Pro with 48 GB, oMLX 0.6.0rc1. [oMLX gv7qn8l9](https://omlx.ai/benchmarks/performance/gv7qn8l9) reports 2,082 and 76.9 at the same context on the same chip, oMLX 0.6.2.
- Measurement. [ComputeArena](https://computearena.ai/models/gemma-4-26b-a4b-it) lists other prompt and generation rates for this checkpoint, including 80.3 generation tok/s and 1,669 prompt tok/s at 512 tokens on an M4 Max. [Rapid-MLX](https://rapidmlx.com/models/gemma-4-26b-4bit) reports 78.5 decode tok/s on an M4 Pro with 48 GB. They do not publish this harness run.
- Mechanism. The carrot cases are greedy, the instruct card, and the thinking card. The coding agent is the thinking card with tools. This record has no drafter.

## Rule

Prefill is the uncached forward of the named prompt. Decode is the sampled-token time after that forward. The varied-prompt rows are one chat encoding of a repeated paragraph, not a run of identical tokens. The carrot rows are the lines `tests/test_gemma4_26b_a4b_smoke.py` printed on 31 October 2026. The coding-agent row is the runtime log copied into the ceilings note. The harness summary for that run printed `no requests`.

## Command

```text
uv run python -m pytest tests/test_gemma4_26b_a4b_smoke.py -s
uv run python -m harness.coding --model gemma4-26b-a4b
```

The pytest command prints the carrot rows. The coding command prints `passed` when the hidden suite passes. No committed command reprints the 1,134.1 and 1,161.9 tok/s rows or the 18.84 tok/s sum.

## Table

| Run | Prefill | Decode |
|---|---|---|
| Varied prompt, 512 tokens | 1,134.1 tok/s | |
| Varied prompt, 1,024 tokens | 1,161.9 tok/s | |
| Carrot, greedy | 262.88 tok/s (50 tok in 0.19 s) | 46.64 tok/s (54 tok in 1.16 s) |
| Carrot, instruct card | 272.79 tok/s (50 tok in 0.18 s) | 44.03 tok/s (77 tok in 1.75 s) |
| Carrot, thinking card | 283.09 tok/s (53 tok in 0.19 s) | 41.97 tok/s (200 tok in 4.77 s) |
| Coding agent | | 18.84 tok/s (820 tok in 43.53 s) |

The coding-agent requests ran from 20.36 tok/s on the first to 16.84 tok/s on the last.

## Search

31 October 2026. Queries: "Gemma 4 26B-A4B oMLX prefill tok/s 1289 M5 Pro 48GB 2026", "gemma-4-26B-A4B 1161.9 OR 18.84 OR 46.64 tok/s Metal decode", and "gemma-4-26b-a4b-it-4bit PP tok/s TG tok/s M5 Pro 20c 1161 OR 18.84 OR 46.64 OR 56.658". The predecessors in this note are the overlaps those queries returned. The search did not find 1,161.9, 46.64, or 18.84 tok/s.

## Corrections

Later finds are appended here. The sections above stay as committed.
