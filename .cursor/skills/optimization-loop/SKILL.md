---
name: optimization-loop
description: Run the MADMetal optimization loop for one pipeline. Establish ceilings from theoretical computations on the designed kernels, cache them in that pipeline's docs, and reuse them when the kernels are unchanged. Then modify the pipeline and run full smoke tests through the long prompt. Use when optimizing a pipeline, raising prefill or decode rate, closing a kernel gap, or when the user asks for an optimization loop. The whole process is non-HITL until 90% of the ceiling is achieved. Once the long prompt passes, commit the related code on the current branch, including main.
---

# Optimization loop

One pipeline. Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) and that pipeline's `docs/<pipeline>/` note. Edit only that write set. Leave every other pipeline untouched.

The whole process is non-HITL until 90% of the ceiling is achieved. Do not ask, and do not stop between iterations, until the measured rate meets that bar or a step cannot proceed.

## 1. Ceiling

The ceiling is a computation on the designed kernels, not a rate from `serve`. Cache it in `docs/<pipeline>/ceilings.md` and reuse it when those kernels are unchanged.

Read that file first. If it records a ceiling and the designed kernels match the kernels named there, use the saved kernel ceiling. Do not recompute it.

Recompute when the file is missing or any designed kernel changed. For the phase being optimized (prefill or decode):

- Count the bytes the designed forward reads and writes for one token, including the attention window that kernel actually walks. Divide by the streaming bandwidth measured on this machine. That is the bandwidth ceiling.
- Time each designed kernel in its own command buffer, then wait. Compose one token from those isolated times. That is the kernel ceiling.
- Do not add `GPUStartTime` / `GPUEndTime` intervals that overlap across in-flight buffers.
- Do not edit production code to take the measurement. A missing kernel stays missing; do not substitute another forward.

Write the recomputed ceilings into `docs/<pipeline>/ceilings.md` in the same change as the kernels they describe. Name the kernels, the phase, the bytes, the bandwidth, and both ceilings. Do not invent a number. No hostnames, home directories, or volume paths.

The ceiling used for the 90% test is the kernel ceiling. The bandwidth ceiling is the bound that kernel design is measured against.

## 2. Modify

Change this pipeline's kernels or the forward that launches them. Update `docs/<pipeline>/` in the same change when the kernels, records, or cache change.

Take `artifacts/pipeline-test.lock` before any command that compiles shaders, dispatches Metal, loads a checkpoint, quantizes, or starts `madmetal serve`. The lock is one JSON object: `pipeline` (the flag stem), `pid` (this shell), `command` (the short test name). Create it with exclusive create. If a live `pid` holds it, wait. Release it when the testing stretch ends, including on failure. Remove the file only when its `pid` is this agent's.

## 3. Smoke, then the long prompt

After a modification, run the full smoke list, then the long prompt. Stop the iteration on the first failure, fix it, and rerun from that step. Do not start the long prompt while a smoke is failing.

1. `uv run pytest` on the tests that import only this package.
2. The smoke commands this pipeline's `--help` lists, in this order when present: `probe`, `matmul`, `kernels`.

```text
uv run madmetal --<pipeline> probe
uv run madmetal --<pipeline> matmul
uv run madmetal --<pipeline> kernels
```

3. After those pass, the long prompt. It starts its own server, stops it after the first turn, and starts another so the second turn loads the prefix from disk. Do not point it at a listener already on `8129`.

```text
uv run python -m harness.long_prompt --model <pipeline>
```

The default target is 65536 tokens. Pass `--tokens` when this pipeline's prompt limit is different. The command prints prefill and decode rates. Those printed rates are the measurement.

## 4. Commit

Once the long prompt passes, commit the related code on the current branch, including `main`.

Related code is this iteration's write set: the pipeline entry, package, `docs/<pipeline>/`, and tests that import only this package. When the printed rate is at least 90% of the kernel ceiling, record that rate in the pipeline note before the commit.

Leave models, artifacts, the machine lock, and other pipelines out of the commit. Use a Conventional Commit. Do not push.

A failed smoke or a failed long prompt is not a commit.

## 5. Compare

A phase is done when its printed long-prompt rate is at least 90% of the kernel ceiling for that phase.

If it is short, modify again and run the smokes through the long prompt. Reuse the saved ceiling unless a designed kernel changed. Each long prompt is a new run. Commit again when that run passes.

When 90% is met, report the ceiling and the printed rate. Do not write that rate into another pipeline's note.
