---
name: optimization-loop
description: Optimize one Favlo language-model pipeline toward its prefill or decode ceiling. Use when raising tok/s, closing a prefill or decode gap, or when the user asks for an optimization loop on a text model. Do not use for image generation time or music prompting.
---

# Language-model optimization

One language-model pipeline. An image pipeline uses [optimization-image](../optimization-image/SKILL.md). Music generation time uses [optimization-music](../optimization-music/SKILL.md). A music prompt uses [music-prompting](../music-prompting/SKILL.md). Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) and that pipeline's `docs/<pipeline>/` note. Edit only that write set. Leave every other pipeline untouched.

The whole process is non-HITL until 90% of the ceiling is achieved. Do not ask, and do not stop between iterations, until the measured rate meets that bar or a step cannot proceed.

## 1. Ceiling

The ceiling is a computation on the designed kernels, not a rate from `serve`. Cache it in `docs/<pipeline>/ceilings.md` and reuse it when those kernels are unchanged.

Read that file first. If it records a ceiling and the designed kernels match the kernels named there, use the saved kernel ceiling. Do not recompute it.

Recompute when the file is missing or any designed kernel changed. For the phase being optimized (prefill or decode):

- Count the bytes the designed forward reads and writes for one token, including the attention window that kernel actually walks. Divide by the streaming bandwidth measured on an M5 Pro with 48 GB. That is the bandwidth ceiling.
- Time each designed kernel in its own command buffer, then wait. Compose one token from those isolated times. That is the kernel ceiling.
- Do not add `GPUStartTime` / `GPUEndTime` intervals that overlap across in-flight buffers.
- Do not edit production code to take the measurement. A missing kernel stays missing; do not substitute another forward.

Write the recomputed ceilings into `docs/<pipeline>/ceilings.md` in the same change as the kernels they describe. Name the kernels, the phase, the bytes, the bandwidth, and both ceilings. Do not invent a number. No hostnames, home directories, or volume paths.

The ceiling used for the 90% test is the kernel ceiling. The bandwidth ceiling is the bound that kernel design is measured against.

## 2. Modify

Change this pipeline's kernels or the forward that launches them. Update `docs/<pipeline>/` in the same change when the kernels, records, or cache change.

Take the GPU with `uv run python -m favlo.smoke_lock acquire --pipeline <stem> --command <name>` before any command that compiles shaders, dispatches Metal, loads a checkpoint, quantizes, or starts `favlo serve`. Run acquire, the command, and `uv run python -m favlo.smoke_lock release` in the same shell. Exit 2 or 3 means wait and run acquire again. Do not kill a serve the user left running.

## 3. Smoke

After a modification, run this package's checks, then load [smoke-testing](../smoke-testing/SKILL.md) and follow it. Stop the iteration on the first failure, fix it, and rerun from that step. Do not start agent smoke while a pipeline smoke is failing.

1. `uv run pytest` on the tests that import only this package.
2. The smoke commands this pipeline's `--help` lists, in this order when present: `probe`, `matmul`, `kernels`.

```text
uv run favlo --<pipeline> probe
uv run favlo --<pipeline> matmul
uv run favlo --<pipeline> kernels
```

3. After those pass, the smoke-testing skill: pipeline smoke, then the file agent, then the coding agent.

## 4. Commit

Once the agent smokes pass, commit the related code on the current branch, including `main`.

Related code is this iteration's write set: the pipeline entry, package, `docs/<pipeline>/`, and tests that import only this package. When a printed series for the phase being optimized is at least 90% of the kernel ceiling, record that rate in the pipeline note before the commit.

Leave models, artifacts, the machine lock, and other pipelines out of the commit. Use a Conventional Commit. Do not push.

A failed pipeline smoke or a failed agent smoke is not a commit.

## 5. Compare

A phase is done when its printed rate on the series for that phase is at least 90% of the kernel ceiling. The file-agent series and the coding-agent series stay separate.

If it is short, modify again and run section 3. Reuse the saved ceiling unless a designed kernel changed. Each agent smoke is a new run. Commit again when that run passes.

When 90% is met, report the ceiling and the printed rate. Do not write that rate into another pipeline's note.
