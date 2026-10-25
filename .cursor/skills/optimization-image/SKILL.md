---
name: optimization-image
description: Optimize one Favlo image pipeline toward its generation-time ceiling. Use when shortening image generation, closing a gap to that kernel ceiling, or when the user asks to optimize an image model. Do not use for language-model prefill or decode, or for music prompting.
---

# Image optimization

One image pipeline. A language model uses [optimization-loop](../optimization-loop/SKILL.md). Music generation time uses [optimization-music](../optimization-music/SKILL.md). A music prompt uses [music-prompting](../music-prompting/SKILL.md). Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) and that pipeline's `docs/<pipeline>/` note. Edit only that write set. Leave every other pipeline untouched.

The whole process is non-HITL until the printed generation time is at most the kernel ceiling divided by 0.9. A shorter time is faster. Do not ask, and do not stop between iterations, until that bar is met or a step cannot proceed.

## 1. Ceiling

The ceiling is a computation on the designed kernels for one generation at the resolution named in `docs/<pipeline>/ceilings.md`, not a time from `serve`. Reuse that file when the kernels named there are unchanged. Do not recompute it.

Recompute when the file is missing or any designed kernel changed. For one generation at that resolution:

- Count the bytes the designed kernels read and write. Divide by the streaming bandwidth measured on an M5 Pro with 48 GB. That is the bandwidth ceiling.
- Time each designed kernel in its own command buffer, then wait. Compose one generation from those isolated times. That is the kernel ceiling.
- Do not add `GPUStartTime` / `GPUEndTime` intervals that overlap across in-flight buffers.
- Do not edit production code to take the measurement. A missing kernel stays missing; do not substitute another forward.

Write the recomputed ceilings into `docs/<pipeline>/ceilings.md` in the same change as the kernels they describe. Name the kernels, the resolution, the bytes, the bandwidth, and both ceilings. Do not invent a number. No hostnames, home directories, or volume paths.

The ceiling used for the 90% test is the kernel ceiling. The bandwidth ceiling is the bound that kernel design is measured against.

## 2. Modify

Change this pipeline's kernels or the forward that launches them. Update `docs/<pipeline>/` in the same change when the kernels, records, or cache change.

Take the GPU with `uv run python -m favlo.smoke_lock acquire --pipeline <stem> --command <name>` before any command that compiles shaders, dispatches Metal, loads a checkpoint, quantizes, or starts `favlo serve`. Run acquire, the command, and `uv run python -m favlo.smoke_lock release` in the same shell. Exit 2 or 3 means wait and run acquire again. Do not kill a serve the user left running.

## 3. Smoke

After a modification, stop at the first failure, fix it, and rerun from that step. Do not start the image harness while `kernels` or the latent smoke is failing. Do not run the language-model smoke skill.

1. `uv run pytest` on the tests that import only this package, excluding a smoke that generates unless that is the next step.
2. `uv run favlo --<pipeline> kernels` when that command exists.
3. The pipeline's recorded generation smoke, when it has one.
4. The image harness for this pipeline, when one exists. That harness is the picture check. It is not the file agent or the coding agent.

## 4. Commit

Once those smokes pass, commit the related code on the current branch, including `main`.

Related code is this iteration's write set: the pipeline entry, package, `docs/<pipeline>/`, and tests that import only this package. When the printed generation time is at most the kernel ceiling divided by 0.9, record that time in the pipeline note before the commit.

Leave models, artifacts, the machine lock, and other pipelines out of the commit. Use a Conventional Commit. Do not push.

A failed `kernels`, latent smoke, or image harness is not a commit.

## 5. Compare

A generation is done when its printed time, at the resolution in the ceiling note, is at most the kernel ceiling divided by 0.9.

If it is short of that, modify again and run section 3. Reuse the saved ceiling unless a designed kernel changed. Each image harness run is a new run. Commit again when that run passes.

When the bar is met, report the ceiling and the printed time. Do not write that time into another pipeline's note.
