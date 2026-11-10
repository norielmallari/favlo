---
name: media-preemption
description: Run the media preemption checks after a change to MediaPipeline, a hold tuple, or a media parse, generate, pack, pause, or job dict. Do not use for a kernel or quantize edit that leaves the request and the checkpoint alone.
---

# Media preemption

Load this skill when `MediaPipeline`, a hold key, or a media package's `parse`, `generate`, `pack`, pause, or job dict changes. A kernel, shader, or quantize edit that leaves the request, the checkpoint, and the job dict alone does not run it.

One agent covers every served media package: `python -m harness.media_pipeline`. It discovers registered packages. Kind names on the command may stay: `image`, `i2v`, `t2v`, `music`, `voice`. An omitted list runs every kind. Do not add a second preemption agent or a probe row that names a package.

```text
uv run pytest tests/test_media_pipeline.py tests/test_media_pipeline_harness.py
uv run python -m harness.media_pipeline
```

The pytest does not open a listener and does not take the machine lock. The harness uses one `favlo serve` on `127.0.0.1:8129` and does not take the lock. `harness.server` waits until no live pid holds `artifacts/pipeline-test.lock`. Exit 0 means every selected probe paused with `task is ongoing` and that same task finished.

A GPU command in its own process uses the smoke-testing skill. A harness that calls the listener uses `attach` and does not take the lock. Do not kill serve to free the GPU. Close during a live hold is 409.

`MediaPaused("preempt", hold)` is raised after the submitted command finishes. Clear a hold with `state[key] = None`. Do not pop the key.
