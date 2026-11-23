# Inkling 2 Mini

Summary for the `inkling_2_mini` language pipeline. Kernels, the checkpoint layout, the forward path, dense calibration, and the chat template are in [ARCHITECTURE.md](ARCHITECTURE.md).

- Clients reach this pipeline through `favlo serve`, which loads `inkling_2_mini.portable`.
- `portable` is the serve entry. `metal41` is the Metal device that entry opens on Apple Silicon.
- After the checkpoint is registered, `GET /v1/models` lists it and `model` on `POST /v1/responses` loads it.
- `kernels`, `setup`, `calibrate`, `bake`, `complete`, and `models` require `favlo --inkling_2_mini`, which loads that same entry.

- The official checkpoint is [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
- Those weights are under the Apache License, Version 2.0.
- The served folder is `Inkling-2-Mini-fp4-gptq-foem`.
- `setup` downloads [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) into that folder, stores former bf16 tensors as float16, writes vision tensors to `vision.safetensors`, leaves `mtp.safetensors` unread, and bakes the expert planes and the dense fp8 planes. The fused attention plane and the output projection are nearest E4M3. The head is pread as float16. `calibrate` rounds layers 0–19 from NVFP4 to Metal fp4 with GPTQ-FOEM. Layers 20–23 stay MXFP8. There is no `--keep-bf16`.
- The download is about 18.2 GiB. The expert pack adds about 14.4 GiB. The open preads `dense-fp8.safetensors`, the float16 head, and the expert pack in 8 MiB pieces. Without the expert pack the open requantizes layers 0–19. Without the dense pack the open quantizes attention and the output projection.
- A finished folder is registered again. A partial download continues. `--repo` accepts `org/name` or a `huggingface.co` link. `--name` is the folder under `models/`.
- The registered sampler is temperature 1.0, top-p 1.0, top-k 0, and no presence penalty. The checkpoint publishes that temperature for every effort, so `thinking` and `instruct` store the same card. A request sets the effort with `reasoning_effort`. The row default is `high`. The row preserves thinking.
- The checkpoint context is 1,048,576 tokens. The served window starts at 131072. The half key/value cache reserves that window for the eight full-attention visits and a 1024-token ring for each local visit. The expert atlas keeps every image when the budget allows, or the eight routed images otherwise. The window shrinks only when those eight images and the cache do not fit.
- A complete atlas and that window use about 25 GB. 16 GB and 24 GB do not hold every expert image. 48 GB does.
- Prompts reuse the shared SSD prefix store.

```text
uv run favlo --inkling_2_mini setup
uv run favlo --inkling_2_mini calibrate
uv run favlo --inkling_2_mini bake
uv run favlo --inkling_2_mini kernels
```

A user image runs the vision tower. `bake` writes `vision-fp8.safetensors` beside the checkpoint. A request reads that file in 8 MiB pieces and releases it after the rows are written.

Measured bandwidth and kernel costs are in [ceilings.md](ceilings.md). Measured runs are in [research/synthetic-prompt-fp8.md](research/synthetic-prompt-fp8.md), [research/synthetic-prompt-fp8-nearest.md](research/synthetic-prompt-fp8-nearest.md), [research/coding-text-fp8.md](research/coding-text-fp8.md), [research/coding-text-fp8-nearest.md](research/coding-text-fp8-nearest.md), [research/file-agent-fp8.md](research/file-agent-fp8.md), [research/file-agent-fp8-nearest.md](research/file-agent-fp8-nearest.md), [research/synthetic-prompt-half.md](research/synthetic-prompt-half.md), [research/coding-text-half.md](research/coding-text-half.md), [research/coding-vision-half.md](research/coding-vision-half.md), [research/file-agent-half.md](research/file-agent-half.md), [research/intelligence-smoke.md](research/intelligence-smoke.md), and [research/intelligence-smoke-fp8-nearest.md](research/intelligence-smoke-fp8-nearest.md). Harness scores are in [benchmarks.md](benchmarks.md).
