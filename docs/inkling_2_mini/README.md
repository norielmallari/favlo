# Inkling 2 Mini

Summary for the `inkling_2_mini` language pipeline. Kernels, the checkpoint layout, the forward path, dense calibration, and the chat template are in [ARCHITECTURE.md](ARCHITECTURE.md).

- `handle_responses` is a `LanguagePipeline` served through `favlo serve`.
- After the checkpoint is registered, `GET /v1/models` lists it and `model` on `POST /v1/responses` loads it.
- `kernels`, `setup`, `calibrate`, `complete`, and `models` require `favlo --inkling_2_mini`.

- The official checkpoint is [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
- Those weights are under the Apache License, Version 2.0.
- The served folder is `Inkling-2-Mini`.
- `setup` registers that folder. It does not requantize.
- `calibrate` writes `dense-fp4.safetensors` into that folder: the dense fp4 planes rounded with GPTQ from calibration inputs. It takes about ten minutes on the GPU and the host. Without that file the open rounds to the nearest code, and tool calls are less reliable.
- The checkpoint context is 1,048,576 tokens. The served window starts at 16384. The half key/value cache is reserved first, and the expert atlas keeps every image when the budget allows, or the eight routed images otherwise. The window shrinks only when those eight images do not fit.
- A complete atlas and that window use about 34 GB. 16 GB and 24 GB do not hold every expert image. 48 GB does.
- Prompts reuse the shared SSD prefix store.

```text
uv run favlo --inkling_2_mini setup
uv run favlo --inkling_2_mini calibrate
uv run favlo --inkling_2_mini kernels
```

Text only. An image request is rejected.

Measured bandwidth and kernel costs are in [ceilings.md](ceilings.md). Measured runs are in [research/synthetic-prompt-half.md](research/synthetic-prompt-half.md) and [research/coding-text-half.md](research/coding-text-half.md).
