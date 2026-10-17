# Qwen3.8-27B

As-built note for the `qwen38-27b` pipeline. The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md). This directory wins for this pipeline's records and kernels. Another pipeline does not edit it.

Clients reach this pipeline through `madmetal serve`. `GET /v1/models` lists it once a record is registered, and the `model` field on `POST /v1/chat/completions` loads it. Quantize, bf16, complete, and `models` require `madmetal --qwen38-27b`. That flag loads `qwen38-27b.py`, which loads `qwen38_27b`. The package does not import `qwen38_flash_next`. `quantize`, `bf16`, and `complete` report that this forward is not available. `models` reads the shared registry. Registration copies the thinking and instruct rows from `sampling.py`. The default variant is thinking.
