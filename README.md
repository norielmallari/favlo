# Favlo

Favlo: Your AI, the Whole Stable.

<p align="center">
  <img src="docs/images/favlo-stable.jpg" alt="Favlo" width="480">
</p>

>This app is directly inspired by [oMLX](https://github.com/jundot/omlx) and [ComfyUI](https://github.com/Comfy-Org/ComfyUI). These are amazing apps - I recommend trying them first before using my more specialized, opinionated runtime.

Favlo runs locally or in the cloud. One process serves language models, embeddings, and media: image, video, music, and voice. One model is one pipeline. When the pipelines share a machine, chat and embeddings take the device. A media job yields at its next safe point, checkpoints, and resumes when that queue has drained. Each pipeline owns its forward, its weight records, and its Metal kernels. `favlo serve` is the entry point. A client selects a registered language model with the OpenAI `model` field. A language row answers `POST /v1/chat/completions`. An embeddings row answers `POST /v1/embeddings`. Agents reach the content-generation models through MCP at `POST /mcp`. The tool catalog is [MCP.md](MCP.md).

The process contract is [ARCHITECTURE.md](ARCHITECTURE.md). The papers are [REFERENCES.md](REFERENCES.md). Measured research claims are [RESEARCH.md](RESEARCH.md). Cite this software from [CITATION.cff](CITATION.cff). Agents start at [AGENTS.md](AGENTS.md). AI-assisted features are [docs/ai-assistance](docs/ai-assistance/README.md).

## Requirements

- Apple Silicon
- Metal 4.1
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

Shaders compile at Metal 4.1. A machine that cannot compile that version cannot run a pipeline. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

This installs the `favlo` package and the `dev` extra, including pytest. The package is not published to PyPI.

## Usage

```bash
uv run favlo serve
uv run favlo serve --non-interactive
```

`favlo serve` on a terminal is the daemon. It listens on `127.0.0.1:8129` and paints progress on stderr. `favlo serve --non-interactive` starts a detached daemon and exits. Chat and embeddings are HTTP routes on that daemon. Image, video, music, and voice are MCP.

`uv run favlo web` starts the workspace beside that listener. The API is [web-api/README.md](web-api/README.md). The app is [web-spa/README.md](web-spa/README.md).

## Startup

On a terminal, an empty registry downloads [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), packs `models/Qwen3.6-35B-A3B-fp8-e20`, and then the daemon listens. The steps are [docs/qwen36-35b-a3b/SETUP.md](docs/qwen36-35b-a3b/SETUP.md). Quantize and the other non-serve commands use that model's flag, as in `favlo --qwen38-flash-next quantize`. The listener is [docs/serve.md](docs/serve.md). The image request is [docs/z-image-turbo/README.md](docs/z-image-turbo/README.md). Clients are [docs/guides/clients.md](docs/guides/clients.md). Assisted features are [docs/ai-assistance](docs/ai-assistance/README.md).

## Supported Models

| Pipeline | Note |
|---|---|
| `qwen36-35b-a3b` | [Qwen3.6-35B-A3B](docs/qwen36-35b-a3b/README.md) |
| `gemma4-26b-a4b` | [Gemma 4 26B-A4B](docs/gemma4-26b-a4b/README.md) |
| `qwen38-27b` | [Qwen3.8-27B](docs/qwen38-27b/README.md) |
| `qwen38-flash-next` | [Qwen3.8-Flash-Next](docs/qwen38-flash-next/architecture.md) |
| `z-image-turbo` | [Z-Image-Turbo](docs/z-image-turbo/README.md) |
| `wan22-i2v` | [Wan 2.2 I2V](docs/wan22-i2v/README.md) |
| `wan22-ti2v-5b` | [Wan2.2 TI2V-5B](docs/wan22-ti2v-5b/README.md) |
| `ace-step-15-turbo` | [ACE-Step 1.5 XL Turbo](docs/ace-step-15-turbo/README.md) |
| `bge-m3` | [BGE-M3](docs/bge-m3/README.md) |
| `voxcpm2` | [VoxCPM2](docs/voxcpm2/README.md) |

## Methodology

A pipeline equation comes from a public document, then maps onto a Metal operation. Another application's source is not a reference. The process citations are [REFERENCES.md](REFERENCES.md). A pipeline's own citations are in that pipeline's note.

The prompt and the generated tokens are separate phases. The key/value cache is what crosses the boundary. A prompt is computed in chunks. Prefill and decode are Metal shaders. A missing kernel stays missing.

Weights follow the Open Compute Project *Microscaling Formats (MX) Specification*: a block of 32 values and an E8M0 scale. Apple's Metal tensor documents name `matmul2d` and that scale plane. WWDC26 session 330 is the tensor-op session. The Metal shading-language specification is the unpack and storage reference.

## Memory management

`src/favlo/memory.py` is the allocator for every device span. A pipeline receives a handle, and the heap stays with the owner of that opening. `close` frees that owner's spans. The next request waits until those pages are reclaimable, and stays queued while an allocation or a release is still open. At open, the pipeline reads the budget and shrinks the one knob it owns until the request fits. The request keeps the resolution, the frame count, and the duration it named. When the minimum working set does not fit, the pipeline refuses to open.

## Pipeline architecture

`Pipeline` in `src/favlo/pipeline.py` is the call the daemon makes. Three classes extend it. Each one writes the reply its route already returns. The forward, the sampler, and the kernels stay in the pipeline package.

| Class | Route | What the class owns |
|---|---|---|
| `LanguagePipeline` | `POST /v1/chat/completions` | The disconnect shell around one chat completion |
| `EmbeddingsPipeline` | `POST /v1/embeddings` | The embeddings list object |
| `MediaPipeline` | `POST /mcp` | The job, the pause, and the host checkpoint |

A served language model inherits `LanguagePipeline` and binds `handle_chat`. A served embedding model inherits `EmbeddingsPipeline` and binds `handle_embed`. A served image, video, music, or voice pipeline inherits `MediaPipeline` and binds `handle_image`, `handle_video`, `handle_music`, or `handle_voice`. A pipeline that is not served inherits none of them. The map is [ARCHITECTURE.md](ARCHITECTURE.md).

The daemon records the one pipeline process. That process loads the registry row the request named and creates the device. The row stores the packed record, the pipeline entry, and the sampling card. A different row halts the process and waits until its pages are reclaimable. A crash fails the job that was running. The daemon stays up, drops that process from the ledger, and listens for the next request.

## Queueing

`src/favlo/schedule.py` admits one request to the device. An embeddings request waits until a chat already on the device returns, and a chat that has not started waits while an embeddings request is queued or running. Image, video, music, and voice run when that foreground queue is empty. A chat pauses the media job at its next safe point and closes the device. The same job resumes after the chat queue has drained. An embeddings request leaves the resident model loaded and closes its own process when the request returns. One media job runs at a time. A chat whose client has gone leaves the queue. The listener is [docs/serve.md](docs/serve.md).

## Prefix cache

`src/favlo/prefix.py` stores token ids and named regions on SSD, under `~/.favlo/prefix/`. Each pipeline packs its own buffers into those regions and copies a matching file back. A prompt that extends the live tokens keeps the device. Any other prompt loads the longest stored prefix that pipeline accepts. One model and one checkpoint stay within 32 GiB. The note is [docs/prefix.md](docs/prefix.md).

## Development

Favlo is an AI-First application. Agentic software delivery is how this repository is built: agents design the change, write the code, and land it. The author directs the work.

## Problems

Report a problem in this repository's issue tracker.

## License

Copyright 2026 Noriel Mallari

Favlo is licensed under the Apache License, Version 2.0. The terms are in [LICENSE.md](LICENSE.md). A checkpoint keeps the license named in its note.

| Checkpoint | License | Note |
|---|---|---|
| Qwen3.6-35B-A3B | Apache License, Version 2.0 | [note](docs/qwen36-35b-a3b/README.md) |
| Gemma 4 26B-A4B | Apache License, Version 2.0 | [note](docs/gemma4-26b-a4b/README.md) |
| Qwen3.8-27B | Apache License, Version 2.0 | [note](docs/qwen38-27b/README.md) |
| Z-Image-Turbo | Apache License, Version 2.0 | [note](docs/z-image-turbo/README.md) |
| Wan 2.2 I2V | Apache License, Version 2.0 | [note](docs/wan22-i2v/SETUP.md) |
| umT5-XXL | Apache License, Version 2.0 | [note](docs/wan22-i2v/SETUP.md) |
| Wan 2.2 TI2V-5B | Apache License, Version 2.0 on the base card. The source checkpoint's license is unset. | [note](docs/wan22-ti2v-5b/SETUP.md) |
| ACE-Step 1.5 XL Turbo | MIT License | [note](docs/ace-step-15-turbo/README.md) |
| bge-m3 | MIT License | [note](docs/bge-m3/README.md) |
| VoxCPM2 | Apache License, Version 2.0 | [note](docs/voxcpm2/references.md) |
| Qwen3.8-Flash-Next | [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) | [note](docs/qwen38-flash-next/references.md) |
