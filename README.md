# Favlo

Favlo: Your AI, the Whole Stable.

<p align="center">
  <img src="docs/images/favlo-stable.jpg" alt="Favlo" width="480">
</p>

>This app is directly inspired by [oMLX](https://github.com/jundot/omlx) and [ComfyUI](https://github.com/Comfy-Org/ComfyUI). These are amazing apps - I recommend trying them first before using my more specialized, opinionated runtime.

Favlo is a runtime for language models, embeddings, and media.

- One program serves chat, embeddings, images, video, music, and voice.
- You name the model, and that request runs on the same server.
- A chat or an embedding goes first.
- A media job pauses safely and continues when that chat is done.
- The command line is for the person at the machine that runs Favlo. It starts the server, prepares a model, and generates media.
- Other programs use that server for chat, embeddings, and media tools. The tool catalog is [MCP.md](MCP.md).

## Requirements

- Apple Silicon with Metal 4.1, or Windows with an NVIDIA graphics driver
- 16 GB of memory
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

Apple Silicon compiles shaders at Metal 4.1. A machine that cannot compile that version cannot run a Metal pipeline. Windows loads `engine.dll` from the checkout. The NVIDIA driver supplies the CUDA runtime. The CUDA Toolkit and Visual Studio are only for rebuilding that DLL. 8 GB is not enough to run a model capable of real-world agentic workflows.

## Install

From a checkout of this repository:

```bash
uv sync --extra dev
```

- This installs the `favlo` package and the `dev` extra, including pytest.
- This checkout is release 0.6.0.
- PyPI publishes the runtime under `src/favlo` as [`favlo`](https://pypi.org/project/favlo/).
- The pipelines, the shaders, and the web app stay in this checkout.

## Usage

```bash
uv run favlo serve
```

`favlo media --pipeline <id>` calls a listener that is already up.

- It does not start `favlo serve`.
- It is the command that generates. A pipeline command does not.
- Remaining arguments are `key=value` body fields.
- Stdout is one `file://` URI per stored file.
- `--out` writes the pack JSON (`metadata` and `mediaUri`).
- A pipeline's arguments are in that pipeline's note.

On a terminal with an empty registry, this downloads Qwen3.5-4B and packs it before the daemon listens. Wait until that finishes, and leave this process running.

The web app needs its packages. From the repository root:

```bash
npm install --prefix web-spa
```

In another terminal:

```bash
uv run favlo web
```

When the app is ready, open the printed link in a browser. That address is `http://127.0.0.1:8012`.

## Gallery

A Gallery is a workspace on an agent's channel.

- New workspace offers Gallery.
- Bind a local folder once.
- Favlo captions every image in that folder, then shows a preview and a filmstrip beside the chat.
- The agent places those views. The browser does not invent the layout.
- Find, search, select, and generate work through that chat.
- A generate call writes a plain file name into the folder.
- On a wide window the chat is a resizable column beside the preview.
- On a narrow window the preview, the filmstrip, and the chat share one column.
- The chat dock has three modes: fullscreen, peek, and expanded.

- App: [web-spa/README.md](web-spa/README.md)
- API: [web-api/README.md](web-api/README.md)
- Tools and greeting: [docs/ai-assistance/README.md](docs/ai-assistance/README.md)

## Supported Models

| Pipeline | Note | 16 GB | 24 GB | 48 GB |
|---|---|---|---|---|
| `qwen35_4b` | [Qwen3.5-4B](docs/qwen35_4b/README.md) | ✅ | ✅ | ✅ |
| `qwen36_35b_a3b` | [Qwen3.6-35B-A3B](docs/qwen36_35b_a3b/README.md) | ⚠️ | ⚠️ | ✅ |
| `inkling_2_mini` | [Inkling 2 Mini](docs/inkling_2_mini/README.md) | ❌ | ❌ | ✅ |
| `gemma4_26b_a4b` | [Gemma 4 26B-A4B](docs/gemma4_26b_a4b/README.md) | ⚠️ | ⚠️ | ✅ |
| `qwen38_27b` | [Qwen3.8-27B](docs/qwen38_27b/README.md) | ⚠️ | ⚠️ | ✅ |
| `bge_m3` | [BGE-M3](docs/bge_m3/README.md) | ✅ | ✅ | ✅ |
| `z_image_turbo` | [Z-Image-Turbo](docs/z_image_turbo/README.md) | ✅ | ✅ | ✅ |
| `z_image_base` | [Z-Image](docs/z_image_base/README.md) | ✅ | ✅ | ✅ |
| `klein_4b` | [FLUX.2 [klein] 4B](docs/klein_4b/README.md) | ✅ | ✅ | ✅ |
| `wan22_i2v` | [Wan 2.2 I2V](docs/wan22_i2v/README.md) | ❌ | ⚠️ | ✅ |
| `wan22_ti2v_5b` | [Wan 2.2 TI2V-5B](docs/wan22_ti2v_5b/README.md) | ✅ | ✅ | ✅ |
| `ace_step_15_turbo` | [ACE-Step 1.5 XL Turbo](docs/ace_step_15_turbo/README.md) | ✅ | ✅ | ✅ |
| `voxcpm2` | [VoxCPM2](docs/voxcpm2/README.md) | ✅ | ✅ | ✅ |

- ⚠️ is degraded mode. The pipeline still runs the request. It shrinks the one knob it owns until that request fits.
- The request keeps the resolution, the frame count, and the duration it named.
- ✅ keeps that knob at its full size.
- ❌ does not fit.

A media package lives at `pipelines/<pipeline>/<runtime>/`.

- Its module returns `mcp_tools()`.
- The host publishes `{pipeline}__{local name}`.
- The call takes a media ticket. A chat preempts it.
- The package note names the tool and the checkpoint.
- `z_image` is the shared image forward and is not a CLI flag.
- `umt5_xxl` is the shared text-encoder pack. The Wan commands pack it.

## Methodology

A pipeline equation comes from a public document, then maps onto a Metal operation. Another application's source is not a reference.

- Process citations: [REFERENCES.md](REFERENCES.md)
- A pipeline's own citations are in that pipeline's note.

The prompt and the generated tokens are separate phases.

- The key/value cache crosses that boundary.
- A prompt is computed in chunks.
- Prefill and decode are Metal shaders.
- A missing kernel stays missing.

Weights follow the Open Compute Project *Microscaling Formats (MX) Specification*.

- A block holds 32 values and an E8M0 scale.
- Apple's Metal tensor documents name `matmul2d` and that scale plane.
- WWDC26 session 330 is the tensor-op session.
- The Metal shading-language specification is the unpack and storage reference.

## Memory management

`src/favlo/memory.py` is the allocator for every device span.

- The serve process keeps the ledger.
- A pipeline child allocates its own Metal buffers and reports each span.
- `budget()` is reclaimable pages plus the live, pending, and returning bytes of every child of this daemon.
- `--max-memory` caps that answer.
- A pipeline receives a handle. The heap stays with the owner of that opening.
- `close` frees that owner's spans.
- The next request stays queued while an allocation or a release is still open.
- The reclaim wait ends when those pages are reclaimable, or when the reclaimable count stops rising.
- Bytes still held stay in the budget as returning.
- At open, the pipeline reads the budget and shrinks the one knob it owns until the request fits.
- The request keeps the resolution, the frame count, and the duration it named.
- When the minimum working set does not fit, the pipeline refuses to open.

## Pipeline architecture

`Pipeline` in `src/favlo/pipeline.py` is the call the daemon makes. Three classes extend it. Each one writes the reply its route already returns. The forward, the sampler, and the kernels stay in the pipeline package.

| Class | Route | What the class owns |
|---|---|---|
| `LanguagePipeline` | `POST /v1/responses` | The disconnect shell around one response |
| `EmbeddingsPipeline` | `POST /v1/embeddings` | The embeddings list object |
| `MediaPipeline` | `POST /mcp` | Shared job, pause, and host checkpoint. The subclass sets kind, id prefix, and object name, and implements `parse`, `generate`, and `pack` |

- A served language model inherits `LanguagePipeline` and binds `handle_responses`.
- The listener translates `POST /v1/chat/completions` into that request and translates the Response back. That translation is not under active development.
- A served embedding model inherits `EmbeddingsPipeline` and binds `handle_embed`.
- A served image, video, music, or voice package subclasses `MediaPipeline`. It sets `kind`, `id_prefix`, and `object_name`, and returns `mcp_tools()`.
- A package that is not served inherits none of them.
- The map is [ARCHITECTURE.md](ARCHITECTURE.md).

The daemon records the resident pipeline process.

- That process loads the registry row the request named and creates the device.
- The row stores the packed record, the pipeline entry, and the sampling card.
- One chat or media process stays resident.
- A different chat or media row halts that process.
- The next open waits until its pages are reclaimable, or until the reclaimable count stops rising and the bytes stay as returning.
- An embeddings request opens its own process beside the resident model and closes that process when the request returns.
- It does not halt the resident model.
- A crash fails the job that was running.
- The daemon stays up, drops that process from the ledger, and listens for the next request.

## Queueing

`src/favlo/schedule.py` admits one request to the device.

- An embeddings request waits until a chat already on the device returns.
- A chat that has not started waits while an embeddings request is queued or running.
- Image, video, music, and voice run when that foreground queue is empty.
- A chat pauses the media job at its next safe point and closes the device.
- The same job resumes after the chat queue has drained.
- An embeddings request leaves the resident model loaded and closes its own process when the request returns.
- One media job runs at a time.
- A chat whose client has gone leaves the queue.
- The listener is [docs/serve.md](docs/serve.md).

## Prefix cache

`src/favlo/prefix.py` stores token ids and named regions on SSD, under `~/.favlo/prefix/`.

- Each pipeline packs its own buffers into those regions and copies a matching file back.
- A prompt that extends the live tokens keeps the device.
- Any other prompt loads the longest stored prefix that pipeline accepts.
- One model and one checkpoint stay within 32 GiB.
- The note is [docs/prefix.md](docs/prefix.md).

## Development

Favlo is an AI-First application. Agentic software delivery is how this repository is built: agents design the change, write the code, and land it. The author directs the work.

## Further reading

- Process contract: [ARCHITECTURE.md](ARCHITECTURE.md)
- Papers: [REFERENCES.md](REFERENCES.md)
- Measured research claims: [RESEARCH.md](RESEARCH.md)
- Citation: [CITATION.cff](CITATION.cff)
- Agents: [AGENTS.md](AGENTS.md)
- AI-assisted features: [docs/ai-assistance](docs/ai-assistance/README.md)

## Problems

Report a problem in this repository's issue tracker.

## License

Copyright 2026 Noriel Mallari

Favlo is licensed under the Apache License, Version 2.0. The terms are in [LICENSE.md](LICENSE.md). A checkpoint keeps the license named in its note.

| Checkpoint | License | Note |
|---|---|---|
| Qwen3.5-4B | Apache License, Version 2.0 | [note](docs/qwen35_4b/README.md) |
| Qwen3.6-35B-A3B | Apache License, Version 2.0 | [note](docs/qwen36_35b_a3b/README.md) |
| Inkling 2 Mini | Apache License, Version 2.0 | [note](docs/inkling_2_mini/README.md) |
| Gemma 4 26B-A4B | [Apache License, Version 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) | [note](docs/gemma4_26b_a4b/README.md) |
| Qwen3.8-27B | Apache License, Version 2.0 | [note](docs/qwen38_27b/README.md) |
| bge-m3 | MIT License | [note](docs/bge_m3/README.md) |
| Z-Image-Turbo | Apache License, Version 2.0 | [note](docs/z_image_turbo/README.md) |
| Z-Image | Apache License, Version 2.0 | [note](docs/z_image_base/README.md) |
| FLUX.2 [klein] 4B | Apache License, Version 2.0 | [note](docs/klein_4b/README.md) |
| Wan 2.2 I2V | Apache License, Version 2.0 | [note](docs/wan22_i2v/README.md) |
| umT5-XXL | Apache License, Version 2.0 | [note](docs/umt5_xxl/README.md) |
| Wan 2.2 TI2V-5B | Apache License, Version 2.0 on the base card. The source checkpoint's license is unset. | [note](docs/wan22_ti2v_5b/README.md) |
| ACE-Step 1.5 XL Turbo | MIT License | [note](docs/ace_step_15_turbo/README.md) |
| VoxCPM2 | Apache License, Version 2.0 | [note](docs/voxcpm2/README.md) |
