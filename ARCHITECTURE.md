# Favlo architecture

One model is one pipeline. `favlo serve` listens with nothing loaded. The `model` field on `POST /v1/chat/completions` loads one registry row. A chat record and a media record are not resident together. An embeddings record opens beside the resident model for that request and closes when the request returns. A pipeline does not import another pipeline, and it does not share that pipeline's forward, records, or kernels. Quantize and the other non-serve commands require `favlo --<pipeline>`.

This file is the map and the write set. It wins for the listener, the registry, the console, and which files an agent may edit. A pipeline note wins for that pipeline's records, cache, and kernels. Implementation plans stay under `artifacts/implementation-plan/` (gitignored). A measured research remainder stays at `docs/<pipeline>/research/<slug>.md` and is indexed from [RESEARCH.md](RESEARCH.md).

## Map

| Area | Code | Note |
|---|---|---|
| Command discovery | `src/favlo/cli.py` | This file. The command scans `pipelines/metal41`, `pipelines/cuda`, and `custom-pipelines/`. Adding a runtime does not edit `cli.py`. |
| Listener and daemon | `src/favlo/listener.py`, `src/favlo/daemon.py` | [docs/serve.md](docs/serve.md) |
| Terminal chat | `src/favlo/talk.py`, `src/favlo/persona.py`, `src/favlo/config_tools.py` | [docs/ai-assistance](docs/ai-assistance/README.md). The persona is the horse. The terminal instructions add the opening line and the memory tools. The configuration tool group adds the guide, setup, model, and download-reminder tools, and the terminal enables that group. |
| Console | `src/favlo/console.py` | [docs/console.md](docs/console.md) |
| Registry | `src/favlo/registry.py`, `models/registry.json` | This file |
| Prefix store | `src/favlo/prefix.py` | [docs/prefix.md](docs/prefix.md) |
| Memory | `src/favlo/memory.py` | Daemon ledger of owners a pipeline process reported, plus that process's device. After that process exits, the ledger waits until those pages are reclaimable. A count that stops rising ends the wait, and the ledger drops the owners |
| Pipeline lifetime | `src/favlo/pipelines.py` | Daemon registry of the pipeline process. `close` halts that process, waits until its pages are reclaimable, and drops its ledger entry. A count that stops rising ends the wait. [docs/serve.md](docs/serve.md) |
| First-run setup | `src/favlo/setup.py` | [docs/guides/models.md](docs/guides/models.md) |
| Harness | `harness/` | [docs/harness.md](docs/harness.md) |

Shared code is `src/favlo/`. `Pipeline` in `pipeline.py` is the daemon call. `LanguagePipeline`, `EmbeddingsPipeline`, and `MediaPipeline` extend it. A served language model inherits `LanguagePipeline`. A served embedding model inherits `EmbeddingsPipeline`. A served image, video, music, or voice pipeline inherits `MediaPipeline`. None of those classes run a forward, name stages, or branch on the pipeline. A failed pipeline inherits none of them. A kernel does not branch on the pipeline. A new abstract method on `Pipeline` updates every subclass of every seam in that same change. A new abstract method on one seam updates every subclass of that seam. A pipeline task does not change `src/favlo/` to reach into another runtime. A pipeline imports the prefix store and adapts its own buffers. The listener passes `owner` into `open_runtime` when that function accepts it, records the owners created during that open, and sweeps them after `close`. An open that raises sweeps the owners that appeared during the call.

`POST /mcp` is the FastMCP media queue. The daemon owns the GPU admission queues in `schedule.py`: embeddings, then chats that have not started, then media. A chat preempts media. `preempt` stays set until the chat queue is empty. An embeddings request does not set `preempt`. `/reload` drops every waiter in both queues. A waiter does not run. The request that holds the GPU is cancelled, and the runtime closes. The listener checkpoints the media task and closes its device before the chat model allocates, resumes that same task only after the foreground queue has drained, and does not serve a media HTTP route. The daemon is a request server. `pipelines` is the daemon registry of open pipeline processes. One chat or media process stays resident. An embeddings process opens beside it for that request and closes when the request returns. `close` halts every registered process. The next call waits until the ledger says those pages are released. A reclaimable count that stops rising ends that wait, and the daemon drops the ledger entry. A pipeline crash fails the running job and leaves the daemon up. `POST /mcp` stays up with none loaded. It does not own chat history, tools, or the terminal session. The CLI owns those. A pipeline writes a reply and does not see a socket. `json` finishes one object, and that object is encoded before the call returns. `begin_events`, `event`, and `done` are one server-sent stream. `stopped` is true when that client has gone. Telemetry is one JSON event per line. The pipeline process forwards those events to the daemon. The terminal paints them as they arrive, and prints the reply after that turn's requests have closed.

`python -m favlo.smoke_lock` is how a pipeline test takes the GPU while the daemon stays up. Acquire closes the resident model when no request is open. Release lets the next chat load one again. The recorded pid is the invoking shell. `uv` and `python` are not that owner. A command started from a shell that already holds the lock leaves the file in place and does not release it. Image, video, and music bytes stay on disk. The uncompressed AVI is `~/.favlo/videos/<id>.avi`. The MP4 is `<id>.mp4` beside it, encoded from that AVI. `video_ttl_seconds` in `settings.json` is how long it stays. The default is 86400. A sweep at startup and after each stored clip deletes a file older than that. The uncompressed RGB is `~/.favlo/images/<id>.rgb`. The PNG is `<id>.png` beside it, encoded from that file at the request's zlib level. `image_ttl_seconds` uses the same default and the same sweep. The WAV is `~/.favlo/music/<id>.wav`. The MP3 and FLAC are encoded from that WAV. `music_ttl_seconds` uses the same default and the same sweep. The spoken WAV is `~/.favlo/voice/<id>.wav`. The MP3 and FLAC are encoded from that WAV. `voice_ttl_seconds` uses the same default and the same sweep. A client reads a stored file with `resources/read`. A path under `/v1/images`, `/v1/videos`, or `/v1/audio` is 404. The listener note is [docs/serve.md](docs/serve.md).

## Runtimes

Each runtime is a separate write set. An agent developing one runtime edits that set and leaves the others untouched.

| Pipeline | Entry | Package | As-built note |
|---|---|---|---|
| `qwen38-flash-next` | `pipelines/metal41/qwen38-flash-next.py` | `pipelines/metal41/qwen38_flash_next/` | [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md) |
| `qwen38-27b` | `pipelines/metal41/qwen38-27b.py` | `pipelines/metal41/qwen38_27b/` | [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md) |
| `qwen36-35b-a3b` | `pipelines/metal41/qwen36-35b-a3b.py` | `pipelines/metal41/qwen36_35b_a3b/` | [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md) |
| `z-image-turbo` | `pipelines/metal41/z-image-turbo.py` | `pipelines/metal41/z_image_turbo/` | [docs/z-image-turbo/README.md](docs/z-image-turbo/README.md) |
| `wan22-i2v` | `pipelines/metal41/wan22-i2v.py` | `pipelines/metal41/wan22_i2v/` | [docs/wan22-i2v/README.md](docs/wan22-i2v/README.md) |
| `wan22-ti2v-5b` | `pipelines/metal41/wan22-ti2v-5b.py` | `pipelines/metal41/wan22_ti2v_5b/` | [docs/wan22-ti2v-5b/README.md](docs/wan22-ti2v-5b/README.md) |
| `ace-step-15-turbo` | `pipelines/metal41/ace-step-15-turbo.py` | `pipelines/metal41/ace_step_15_turbo/` | [docs/ace-step-15-turbo/README.md](docs/ace-step-15-turbo/README.md) |
| `bge-m3` | `pipelines/metal41/bge-m3.py` | `pipelines/metal41/bge_m3/` | [docs/bge-m3/README.md](docs/bge-m3/README.md) |
| `voxcpm2` | `pipelines/metal41/voxcpm2.py` | `pipelines/metal41/voxcpm2/` | [docs/voxcpm2/README.md](docs/voxcpm2/README.md) |

The entry filename may contain hyphens. The command loads it by path from `pipelines/<runtime>/`. It is not an importable module name. The package beside it uses underscores. The entry sets `PIPELINE` to that filename stem. A runtime's write set is the entry file, the package, `docs/<pipeline>/`, and tests that import only that package. A new runtime adds a row to the table above and its own doc directory. It does not edit another row's package, shaders, tests, or docs. Shipped rows use `runtime` `metal41`. `pipelines/cuda` is empty until a CUDA pipeline names the toolkit it needs. A custom entry lives under `custom-pipelines/<runtime>/` and its registry row sets `custom` true. A missing or unknown `runtime` is an error.

## Failed

A failed pipeline is not discovered and is not served. Its entry is not under `pipelines/metal41` or `pipelines/cuda`. `serve` does not load it, and the command scan does not list its flag.

| Pipeline | Entry | Package | Note |
|---|---|---|---|
| `ternary-bonsai-2` | `pipelines/failed/ternary-bonsai-2.py` | `pipelines/failed/ternary_bonsai_2/` | [docs/ternary-bonsai-2/README.md](docs/ternary-bonsai-2/README.md) |
| `zonos2` | `pipelines/failed/zonos2.py` | `pipelines/failed/zonos2/` | [docs/zonos2/README.md](docs/zonos2/README.md) |

Tests for the listener, the registry, and the console live in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_registry.py`, and `tests/test_console.py`. A pipeline's tests do not import another pipeline.

## Registry

`models/registry.json` is gitignored. A row stores the folder, `directory` (the quantized record that was written), the pipeline entry, `runtime`, an alias, the BF16 source path, the key/value mode, the resident percent, `context_length`, and the sampling card copied from that pipeline's `sampling.py`. A row may store `parts`, named quantized directories such as `umt5`, `high`, and `low`. `serve` opens `directory` and passes `parts` to a runtime that accepts them. A row that omits `directory` opens `models/<folder>`. `runtime` is `metal41` or `cuda`. A shipped row omits `custom`. A custom row sets `custom` true. `serve` loads `pipelines/<runtime>/<pipeline>.py`, or `custom-pipelines/<runtime>/<pipeline>.py` when `custom` is true. A row that omits `context_length` uses 131072. The stored names are the `:suffix` ids. A bare model id selects no row. The pipeline applies its default variant, or the row the request selects. `serve` does not register a record it finds on disk. A fresh tree has no rows. `GET /v1/models` lists every registered language row, including each sampling suffix that row stores, and each object includes `context_length`. A pipeline that defines `handle_image`, `handle_video`, `handle_music`, `handle_voice`, or `handle_embed` is omitted, and `GET /v1/models/{id}` for that id is 404. `POST /v1/embeddings` loads the pipeline that defines `handle_embed`. An embeddings request waits until the open chat completion returns, does not set `preempt`, and does not close the resident model. Its process closes when the request returns. A chat that has not started waits while an embeddings request is queued or running.

## Layout

```text
pipelines/metal41/      shipped pipelines that require Metal 4.1
pipelines/cuda/         empty until a CUDA pipeline exists
pipelines/failed/       pipelines that are not discovered and are not served
custom-pipelines/       gitignored; one entry and package per custom model
src/favlo/              process entry; the listener loads the pipeline a request names
  cli.py                discovers entry files; a new runtime does not edit this
  registry.py           folders, aliases, pipelines, and sampling variants
  listener.py           process, model load, and the request lock; a terminal starts it and keeps the CLI
  http_app.py           ASGI routes for chat and MCP
  schedule.py           foreground queue and media queue; the daemon admits the GPU
  pipeline.py           daemon call every served seam extends
  language_pipeline.py  disconnect shell a served language model inherits
  embeddings_pipeline.py list object a served embedding model inherits
  media_pipeline.py     listener contract a served media pipeline inherits
  mcp_server.py         FastMCP media queue, Tasks, and favlo:// reads
  reply.py              the object a pipeline writes; the server turns it into HTTP
  daemon.py             spawn and stop the listener; no pipeline imports
  console.py            status row; no pipeline imports
  talk.py               terminal chat
  persona.py            voice and saved openings
  config_tools.py       configuration tool group for the terminal and the web
  prefix.py             SSD prefix store
  memory.py             device, placement heaps, and span handles
  memory_notes.py       ~/.favlo/memory.db for MCP remember, recall, and forget
  pipelines.py          the daemon registry of the pipeline process
  setup.py              empty-registry download
  logs.py               rolling server log; size is settings.json log_max_bytes; heartbeat_seconds is the terminal quiet interval
  videos.py             stored AVI and the MP4 encoded from it; video_ttl_seconds drops an older clip
  images.py             stored RGB and the PNG encoded from it; image_ttl_seconds drops an older pair
  music.py              stored WAV and the MP3 or FLAC encoded from it; music_ttl_seconds drops an older clip
  voice.py              stored spoken WAV and the MP3 or FLAC encoded from it; voice_ttl_seconds drops an older clip
harness/                file agent, coding agent, worktree agent, and the three benchmarks
docs/                   as-built notes; see docs/README.md
```

Probe, matmul, kernels, bf16, complete, and quantize stay on that pipeline's flag: `favlo --qwen38-flash-next quantize`, and the same flag for the other commands. The listener runs one request. A language-model pipeline does not fuse a whole layer into one kernel. An image, video, music, or voice pipeline may place several kernels of one generation in one command buffer and may fuse those kernels. Prefill, decode, and the calibration prefill are Metal shaders. A missing kernel stays missing. The Neural Engine may run a module.

## Commands

```text
uv sync --extra dev
uv run pytest
uv run favlo serve
uv run favlo --qwen38-flash-next probe
uv run favlo --qwen38-flash-next matmul
uv run favlo --qwen38-flash-next kernels
uv run favlo --qwen38-flash-next quantize
uv run favlo --qwen38-27b quantize
uv run favlo --qwen36-35b-a3b kernels
uv run favlo --qwen36-35b-a3b quantize
uv run python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness --model qwen38-27b
uv run python -m harness.benchmarks --model qwen38-27b
uv run python -m harness.coding --model qwen38-27b
uv run python -m harness.worktree --model qwen36-35b-a3b
uv run python -m harness.image_smoke_agent
uv run python -m harness.image_persona
uv run python -m harness.video_smoke_agent
uv run python -m harness.media_pipeline
uv run python -m harness.language_pipeline
uv run python -m harness.voice_smoke_agent
uv run python -m harness.music_long_agent
uv run python -m harness.embeddings_agent
uv run python -m harness.server status
```

What the harness scores is [docs/harness.md](docs/harness.md). A pipeline's quantize, probe, and matmul commands are that pipeline's note.
