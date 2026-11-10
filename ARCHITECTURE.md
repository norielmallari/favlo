# Favlo architecture

One model is one pipeline. `favlo serve` listens with nothing loaded. A new `favlo serve` stops another listener on that address and any pipeline process before it listens. The `model` field on `POST /v1/responses` loads one registry row. Every served language pipeline answers that route through `handle_responses`. `POST /v1/chat/completions` is translated on the listener (`chat_to_responses` and `ChatWire` in `favlo.responses`) into that same path; pipelines do not implement Chat Completions. A chat record and a media record are not resident together. An embeddings record opens beside the resident model and stays warm for `embeddings_ttl_seconds` after the request returns (`0` closes immediately). A pipeline does not import another pipeline. Z-Image-Turbo and Z-Image both call the shared forward in `pipelines/z_image/metal41/`. No other pipeline shares a forward, records, or kernels. Quantize and the other non-serve commands require `favlo --<pipeline>`. A pipeline command does not generate. `favlo media` is that call.

This file is the map and the write set. It wins for the listener, the registry, the console, and which files an agent may edit. A pipeline note wins for that pipeline's records, cache, and kernels. A measured research remainder stays at `docs/<pipeline>/research/<slug>.md` and is indexed from [RESEARCH.md](RESEARCH.md).

## Map

| Area | Code | Note |
|---|---|---|
| Command discovery | `src/favlo/cli.py` | This file. The command scans `pipelines/<pipeline>/<runtime>/` and each `pipeline_root` stored on a registry row. Adding a runtime does not edit `cli.py`. |
| Listener and daemon | `src/favlo/listener.py`, `src/favlo/daemon.py` | [docs/serve.md](docs/serve.md) |
| Persona and configuration tools | `src/favlo/persona.py`, `src/favlo/config_tools.py` | [docs/ai-assistance](docs/ai-assistance/README.md). The persona is the horse. Favlo web stores it on the Favlo agent and enables the configuration tool group there. |
| Console | `src/favlo/console.py` | [docs/console.md](docs/console.md) |
| Registry | `src/favlo/registry.py`, `models/registry.json` | This file |
| Prefix store | `src/favlo/prefix.py` | [docs/prefix.md](docs/prefix.md) |
| Memory | `src/favlo/memory.py` | The serve process keeps the ledger. A pipeline child allocates its own Metal buffers and reports each span on the worker pipe. It asks the serve process for the budget whenever it sizes a working set. The answer is reclaimable pages plus the live, pending, and returning bytes of every child of this daemon. A `--max-memory` cap is applied to that answer. After a child exits, the ledger waits until those pages are reclaimable. A count that stops rising leaves the bytes as returning |
| Pipeline lifetime | `src/favlo/pipelines.py` | Chooses the request in service and the runtime that serves it. One chat or media process stays resident. The next process is spawned after `memory.wait_released()`. An embeddings process opens beside it and stays warm for `embeddings_ttl_seconds`. Dropping the resident does not release that parked child. A crash sweeps that pipeline's owners and leaves the listener up. [docs/serve.md](docs/serve.md) |
| First-run setup | `src/favlo/setup.py` | [docs/guides/models.md](docs/guides/models.md) |
| Harness | `harness/` | [docs/harness.md](docs/harness.md) |

Shared code is `src/favlo/`. `Pipeline` in `pipeline.py` is the daemon call. `LanguagePipeline`, `EmbeddingsPipeline`, and `MediaPipeline` extend it. A served language model inherits `LanguagePipeline`. A served embedding model inherits `EmbeddingsPipeline`. A served image, video, music, or voice pipeline subclasses `MediaPipeline` and sets `kind`, `id_prefix`, and `object_name`. None of those classes run a forward, name stages, or branch on the pipeline. A failed pipeline inherits none of them. A kernel does not branch on the pipeline. A new abstract method on `Pipeline` updates every subclass of every seam in that same change. A new abstract method on one seam updates every subclass of that seam. A pipeline task does not change `src/favlo/` to reach into another runtime. A pipeline imports the prefix store and adapts its own buffers. The pipeline manager passes `owner` into `open_runtime` when that function accepts it, and `mtp` when that function accepts it. A row that omits `mtp` passes false. It records the owners created during that open, and sweeps them after `close`. An open that raises sweeps the owners that appeared during the call. A crash sweeps that pipeline's owners.

`POST /mcp` is the FastMCP media queue. The pipeline manager admits requests through the queues in `schedule.py`: embeddings, then chats that have not started, then media. Those ranks are the admission fields on `LanguagePipeline`, `EmbeddingsPipeline`, and `MediaPipeline`. A chat preempts media. `preempt` stays set until the chat queue is empty. A chat whose client leaves before the model is opened does not open a runtime. A chat that leaves without a reply is not reported as a pipeline failure. After a chat releases, a one-second debounce keeps media from admitting so the next model call (the reply to a tool result) can reuse the resident language model. An embeddings request does not set `preempt`. `reload` drops every waiter in both queues. A waiter does not run. The request that holds the GPU is cancelled, and the runtime closes. The pipeline manager checkpoints the media task and closes its device before the chat model allocates, resumes that same task only after the foreground queue has drained, and `POST /v1/dispatch` returns the shared media pack. `favlo media --pipeline <id>` calls that route on a listener that is already up. The listener is the socket. `pipelines` chooses the request in service and the runtime that serves it. One chat or media process stays resident. SIGINT and SIGTERM on the daemon are forwarded to each pipeline process. An embeddings process opens beside it for that request and stays warm for `embeddings_ttl_seconds` in `settings.json` (default 60; `0` closes when the request returns). `close` halts every registered process. The next call waits until the ledger says those pages are released. A reclaimable count that stops rising ends that wait and leaves the bytes still missing from that count as returning, so the next open counts those pages once. A pipeline crash fails the running job and leaves the daemon up. `POST /mcp` stays up with none loaded. It does not own chat history or tools. Favlo web owns those. A terminal `favlo serve` is the daemon and does not keep a session. A pipeline writes a reply and does not see a socket. `json` finishes one object, and that object is encoded before the call returns. `begin_events`, `event`, and `done` are one server-sent stream. Idle streamed Responses keep the socket with the same sse-starlette ping comments MAF AG-UI uses (`: keepalive` every 15 seconds). `stopped` is true when that client has gone. Telemetry is one JSON event per line. The pipeline process forwards those events to the daemon. Pipelines report through `begin`, `end`, and `note` on `favlo.console`. That process paints them as they arrive. Pretty folds one request burst into one durable line.

`python -m favlo.smoke_lock run` is how a pipeline test takes the GPU and gives it back. It closes the resident model when no request is open, runs the command, and resumes chat when that command exits. `release` is the same resume. `python -m favlo.smoke_lock attach` does not take that lock. It reuses `127.0.0.1:8129`, starts the daemon only when that port is closed, and stops it only when this call started it. When the named pipeline is resident and its sources are newer than that worker, `attach` closes the model first. The listener closes a resident worker on the next request when that pipeline's sources are newer than the worker, then loads it again. The daemon stays up. The recorded pid is the invoking shell. `uv` and `python` are not that owner. A command started from a shell that already holds the lock leaves the file in place and does not release it. Image, video, and music bytes stay on disk. The uncompressed AVI is `~/.favlo/videos/<id>.avi`. The MP4 is `<id>.mp4` beside it, encoded from that AVI. `video_ttl_seconds` in `settings.json` is how long it stays. The default is 86400. A sweep at startup and after each stored clip deletes a file older than that. The uncompressed RGB is `~/.favlo/images/<id>.rgb`. The PNG is `<id>.png` beside it, encoded from that file at the request's zlib level. Standardized metadata is `images/.favlo_metadata/<id>.png.json` (semantic fields plus a nested `generation` object for the seed, prompt, model, and other parameters). Video, music, and voice use the same `.favlo_metadata` layout beside their kind folders. An omitted seed is a random integer. A resumed job keeps the seed it drew. `image_ttl_seconds` defaults to 432000 and uses the same sweep. The WAV is `~/.favlo/music/<id>.wav`. The MP3 and FLAC are encoded from that WAV. `music_ttl_seconds` defaults to 86400 and uses the same sweep. The spoken WAV is `~/.favlo/voice/<id>.wav`. The MP3 and FLAC are encoded from that WAV. `voice_ttl_seconds` defaults to 86400 and uses the same sweep. A client reads a stored file with `resources/read`. A path under `/v1/images`, `/v1/videos`, or `/v1/audio` is 404. The listener note is [docs/serve.md](docs/serve.md).

## Runtimes

Each runtime is a separate write set. An agent developing one runtime edits that set and leaves the others untouched.

| Pipeline | Package | As-built note |
|---|---|---|
| `qwen38_27b` | `pipelines/qwen38_27b/metal41/` | [docs/qwen38_27b/README.md](docs/qwen38_27b/README.md) |
| `qwen36_35b_a3b` | `pipelines/qwen36_35b_a3b/metal41/` | [docs/qwen36_35b_a3b/README.md](docs/qwen36_35b_a3b/README.md) |
| `gemma4_26b_a4b` | `pipelines/gemma4_26b_a4b/metal41/` | [docs/gemma4_26b_a4b/README.md](docs/gemma4_26b_a4b/README.md) |
| `qwen35_4b` | `pipelines/qwen35_4b/metal41/` | [docs/qwen35_4b/README.md](docs/qwen35_4b/README.md) |
| `umt5_xxl` | `pipelines/umt5_xxl/metal41/` | [docs/umt5_xxl/README.md](docs/umt5_xxl/README.md) |
| `bge_m3` | `pipelines/bge_m3/metal41/` | [docs/bge_m3/README.md](docs/bge_m3/README.md) |

A media pipeline is `pipelines/<pipeline>/<runtime>/` and is registered by its module.

| Pipeline | Package | As-built note |
|---|---|---|
| `z_image_turbo` | `pipelines/z_image_turbo/metal41/` | [docs/z_image_turbo/README.md](docs/z_image_turbo/README.md) |
| `z_image_base` | `pipelines/z_image_base/metal41/` | [docs/z_image_base/README.md](docs/z_image_base/README.md) |

Z-Image-Turbo and Z-Image share the forward in `pipelines/z_image/metal41/`. That package has no `PIPELINE` assignment, so the command scan does not list it. Served pipelines import `z_image.metal41`. The note is [docs/z_image/README.md](docs/z_image/README.md).

Layout is model → runtime: `pipelines/<pipeline>/<runtime>/` is the importable package (`import qwen35_4b.metal41`). The model directory name is the pipeline id and the CLI flag stem. `PIPELINE` on the runtime package equals that id. The write set for one backend is `pipelines/<pipeline>/<runtime>/`, `docs/<pipeline>/`, and tests that import only that package. A new runtime under an existing model adds `pipelines/<pipeline>/<runtime>/` and does not edit another model's tree. Shipped rows use `runtime` `metal41` and `pipeline_root` `pipelines/<pipeline>/metal41`. A CUDA backend for the same model is `pipelines/<pipeline>/cuda/`. A missing or unknown `runtime` is an error.

Parked and failed pipelines are not in the actively developed tree. Measured claims that remain are under [docs/research/history/](docs/research/history/) and say the source exists only in git history.

Tests for the listener, the registry, and the console live in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_registry.py`, and `tests/test_console.py`. A pipeline's tests do not import another pipeline.

## Registry

`models/registry.json` is gitignored. A row stores the folder, `directory` (the quantized record that was written), the pipeline id, `runtime`, an alias, the BF16 source path, the key/value mode, the resident percent, `context_length`, `mtp`, `mtp_depth`, `enable_thinking`, optional `reasoning_effort`, and the sampling card copied from that pipeline's `sampling.py`. A row that omits `mtp` does not draft. A row that omits `mtp_depth` uses 3. A chat request that sets either field overrides the row. A row may store `parts`, named quantized directories such as `umt5`, `high`, and `low`. `serve` opens `directory` and passes `parts` to a runtime that accepts them. A row that omits `directory` opens `models/<folder>`. `runtime` is `metal41` or `cuda`. Every row stores `pipeline_root`. An omitted root is `pipelines/<pipeline>/<runtime>`. A relative root resolves against the checkout. `serve` imports `{pipeline}.{runtime}`. A row that omits `context_length` uses 131072. The stored names are the `:suffix` ids. A bare model id selects no sampling suffix. The pipeline applies its default sampling card, or the card the request selects. A suffix is a sampling preset only. Thinking follows OpenAI request fields, else `enable_thinking` / `reasoning_effort` on the row. `serve` does not register a record it finds on disk. A fresh tree has no rows. `GET /v1/models` lists every registered language and embeddings row from `registry.json`, including each sampling suffix that row stores, and each object includes `context_length`. The list does not import pipeline packages. A media row is omitted, and `GET /v1/models/{id}` for that id is 404. `POST /v1/embeddings` loads the pipeline that defines `handle_embed`. An embeddings request waits until the open chat completion returns, does not set `preempt`, and does not close the resident model. Its process closes when the request returns. A chat that has not started waits while an embeddings request is queued or running. `media` stores default folder ids for `image`, `i2v`, `t2v`, `music`, and `voice`. The first registered row for a kind fills an empty key. A later read fills a key that is still empty and saves it. `embeddings` stores the folder id `POST /v1/embeddings` uses when the body omits `model`. The first registered row whose pipeline defines `handle_embed` fills an empty key. A later read fills a key that is still empty and saves it.

## Layout

```text
pipelines/<model>/metal41/   Metal 4.1 implementation package for that model
pipelines/<model>/cuda/      CUDA implementation when that backend exists
src/favlo/              process entry; the pipeline manager loads the pipeline a request names
  cli.py                discovers model packages; a new model does not edit this
  web_host.py           starts the web API and the Expo app; it does not start the listener. It does not start when the registry has no language model. The first run stores that registry model as the application default
  registry.py           folders, aliases, pipelines, and sampling variants
  listener.py           the socket; a terminal runs this process
  http_app.py           ASGI routes for chat and MCP; writes the reply the pipeline manager returns
  schedule.py           foreground queue and media queue; the pipeline manager admits through them
  pipeline.py           daemon call every served seam extends
  language_pipeline.py  disconnect shell a served language model inherits
  embeddings_pipeline.py list object a served embedding model inherits
  media_pipeline.py     media shell; a subclass sets kind, id prefix, and object name
  mcp_server.py         FastMCP media queue, Tasks, and favlo:// reads
  reply.py              the object a pipeline writes; the server turns it into HTTP and fills chat usage
  responses.py          Responses helpers; listener Chat Completions translation (`chat_to_responses`, `ChatWire`, `ResponseWire`)
  usage.py              OpenAI token fields on a chat completion, including max_prompt_tokens
  daemon.py             spawn and stop the listener; no pipeline imports
  console.py            status row; no pipeline imports
  host_progress.py      CPU consumer between the GPU thread and the console queue
  progress.py           sanitizes one raw progress event before the console paints it
  chat_client.py        Agent Framework chat client; reasoning stays on reasoning_content. It keeps max_prompt_tokens and leaves older messages out at half that cap
  persona.py            horse persona
  config_tools.py       configuration tool group for the web
  prefix.py             SSD prefix store
  memory.py             device, placement heaps, and span handles; the pipeline manager sweeps an owner on close and on crash
  memory_notes.py       ~/.favlo/memory.db for MCP remember, recall, and forget
  pipelines.py          the request in service, the runtime that serves it, and that runtime's owners
  setup.py              empty-registry download
  logs.py               rolling text logs; each file stays within settings.json log_max_bytes
  videos.py             stored AVI and the MP4 encoded from it; video_ttl_seconds moves an older clip to Trash
  images.py             stored RGB and the PNG encoded from it; image_ttl_seconds moves an older pair to Trash
  music.py              stored WAV and the MP3 or FLAC encoded from it; music_ttl_seconds moves an older clip to Trash
  voice.py              stored spoken WAV and the MP3 or FLAC encoded from it; voice_ttl_seconds moves an older clip to Trash
  trash.py              NSFileManager move into the system Trash
web-api/                FastAPI process for favlo web, package web_api. The note is web-api/README.md
web-spa/                Expo web app. The note is web-spa/README.md
harness/                file agent, coding agent, needle agent, worktree agent, chat agent, and the three benchmarks
docs/                   as-built notes; see docs/README.md
```

Probe, matmul, kernels, bf16, complete, and quantize stay on that pipeline's flag: `favlo --qwen38_27b quantize`, and the same flag for the other commands. The listener runs one request. A language-model pipeline does not fuse a whole layer into one kernel. An image, video, music, or voice pipeline may place several kernels of one generation in one command buffer and may fuse those kernels. Prefill, decode, and the calibration prefill are Metal shaders. A missing kernel stays missing. The Neural Engine may run a module.

## Commands

```text
uv sync --extra dev
uv run pytest
uv run favlo serve
uv run favlo web
uv run favlo --qwen38_27b quantize
uv run favlo --qwen36_35b_a3b kernels
uv run favlo --qwen36_35b_a3b quantize
uv run python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness --model qwen38_27b
uv run python -m harness.benchmarks --model qwen38_27b
uv run python -m harness.coding --model qwen38_27b
uv run python -m harness.needle --model qwen36_35b_a3b --tokens 65536
uv run python -m harness.needle --model qwen36_35b_a3b --tokens 65536 --mode retrieve --depth 0.5
uv run python -m harness.worktree --model qwen36_35b_a3b
uv run python -m harness.image_smoke_agent --pipeline z_image_turbo --tool generate_image
uv run python -m harness.image_persona --pipeline z_image_turbo --tool generate_image
uv run python -m harness.video_smoke_agent --pipeline wan22_i2v --tool generate_video
uv run python -m harness.media_pipeline
uv run python -m harness.language_pipeline
uv run python -m harness.voice_smoke_agent --pipeline voxcpm2 --tool generate_voice
uv run python -m harness.music_long_agent --pipeline ace_step_15_turbo --tool generate_music
uv run python -m harness.embeddings_agent
uv run python -m harness.server status
```


What the harness scores is [docs/harness.md](docs/harness.md). A pipeline's quantize, probe, and matmul commands are that pipeline's note.
