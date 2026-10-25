# AI-assisted features

These are the behaviors on `favlo serve` where a model writes or decides. Pipeline math, kernels, and quantize stay in that pipeline's note. A change to a feature on this page updates this folder in the same change.

The guides the terminal reads are [../guides](../guides). The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

## Terminal chat

On a terminal, `favlo serve` is the CLI. The server is a daemon. The CLI reads stdin and answers through that daemon's `POST /v1/chat/completions`. The model field is the first folder in the registry. One Microsoft Agent Framework session keeps that history. A line is sent when it is ready. The prompt is `you:`. It starts on a new line after that turn's telemetry. The reply is `favlo:`. Telemetry is painted as it arrives, including the load row. The reply is printed after that turn's requests have closed. The terminal wraps that text. A tool call and a tool result are not printed. `/quit` closes every open pipeline, then stops the daemon. Ctrl+C closes every open pipeline and exits the CLI. The daemon keeps listening. `/reload` closes every open pipeline, drops the imported pipeline modules, and stays in the chat. Closing the terminal leaves the daemon running. `favlo serve --non-interactive` starts the daemon and exits.

## Voice

The voice is a small horse who likes carrots. A line is sent to the model. The instructions tell the model to refuse violent, sexual, NSFW, erotic, and obscene requests, including fiction and hypotheticals, and sexual or romantic content. A refusal is one short sentence and does not describe the request.

## Opening line

The first start, when `~/.favlo/intros.json` has no lines, sends one chat completion immediately. That request includes the persona and asks for six different introductions, each with emojis. Fewer than four distinct lines is not stored. `tool_choice` names `save_intro`. The daemon opens the load row when it reads the record. The first start prints the first saved line. Each later start prints the next line, then the first again, and does not call the model until someone types. The line just shown is stored as the assistant's first turn.

## Guides

`read_guide` returns one page from [../guides](../guides):

| Guide | What it covers |
|---|---|
| [clients](../guides/clients.md) | Server address, model field, `/quit`, `/reload`, the opening line, the idle check, and download reminders |
| [models](../guides/models.md) | Download, quantize, and registration for the supported checkpoints |
| [history](../guides/history.md) | Where the name Favlo came from |

A question the guide does not answer stays unanswered.

## Setup guides

`read_setup` returns `docs/<pipeline>/SETUP.md`. The argument is the pipeline id, the same stem as `favlo --<pipeline>`. `qwen36-35b-a3b` is Qwen3.6-35B-A3B. `qwen38-27b` is Qwen3.8-27B. `z-image-turbo` is Z-Image-Turbo. `wan22-i2v` is Wan 2.2 I2V. `ace-step-15-turbo` is ACE-Step. The chat calls it when someone wants to set up, download, quantize, or delete that model. A pipeline with no setup page is an error that names the pipelines that have one. The page is not part of the standing instructions. It is loaded when the chat asks for it.

## Setup commands

`run_setup` runs one command from that page. The working directory is the checkout where Favlo is installed. `FAVLO_ROOT` is that directory. The command is not a shell: there is no `cd`, no pipe, and a path has to stay inside the checkout. It accepts `uv sync --extra dev` and `uv run favlo --<pipeline>` with `setup`, `quantize`, `quantize-text`, `delete`, `download`, or `models`, and the flags those setup pages use. `setup`, `quantize`, `quantize-text`, `delete`, and `download` close the resident model first. `serve` is not one of these commands.

## Registered models

`list_models` reads `models/registry.json` and returns one line per record: the folder, the pipeline, the alias, and the key/value mode. An unregistered complete record is named on its own line. An empty registry returns `no models are registered`. The chat calls this when someone asks which models are registered. Download and quantize stay in the models guide.

## Shared memory

`POST /mcp` lists `remember`, `recall`, and `forget`. They read and write `~/.favlo/memory.db`. A recall ranks stored sentences with the BGE-M3 dense vector from `POST /v1/embeddings`. Hermes and other MCP clients call the same three tools on the listener. The terminal chat registers the same names and uses that database. The store is not injected into anyone's prompt.

## Idle download suggestion

After 30 minutes with no completion from this terminal and no typing, one session checks the machine. `machine` reports physical memory and free disk. When the machine has at least 16 GB and the volume can hold the published files, `download_qwen38_27b` starts `favlo --qwen38-27b download` into `models/Qwen3.8-27B-bf16`. The line asks whether to remind them again.

`download_suggestions` stores the answer in `~/.favlo/suggestions.json`. `enabled` false stops later idle suggestions. `enabled` true turns them back on. A missing file leaves reminders on. A machine that is not ready, or a checkout that is already downloading or present, does not call the model and does not print. Packing the record stays `favlo --qwen38-27b quantize`. The log for that download is `~/.favlo/logs/qwen38-27b-download.log`. A pipeline heartbeat runs in the daemon after the request closes, not in this terminal.
