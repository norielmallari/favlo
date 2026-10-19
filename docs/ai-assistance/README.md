# AI-assisted features

These are the behaviors on `favlo serve` where a model writes or decides. Pipeline math, kernels, and quantize stay in that pipeline's note. A change to a feature on this page updates this folder in the same change.

The guides the terminal reads are [../guides](../guides). The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

## Terminal chat

On a terminal, `favlo serve` is the CLI. The server is a daemon. The CLI reads stdin and answers through that daemon's `POST /v1/chat/completions`. The model field is the first folder in the registry. One Microsoft Agent Framework session keeps that history. A line is sent when it is ready. The prompt is `you:`. It starts on a new line after that turn's telemetry. The reply is `favlo:`, and the tokens appear as they arrive. The terminal wraps that text. That reply finishes before its telemetry is printed. A tool call and a tool result are not printed. `/quit` stops the daemon. Ctrl+C stops the daemon when this terminal started it. Closing the terminal leaves the daemon running. `favlo serve --non-interactive` starts the daemon and exits.

## Voice

The voice is a small horse who likes carrots. A line that matches the violent, sexual, or obscene list is answered from a fixed refusal and is not sent to the model.

## Opening line

The first start, when `~/.favlo/intros.json` has no lines, sends one chat completion immediately. That request includes the persona and asks for several introductions, each with emojis. `tool_choice` names `save_intro`. The daemon opens the load row when it reads the record. The first start prints the first saved line. Each later start prints the next line, then the first again, and does not call the model until someone types. The line just shown is stored as the assistant's first turn.

## Guides

`read_guide` returns one page from [../guides](../guides):

| Guide | What it covers |
|---|---|
| [clients](../guides/clients.md) | Server address, model field, `/quit`, the opening line, the idle check, and download reminders |
| [models](../guides/models.md) | Download, quantize, and registration for the supported checkpoints |
| [history](../guides/history.md) | Where the name Favlo came from |

A question the guide does not answer stays unanswered.

## Registered models

`list_models` reads `models/registry.json` and returns one line per record: the folder, the pipeline, the alias, and the key/value mode. An unregistered complete record is named on its own line. An empty registry returns `no models are registered`. The chat calls this when someone asks which models are registered. Download and quantize stay in the models guide.

## Idle download suggestion

After 30 minutes with no completion from this terminal and no typing, one session checks the machine. `machine` reports physical memory and free disk. When the machine has at least 16 GB and the volume can hold the published files, `download_qwen38_27b` starts `favlo --qwen38-27b download` into `models/Qwen3.8-27B-bf16`. The line asks whether to remind them again.

`download_suggestions` stores the answer in `~/.favlo/suggestions.json`. `enabled` false stops later idle suggestions. `enabled` true turns them back on. A missing file leaves reminders on. A machine that is not ready, or a checkout that is already downloading or present, does not call the model and does not print. Packing the record stays `favlo --qwen38-27b quantize`.
