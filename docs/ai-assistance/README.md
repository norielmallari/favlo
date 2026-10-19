# AI-assisted features

These are the behaviors on `favlo serve` where a model writes or decides. Pipeline math, kernels, and quantize stay in that pipeline's note. A change to a feature on this page updates this folder in the same change.

The guides the terminal reads are [../guides](../guides). The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md).

## Terminal chat

On a terminal, `favlo serve` is the CLI. The server is a daemon. The CLI reads stdin and answers through that daemon's `POST /v1/chat/completions`. One Microsoft Agent Framework session keeps that history. A turn waits until no other request is open. The prompt is `you:`. `/quit` stops the daemon. Ctrl+C stops the daemon when this terminal started it. Closing the terminal leaves the daemon running. `favlo serve --non-interactive` starts the daemon and exits.

## Voice

The voice is a small horse who likes carrots. A line that matches the violent, sexual, or obscene list is answered from a fixed refusal and is not sent to the model.

## Opening line

The first start, when `~/.favlo/intros.json` has no lines, sends one chat completion immediately. That request includes the persona and asks for a longer introduction with emojis. The first saved line is the one shown. `tool_choice` names `save_intro`. The load row opens when that request is sent, and a daemon load begin continues that row. The stored first line is what the terminal prints. The request checks that the model can complete a chat. Later starts print that first line and do not call the model until someone types. The line is stored as the assistant's first turn.

## Guides

`read_guide` returns one page from [../guides](../guides):

| Guide | What it covers |
|---|---|
| [clients](../guides/clients.md) | Server address, model field, `/quit`, the opening line, the idle check, and download reminders |
| [models](../guides/models.md) | Download, quantize, and registration for the supported checkpoints |
| [history](../guides/history.md) | Where the name Favlo came from |

A question the guide does not answer stays unanswered.

## Idle download suggestion

After 30 minutes with no open request and no typing, one session checks the machine. `machine` reports physical memory and free disk. When the machine has at least 16 GB and the volume can hold the published files, `download_qwen38_27b` starts `favlo --qwen38-27b download` into `models/Qwen3.8-27B-bf16`. The line asks whether to remind them again.

`download_suggestions` stores the answer in `~/.favlo/suggestions.json`. `enabled` false stops later idle suggestions. `enabled` true turns them back on. A missing file leaves reminders on. A machine that is not ready, or a checkout that is already downloading or present, does not call the model and does not print. Packing the record stays `favlo --qwen38-27b quantize`.
