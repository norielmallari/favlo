# AI-assisted features

These are the behaviors on `favlo serve` where a model writes or decides. Pipeline math, kernels, and quantize stay in that pipeline's note. A change to a feature on this page updates this folder in the same change.

The guides the configuration tools read are [../guides](../guides). The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md). The operator console is [../serve.md](../serve.md).

## Voice

The voice is a small horse who likes carrots. Favlo web stores that persona on the agent named Favlo. The configuration tool group adds the guide, setup, model, and download-reminder tools. Favlo web enables that group from an agent's details, and the Favlo agent has it on. The API note is [../../web-api/README.md](../../web-api/README.md). The app note is [../../web-spa/README.md](../../web-spa/README.md).

## Guides

`read_guide` returns one page from [../guides](../guides):

| Guide | What it covers |
|---|---|
| [clients](../guides/clients.md) | Server address, model field, `quit`, and `reload` |
| [models](../guides/models.md) | Download, quantize, and registration for the supported checkpoints |
| [history](../guides/history.md) | Where the name Favlo came from |

A question the guide does not answer stays unanswered.

## Setup guides

`read_setup` returns `docs/<pipeline>/SETUP.md`. The argument is the pipeline id, the same stem as `favlo --<pipeline>`. `qwen36-35b-a3b` is Qwen3.6-35B-A3B. `qwen38-27b` is Qwen3.8-27B. `z-image-turbo` is Z-Image-Turbo. `wan22-i2v` is Wan 2.2 I2V. `ace-step-15-turbo` is ACE-Step. The chat calls it when someone wants to set up, download, quantize, or delete that model. A pipeline with no setup page is an error that names the pipelines that have one. The page is not part of the standing instructions. It is loaded when the chat asks for it.

## Setup commands

`run_setup` runs one command from that page. The working directory is the checkout where Favlo is installed. `FAVLO_ROOT` is that directory. The command is not a shell: there is no `cd`, no pipe, and a path has to stay inside the checkout. It accepts `uv sync --extra dev` and `uv run favlo --<pipeline>` with `setup`, `quantize`, `quantize-text`, `delete`, `download`, or `models`, and the flags those setup pages use. `setup`, `quantize`, `quantize-text`, `delete`, and `download` close the resident model first. `serve` is not one of these commands.

## Registered models

`list_models` reads `models/registry.json` and returns one line per record: the folder, the pipeline, the alias, and the key/value mode. An unregistered complete record is named on its own line. An empty registry returns `no models are registered`. The chat calls this when someone asks which models are registered. Download and quantize stay in the models guide.

## Session tools

Favlo web enables Session tools from an agent's details. The switch starts off, and the Favlo agent has it on. The tool is `clear_session`. Each turn builds a new in-memory Agent Framework session and copies the stored messages into it. That session is not written to the database. The user message and the reply are. `clear_session` deletes the stored messages for the session the agent is in, drops finished runs on that session, and clears the in-memory session for the turn in progress. The open chat drops the transcript when that clear arrives.

## Conversation tools

Favlo web enables Conversation tools from an agent's details. The switch starts off. The tools are `list_conversations`, `search_messages`, and `read_recent_messages`. List names the stored agent sessions, channels, and workspace chats. Search matches message text and returns a short snippet with the conversation id. Read returns the latest messages from that conversation, oldest first. The read count is capped at 50 and the search count at 20. While the switch is on, the tools read every one of those conversations.

## MCP media

Favlo web connects an agent to an MCP server from that agent's details. The transport is HTTP or stdio. Test connection calls `tools/list` on MCP `2026-07-28` and stores the catalog. The client advertises `io.modelcontextprotocol/tasks`. `generate_image`, `generate_video_i2v`, `generate_video_t2v`, `generate_music`, and `generate_voice` then return a task handle. The web API polls `tasks/get` at `pollIntervalMs` until the task is completed, failed, or cancelled, and writes that result into the agent session that called the tool. A failed result, including a completed task whose text is a tool error, starts another turn in that session. The turn instructions require `generate_image`, `generate_video_i2v`, `generate_video_t2v`, `generate_music`, and `generate_voice` to pass a model id returned by `list_image_models`, `list_video_models_i2v`, `list_video_models_t2v`, `list_music_models`, or `list_voice_models`. That follow-up turn calls the matching list tool and retries the generate tool with an id from that list. The turn tells the user it is generating that file. It does not write a `favlo://` link, an image id, or a claim that the file is already ready. That file arrives in a later message. A disabled server or a disabled tool group is left out of the next turn. A task already running still finishes.

## Shared memory

`POST /mcp` lists `remember`, `recall`, and `forget`. They read and write `~/.favlo/memory.db`. A recall ranks stored sentences with the BGE-M3 dense vector from `POST /v1/embeddings`. Hermes and other MCP clients call the same three tools on the listener. Favlo web can enable that group on an agent. The store is not injected into anyone's prompt.
