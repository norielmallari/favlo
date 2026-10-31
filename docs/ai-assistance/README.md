# AI-assisted features

These are the behaviors on `favlo serve` where a model writes or decides. Pipeline math, kernels, and quantize stay in that pipeline's note. A change to a feature on this page updates this folder in the same change.

The guides the configuration tools read are [../guides](../guides). The process contract is [../../ARCHITECTURE.md](../../ARCHITECTURE.md). The listener is [../serve.md](../serve.md).

## Voice

The voice is a small horse who likes carrots. Favlo web stores that persona on the agent named Favlo. The configuration tool group adds the guide, setup, model, and download-reminder tools. Favlo web enables that group from an agent's details, and the Favlo agent has it on. The API note is [../../web-api/README.md](../../web-api/README.md). The app note is [../../web-spa/README.md](../../web-spa/README.md).

## Guides

`config_read_guide` returns one page from [../guides](../guides):

| Guide | What it covers |
|---|---|
| [clients](../guides/clients.md) | Server address, model field, and `POST /v1/pipelines/reload` |
| [models](../guides/models.md) | Download, quantize, and registration for the supported checkpoints |
| [history](../guides/history.md) | Where the name Favlo came from |

A question the guide does not answer stays unanswered.

## Setup guides

`config_read_setup` returns `docs/<pipeline>/SETUP.md`. The argument is the pipeline id, the same stem as `favlo --<pipeline>`. `qwen36-35b-a3b` is Qwen3.6-35B-A3B. `qwen38-27b` is Qwen3.8-27B. `z-image-turbo` is Z-Image-Turbo. `wan22-i2v` is Wan 2.2 I2V. `ace-step-15-turbo` is ACE-Step. The chat calls it when someone wants to set up, download, quantize, or delete that model. A pipeline with no setup page is an error that names the pipelines that have one. The page is not part of the standing instructions. It is loaded when the chat asks for it.

## Setup commands

`config_run_setup` runs one command from that page. The working directory is the checkout where Favlo is installed. `FAVLO_ROOT` is that directory. The command is not a shell: there is no `cd`, no pipe, and a path has to stay inside the checkout. It accepts `uv sync --extra dev` and `uv run favlo --<pipeline>` with `setup`, `quantize`, `quantize-text`, `delete`, `download`, or `models`, and the flags those setup pages use. `setup`, `quantize`, `quantize-text`, `delete`, and `download` close the resident model first. `serve` is not one of these commands.

## Registered models

`config_list_models` reads `models/registry.json` and returns one line per record: the folder, the pipeline, the alias, the key/value mode, and whether MTP is on with its draft depth. An unregistered complete record is named on its own line. An empty registry returns `no models are registered`. The chat calls this when someone asks which models are registered. `config_set_model_mtp` stores `mtp` and `mtp_depth` on that row. The next completion uses the row when the request omits those fields. A request field still wins. A row that omits `mtp` does not draft. A row that omits `mtp_depth` uses 3. Download and quantize stay in the models guide.

## Gallery tools

A workspace session adds a standing prompt for that workspace type. Gallery says the agent is operating in a gallery workspace. Coding says the agent is operating in a coding workspace. The home channel does not receive either. A new gallery workspace receives the `manage-gallery` skill from `web-api/web_api/skills-workspace/gallery`. That skill saves each generated image into the open folder with `gallery_write_image`. A gallery workspace injects folder tools only on that workspace session, and only after the user has saved a local folder. The tools are `gallery_list_folder`, `gallery_write_image`, `gallery_create_folder`, `gallery_rename_folder`, `gallery_move_folder`, `gallery_move_image`, `gallery_delete_image`, and `gallery_delete_folder`. The turn names that folder and, when an image is selected, that image. Paths are relative to that local folder. The home channel session does not receive them. The agent does not add or delete a folder row, and it cannot see a cloud row. `gallery_write_image` copies a `favlo://` image from the listener into the gallery folder. The API note is [../../web-api/README.md](../../web-api/README.md). The app note is [../../web-spa/README.md](../../web-spa/README.md).

## Skills

A skill is a name, a description, and a definition. Configuration stores a catalog of global skills. A global skill is not added to a prompt until it is copied. Add to new agents copies that skill onto an agent created later. Skills in `skills-global` follow the [Agent Skills](https://agentskills.io/specification) file: one directory, `SKILL.md`, YAML frontmatter for `name` and `description`, and the Markdown body as the definition. They are seeded into the catalog with that switch on, and onto the Favlo agent. `session-clear` asks whether to store the discussion before `session_clear_session`. When the answer is yes, it follows `session-remember`, which uses `remember`, `recall`, and `forget`. Copy places the name, description, and definition on an agent or a workspace. The copy has its own revision, and a later edit or delete of the global skill leaves the copy. A turn appends the copies that apply as one markdown block: `# Skills`, then each skill as `##` plus its name, then the description, then the definition. A skill with no definition injects its description, which is the whole skill. Agent skills come first. An empty list adds no heading. The home channel receives that agent's skills. A workspace session receives that agent's skills and the workspace skills. The API note is [../../web-api/README.md](../../web-api/README.md). The app note is [../../web-spa/README.md](../../web-spa/README.md).

## Session tools

Favlo web enables Session tools from an agent's details. The switch starts off, and the Favlo agent has it on. The tool is `session_clear_session`. Each turn builds a new in-memory Agent Framework session and copies the stored messages into it. That session is not written to the database. The user message and the reply are. Stop closes that stream so the listener drops the completion. `session_clear_session` deletes the stored messages for the session the agent is in, drops finished runs on that session, and clears the in-memory session for the turn in progress. The open chat drops the transcript when that clear arrives.

## Conversation tools

Favlo web enables Conversation tools from an agent's details. The switch starts off. The tools are `conversation_list_conversations`, `conversation_search_messages`, and `conversation_read_recent_messages`. List names the stored agent sessions, channels, and workspace chats. Search matches message text and returns a short snippet with the conversation id. Read returns the latest messages from that conversation, oldest first. The read count is capped at 50 and the search count at 20. While the switch is on, the tools read every one of those conversations.

## MCP media

Favlo web connects an agent to an MCP server from that agent's details. The transport is HTTP or stdio. Test connection calls `tools/list` on MCP `2026-07-28`. The catalog stays in the API process and is not stored in a table. On a turn, each MCP tool name starts with the server name, then the catalog name. The client advertises `io.modelcontextprotocol/tasks`. `generate_image`, `generate_video_i2v`, `generate_video_t2v`, `generate_music`, and `generate_voice` then return a task handle. The web API polls `tasks/get` at `pollIntervalMs` until the task is completed, failed, or cancelled, and starts another turn with that result as the prompt. The result is not a chat message. The model's reply is. Those generate tools use the registry default for that media type. To change a model, the turn calls the matching list tool, then `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, or `set_voice_model`, and repeats the returned session id in the reply. A later generate call passes that id as `session`. An omitted `session` uses the registry default. A failed result tells the turn to call the matching list and set tools and retry the generate tool with that session. The turn tells the user it is generating that file. While the task is working it does not write a `favlo://` link, an image id, or a claim that the file is already ready. The later turn includes each `favlo://` link and leaves out the JSON. A disabled server or a disabled tool group is left out of the next turn. A task already running still finishes.

## Shared memory

`POST /mcp` lists `remember`, `recall`, and `forget`. They read and write `~/.favlo/memory.db`. A recall ranks stored sentences with the BGE-M3 dense vector from `POST /v1/embeddings`. Hermes and other MCP clients call the same three tools on the listener. Favlo web can enable that group on an agent. A turn from that agent stores, searches, and deletes only the sentences kept for that agent. The store is not injected into anyone's prompt.
