# Web API

FastAPI process for the Favlo web workspace. It keeps agents, conversations, heartbeats, and MCP servers, and it asks the Favlo listener for chat completions. This directory is part of the Favlo repository.

The browser is [../web-spa/README.md](../web-spa/README.md). The process map is [../ARCHITECTURE.md](../ARCHITECTURE.md).

## Getting started

`favlo web` starts this API and the Expo app together. The listener stays a separate process. From a checkout of this repository, with at least one language model registered:

```bash
cd web-spa && npm install
uv run favlo serve --non-interactive
uv run favlo web
```

The API listens on `127.0.0.1:8130`. The app listens on `127.0.0.1:8081`. Stopping `favlo web` stops both children. A port that is already taken ends the command and names that port. The first run stores the registry's language model as the application default.

## Configuration

| Item | Value |
|---|---|
| Database | `~/.favlo/web/app.db` |
| Home override | `FAVLO_WEB_HOME` replaces `~/.favlo/web` |
| Listener | `http://127.0.0.1:8129/v1` |
| Browser origin | `http://127.0.0.1:8081` |

The listener is a provider by default. Refresh models reads that provider's models endpoint again. That list is the listener's language models. Media models stay on the media MCP tools. Chat completions go to the listener. When a completion includes reasoning, that text is stored beside the reply. The next turn sends that reasoning as `reasoning_content` with the reply. The browser shows the reasoning as Thinking. A completion that stops inside thinking stores that reasoning and reports that the reply was empty.

## Agents and tools

A new database includes an agent named Favlo. Its persona is the small horse. Each agent has one Agent Channel and one long-lived channel session. Creating an agent stores the name, the persona, the chosen model, and the chosen tool groups. Deleting an agent removes that agent, its channel, the messages and runs on its sessions, its heartbeats, and its MCP servers. Favlo stays. An agent's saved provider and model override the application default. Until that choice is saved, the agent uses the application default.

Tool groups are one switch each. They start off, except the configuration group and the session group on Favlo.

| Group | Tools |
|---|---|
| Configuration tools | `config_read_guide`, `config_read_setup`, `config_run_setup`, `config_list_models`, `config_set_model_mtp`, `config_download_suggestions` |
| Conversation tools | `conversation_list_conversations`, `conversation_search_messages`, `conversation_read_recent_messages` |
| Heartbeat tools | `heartbeat_list_heartbeats`, `heartbeat_add_heartbeat`, `heartbeat_update_heartbeat`, `heartbeat_delete_heartbeat` |
| Session tools | `session_clear_session` |

Each turn builds a new in-memory Agent Framework session and copies the stored messages into it. That session is not written to the database. The user message and the reply are. `POST /runs/{id}/cancel` closes that stream. The listener sees the disconnect and stops the completion. A chat run left in progress when the API starts is marked failed. `session_clear_session` deletes the stored messages for the session the agent is in, drops finished runs on that session, and clears the in-memory session for the turn in progress. Conversation tools read every agent session, channel, and workspace chat. A search returns a short snippet. A read returns the latest messages, oldest first. The read count is capped at 50 and the search count at 20. Heartbeat tools store, change, and remove a schedule for that agent only. The configuration group and the session group are on for Favlo. The note is [../docs/ai-assistance/README.md](../docs/ai-assistance/README.md).

## Heartbeats

A heartbeat stores a five-field cron expression, an IANA timezone, instructions, and whether it is enabled. The smallest interval is one minute. The cron is read in that timezone. Stored timestamps stay in UTC.

The API checks enabled heartbeats when it starts and again at each minute. A schedule that matches that minute starts a run in the heartbeat's own session. A minute that passed while the API was down is not run later. A heartbeat whose previous run is still going skips that minute. A run left in progress when the API starts is marked failed. Run heartbeat starts that schedule immediately. Two heartbeats can run at the same time.

## MCP

An MCP server is HTTP or stdio. HTTP takes a URL. Stdio takes a command. Test connection calls `tools/list` on MCP `2026-07-28` and returns the live tools. The Favlo listener address and its `/v1` prefix are read at `/mcp`. Save stores the server and which groups are on. No table holds a tool name, description, or input schema. Each read uses a process-local cache that expires after 60 seconds, and a connection test replaces that entry immediately. A second reader of the same connection waits for the fetch already in flight. A failed fetch is not cached.

Favlo media tools are grouped as Image, Video, Music, Voice, Media tasks, and Memory. While a group is on, its tools are available on that agent's turns. Each MCP tool name on a turn starts with the server name, then the catalog name. The client advertises `io.modelcontextprotocol/tasks`. `generate_image`, `generate_video_i2v`, `generate_video_t2v`, `generate_music`, and `generate_voice` then return a task handle. The API polls `tasks/get` at the server's interval until the task is completed, failed, or cancelled, and writes that result into the agent session that called the tool. A failed result, including a completed task whose text is a tool error, starts another turn in that session. Those generate tools use the registry default for that media type. To change a model, the turn calls the matching list tool, then `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, or `set_voice_model`, and repeats the returned session id in the reply. A later generate call passes that id as `session`. An omitted `session` uses the registry default. A failed result tells the turn to call the matching list and set tools and retry the generate tool with that session. The turn tells the user it is generating that file. It does not write a `favlo://` link, an image id, or a claim that the file is already ready. That file arrives in a later message. A disabled server or a disabled group is left out of the next turn. A task already running still finishes.

## Stored media

A `favlo://` image, video, or audio link names a file the listener already stored. `GET /media/{kind}/{filename}` streams that file. Kinds are `images`, `videos`, `voice`, and `music`. The app shows the file in the chat. The screen is [../web-spa/README.md](../web-spa/README.md).

## Gallery workspace

`GET /profile` returns `{ profile, folder_kinds }`. On `favlo web`, `profile` is `local`. `local` is enabled. `cloud` / `onedrive` and `cloud` / `dropbox` are listed and disabled.

A gallery workspace belongs to a channel. Creating one opens a workspace session. That session is not the home channel session. `POST /workspaces/{id}/root/pick` opens the macOS folder dialog on this machine and returns that directory. It does not save the root. The browser does not read the disk. `PUT /workspaces/{id}/root` stores one root folder. `kind: local` needs an existing directory on this machine. `kind: cloud` is rejected and names the subtype. Until the root is saved, image and folder routes return an error. After it is saved, list, read, and write stay inside that directory. A path that leaves it is rejected. Deleting the workspace deletes the root row and the workspace session. It does not delete the files.

A turn on that workspace session adds a standing prompt for the workspace type. Gallery tells the agent it is operating in a gallery workspace. Coding tells the agent it is operating in a coding workspace. The home channel does not receive either. Skills are appended as markdown: `# Skills`, then each skill name as `##` and its description. `GET /agent-skills` lists skills stored on agents. `POST /agent-skills` stores a name and a description for one agent. Those skills are on the home channel and on a workspace session for that agent. `POST /workspaces/{id}/skills` stores a name and a description on the workspace. A workspace session appends the agent skills, then the workspace skills. `PATCH` and `DELETE` edit or remove one skill. A name is unique on that agent or that workspace. On a gallery session after the root is saved, the runtime adds `gallery_list_folder`, `gallery_write_image`, `gallery_create_folder`, `gallery_rename_folder`, `gallery_move_folder`, and `gallery_delete_folder`. The home channel does not get those tools. `gallery_write_image` takes a relative path and either base64 `bytes` or a `favlo://images/{id}.png` source. The note is [../docs/ai-assistance/README.md](../docs/ai-assistance/README.md).

## Development

The importable package is `web_api`. Tests live in the repository `tests/` directory:

```bash
uv run pytest tests/test_web_api.py tests/test_web_folder_pick.py tests/test_web_gallery.py tests/test_web_heartbeat.py tests/test_web_host.py tests/test_web_mcp.py tests/test_web_media.py
```

## Problems

Report a problem in this repository's issue tracker.

## License

The license is the Apache License, Version 2.0. The terms are in [../LICENSE.md](../LICENSE.md).
