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

The API listens on `127.0.0.1:8130`. The app listens on `127.0.0.1:8012`. Stopping `favlo web` stops both children. A port that is already taken ends the command and names that port. The first run stores the registry's language model as the application default.

## Configuration

| Item | Value |
|---|---|
| Database | `~/.favlo/web/app.db` |
| Home override | `FAVLO_WEB_HOME` replaces `~/.favlo/web` |
| Listener | `http://127.0.0.1:8129/v1` |
| Browser origin | `http://127.0.0.1:8012` |

The listener is a provider by default. Refresh models reads that provider's models endpoint again. That list is the listener's language models. Media models stay on the media MCP tools. Chat goes to the listener on the Responses route. An agent's model settings may set `reasoning_effort`, `preserve_thinking`, and `expand_thinking` and `expand_tools` (`hidden`, `collapsed`, `running`, or `always`). Omitted thinking stays collapsed. Omitted tool calls stay hidden, so that block is left out and the activity chip says Working. `hidden` does the same for whichever block it is set on. Providers may set `reasoning_effort` too. The agent value wins. Not set on both leaves that field off the request. A set effort becomes Responses `reasoning.effort` (`none`, `low`, `medium`, `high`, `xhigh`, or `max`). Thinking follows that field, else the selected model's registry `enable_thinking`, not the `:thinking` / `:instruct` sampling suffix. A legacy `enable_thinking` value still maps when effort is unset (`none` or `medium`). `preserve_thinking` stays on the body. Qwen3.6 then keeps historical reasoning, so the next prompt extends the stored prefix. A completion's `content` and `tool_calls` are stored as the framework returned them. When a completion includes reasoning, that text is stored beside the reply and the browser shows it as Thinking. Each tool call is stored on that message and shown in its own row. The stored turn keeps thinking, tool calls, and the reply in the order they arrived. Thinking closes when a tool call or the reply starts, and a tool call closes when thinking or the reply starts. The next turn sends that reasoning as `reasoning_content` with the reply. A completion with reasoning and no reply text still finishes the turn.

## Agents and tools

A new database includes an agent named Favlo. Its persona is the small horse. Each agent has one Agent Channel and one long-lived channel session. Creating an agent stores the name, the persona, the chosen model, reasoning effort, preserve thinking, and the chosen tool groups, and attaches its own Favlo media server. Not set stores null and leaves that field off the chat request. Deleting an agent removes that agent, its channel, the messages and runs on its sessions, its heartbeats, and its MCP servers. Favlo stays. An agent's saved provider and model override the application default. Until that choice is saved, the agent uses the application default. `favlo web` stores the registry default once when the database has none. That is the model id `favlo serve` uses when a request names none. Chat and gallery vision use the agent's provider and model, and the settings default when the agent has none. A missing or disabled choice raises. A new database also attaches one HTTP MCP server named Favlo media on Favlo, pointed at `http://127.0.0.1:8129/mcp`, with every media group left on.

Tool groups are one switch each. They start off, except the configuration group and the session group on Favlo.

| Group | Eager | After loader |
|---|---|---|
| Configuration tools | `config_load_tools` | `config_read_guide`, `config_read_setup`, `config_run_setup`, `config_list_models`, `config_set_model_mtp`, `config_download_suggestions` |
| Conversation tools | `conversation_load_tools` | `conversation_list_conversations`, `conversation_search_messages`, `conversation_read_recent_messages` |
| Heartbeat tools | `heartbeat_load_tools` | `heartbeat_list_heartbeats`, `heartbeat_add_heartbeat`, `heartbeat_update_heartbeat`, `heartbeat_delete_heartbeat` |

Each chat stores one Agent Framework session. A turn loads that session and sends the new user message through the framework's OpenAI client, which also sends the messages that session already holds. The agent persists that session after every model call and sets `store` false, so a tool round sends the same full transcript. The request does not send `max_tokens`. The listener stops on a stop id or at the last context position. The session is written back when the turn finishes. When that completion's prompt is at least half of `max_prompt_tokens`, older messages are left out of the next turn. A cancelled turn leaves the stored session as it was. The user message and the reply are still stored for the transcript. `POST /sessions/{id}/cancel` stops every running turn on that chat (and still accepts `POST /runs/{id}/cancel` for one run). The Agent Framework stream and chat client close; the listener sees the disconnect and stops the completion. A chat run left in progress when the API starts is marked failed. A failed reply stores the reason as a sentence. A listener exception is not stored as a class name. Typing `/session clear` deletes the messages, the runs, and the stored Agent Framework session for the session that received it. Typing `/session archive` starts a turn on that session whose prompt asks the agent to store worthwhile memories, and deletes that session, including its Agent Framework session, when the turn stops. A failed or cancelled archive leaves the session. Neither command changes any other session. Conversation tools read every agent session, channel, and workspace chat. A search returns a short snippet. A read returns the latest messages, oldest first. The read count is capped at 50 and the search count at 20. Heartbeat tools store, change, and remove a schedule for that agent only. The configuration group is on for Favlo. The note is [../docs/ai-assistance/README.md](../docs/ai-assistance/README.md).

## Heartbeats

A heartbeat stores a five-field cron expression, an IANA timezone, instructions, and whether it is enabled. The smallest interval is one minute. The cron is read in that timezone. Stored timestamps stay in UTC.

The API checks enabled heartbeats when it starts and again at each minute. A schedule that matches that minute starts a run in the heartbeat's own session. A minute that passed while the API was down is not run later. A heartbeat whose previous run is still going skips that minute. A run left in progress when the API starts is marked failed. Run heartbeat starts that schedule immediately. Two heartbeats can run at the same time.

## MCP

An MCP server is HTTP or stdio. HTTP takes a URL. Stdio takes a command. Test connection calls `tools/list` on MCP `2026-07-28` and returns the live tools. The Favlo listener address and its `/v1` prefix are read at `/mcp`. Save stores the server and which groups are on. No table holds a tool name, description, or input schema. Each read uses a process-local cache that expires after 60 seconds, and a connection test replaces that entry immediately. A second reader of the same connection waits for the fetch already in flight. A failed fetch is not cached. A new database seeds Favlo media on Favlo as above, and creating an agent attaches the same server with its own id. A later edit or delete of one copy stays on that agent.

Favlo media tools are grouped as Image, Video, Music, Voice, Media tasks, and Memory. While a group is on, its tools stay in that server's `allowed_tools`. A turn passes each enabled server to the agent as an Agent Framework MCP server with progressive disclosure. Eager remotes are image list/set/generate, memory, and `media_status` / `media_cancel` / `read_media`. Video, music, and voice load through that server's `load_tool`. Session column `loaded_mcp` keeps those remotes until `/session clear`. Each tool name starts with the server name, then the catalog name. `remember`, `recall`, and `forget` receive this agent as `client`. A generate call returns when the task is accepted. A later turn includes each `favlo://` link and leaves out the JSON. A failed result, including a completed task whose text is a tool error, tells that turn to retry. Those generate tools use the registry default for that media type. To change a model, the turn calls the matching list tool, then `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, or `set_voice_model`, and repeats the returned session id in the reply. A later generate call passes that id as `session`. An omitted `session` uses the registry default. A failed result tells the turn to call the matching list and set tools and retry the generate tool with that session. The accepting turn tells the user that file is being generated. A disabled server or a disabled group is left out of the next turn. A turn enters an MCP client for each enabled server, with `io.modelcontextprotocol/tasks`, and passes that client to the framework tool. The client negotiates MCP `2026-07-28`. A generate tool runs as a task inside that client. The call returns when the task is accepted, and a later turn brings the finished file.

## Stored media

A `favlo://` image, video, or audio link names a file the listener already stored. `GET /media/{kind}/{filename}` streams that file. Kinds are `images`, `videos`, `voice`, and `music`. The app shows the file in the chat. The screen is [../web-spa/README.md](../web-spa/README.md).

## Workspaces

`POST /workspaces` takes `type` `gallery` or `coding`. Chat turns use `POST /ag-ui` (AG-UI POST plus SSE). Record ids are GUIDs. A turn freezes the definition for that chat until clear. `GET /sessions/{id}` includes `state` for the workspace shell. Clear and archive mint a new chat GUID for the slot.

`GET /shells` lists the workspace kinds stored in the database (`shell`, `catalog_element`, `shell_element`). Creating a workspace sends one of those ids as `type`. The list icon and the page renderer come from that row. Create stores `provisioning_state` `new`, the response is `provisioning`, and a background job runs. A gallery job writes a path sidecar for every media file, captions every image with the workspace agent's model, listing every visible human anatomical feature, (the Providers default when that agent has no model), then the opening placement puts every image on the filmstrip and selects the first. The strip shows the image. The sidecar caption stays in the placement. That call uses the same clamps as `workspace_place` (60 images, unique paths, 80-character captions). The greeting is one message and does not call tools. A coding job only greets. The cue is not stored as a user message. The gallery stays `provisioning` until every image has a caption. A failure after that still sets `provisioned` and leaves that failure in the chat. `workspace.provisioned` is published on the app stream and the session stream. A **gallery** workspace is generative UI: closed catalog elements (`preview`, `filmstrip`), agent placement via `workspace_place`, local-folder bind, metadata index, and gallery tools. The session snapshot's first placement comes from that shell's element rows. A new gallery starts on `rose-pine`. `PUT /workspaces/{id}/theme` stores `favlo`, `rose-pine`, or `catppuccin` and publishes `workspace.theme_set`. The gallery session also exposes `gallery_set_theme`. Appearance (system, light, dark) stays the browser preference. As-built detail is in the closed tree: `favlo-cloud/docs/maf/curated-gallery.md`.

A **coding** workspace reuses the same runner without gallery tools or the shell. Skills (global catalog, agent copies, workspace copies) append as markdown on the turn; seeding and copy routes stay under `/global-skills`, `/agent-skills`, and `/workspaces/{id}/skills`. `GET /profile` reports `local` on `favlo web`. Folder pick and media file routes stay under `/workspaces/{id}/…`. Deleting a workspace removes its rows and session and leaves files on disk.

## Development

The importable package is `web_api`. Tests live in the repository `tests/` directory:

```bash
uv run pytest tests/test_web_api.py tests/test_web_agui.py tests/test_web_definition.py tests/test_web_folder_pick.py tests/test_web_gallery.py tests/test_web_gallery_index.py tests/test_web_gallery_metadata.py tests/test_web_guid_migration.py tests/test_web_heartbeat.py tests/test_web_host.py tests/test_web_mcp.py tests/test_web_media.py tests/test_web_placement.py
```

## Problems

Report a problem in this repository's issue tracker.

## License

The license is the Apache License, Version 2.0. The terms are in [../LICENSE.md](../LICENSE.md).
