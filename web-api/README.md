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

The listener is a provider by default. Refresh models reads that provider's models endpoint again. That list is the listener's language models. Media models stay on the media MCP tools. Chat completions go to the listener. An agent's model settings may set `enable_thinking`, `preserve_thinking`, and `reasoning_effort`. Not set leaves that field off the request. Qwen3.6 then keeps historical reasoning, so the next prompt extends the stored prefix. A completion's `content` and `tool_calls` are stored as the framework returned them. When a completion includes reasoning, that text is stored beside the reply and the browser shows it as Thinking. Each tool call is stored on that message and shown in its own row. The stored turn keeps thinking, tool calls, and the reply in the order they arrived. Thinking closes when a tool call or the reply starts, and a tool call closes when thinking or the reply starts. The next turn sends that reasoning as `reasoning_content` with the reply. A completion with reasoning and no reply text still finishes the turn.

## Agents and tools

A new database includes an agent named Favlo. Its persona is the small horse. Each agent has one Agent Channel and one long-lived channel session. Creating an agent stores the name, the persona, the chosen model, thinking, preserve thinking, reasoning effort, and the chosen tool groups. Not set stores null and leaves that field off the chat request. Deleting an agent removes that agent, its channel, the messages and runs on its sessions, its heartbeats, and its MCP servers. Favlo stays. An agent's saved provider and model override the application default. Until that choice is saved, the agent uses the application default.

Tool groups are one switch each. They start off, except the configuration group and the session group on Favlo.

| Group | Tools |
|---|---|
| Configuration tools | `config_read_guide`, `config_read_setup`, `config_run_setup`, `config_list_models`, `config_set_model_mtp`, `config_download_suggestions` |
| Conversation tools | `conversation_list_conversations`, `conversation_search_messages`, `conversation_read_recent_messages` |
| Heartbeat tools | `heartbeat_list_heartbeats`, `heartbeat_add_heartbeat`, `heartbeat_update_heartbeat`, `heartbeat_delete_heartbeat` |

Each chat stores one Agent Framework session. A turn loads that session and sends the new user message through the framework's OpenAI client, which also sends the messages that session already holds. The agent persists that session after every model call and sets `store` false, so a tool round sends the same full transcript. The request does not send `max_tokens`. The listener stops on a stop id or at the last context position. The session is written back when the turn finishes. When that completion's prompt is at least half of `max_prompt_tokens`, older messages are left out of the next turn. A cancelled turn leaves the stored session as it was. The user message and the reply are still stored for the transcript. `POST /runs/{id}/cancel` closes that stream. The listener sees the disconnect and stops the completion. A chat run left in progress when the API starts is marked failed. A failed reply stores the reason as a sentence. A listener exception is not stored as a class name. Typing `/session clear` deletes the messages, the runs, and the stored Agent Framework session for the session that received it. Typing `/session archive` starts a turn on that session whose prompt asks the agent to store worthwhile memories, and deletes that session, including its Agent Framework session, when the turn stops. A failed or cancelled archive leaves the session. Neither command changes any other session. Conversation tools read every agent session, channel, and workspace chat. A search returns a short snippet. A read returns the latest messages, oldest first. The read count is capped at 50 and the search count at 20. Heartbeat tools store, change, and remove a schedule for that agent only. The configuration group is on for Favlo. The note is [../docs/ai-assistance/README.md](../docs/ai-assistance/README.md).

## Heartbeats

A heartbeat stores a five-field cron expression, an IANA timezone, instructions, and whether it is enabled. The smallest interval is one minute. The cron is read in that timezone. Stored timestamps stay in UTC.

The API checks enabled heartbeats when it starts and again at each minute. A schedule that matches that minute starts a run in the heartbeat's own session. A minute that passed while the API was down is not run later. A heartbeat whose previous run is still going skips that minute. A run left in progress when the API starts is marked failed. Run heartbeat starts that schedule immediately. Two heartbeats can run at the same time.

## MCP

An MCP server is HTTP or stdio. HTTP takes a URL. Stdio takes a command. Test connection calls `tools/list` on MCP `2026-07-28` and returns the live tools. The Favlo listener address and its `/v1` prefix are read at `/mcp`. Save stores the server and which groups are on. No table holds a tool name, description, or input schema. Each read uses a process-local cache that expires after 60 seconds, and a connection test replaces that entry immediately. A second reader of the same connection waits for the fetch already in flight. A failed fetch is not cached.

Favlo media tools are grouped as Image, Video, Music, Voice, Media tasks, and Memory. While a group is on, its tools are available on that agent's turns. A turn passes each enabled server to the agent as an Agent Framework MCP server. The framework connects that server for the run and loads its tools. Each tool name starts with the server name, then the catalog name. `remember`, `recall`, and `forget` receive this agent as `client`. A generate call returns when the task is accepted. A later turn includes each `favlo://` link and leaves out the JSON. A failed result, including a completed task whose text is a tool error, tells that turn to retry. Those generate tools use the registry default for that media type. To change a model, the turn calls the matching list tool, then `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, or `set_voice_model`, and repeats the returned session id in the reply. A later generate call passes that id as `session`. An omitted `session` uses the registry default. A failed result tells the turn to call the matching list and set tools and retry the generate tool with that session. The accepting turn tells the user that file is being generated. A disabled server or a disabled group is left out of the next turn. A turn enters an MCP client for each enabled server, with `io.modelcontextprotocol/tasks`, and passes that client to the framework tool. The client negotiates MCP `2026-07-28`. A generate tool runs as a task inside that client. The call returns when the task is accepted, and a later turn brings the finished file.

## Stored media

A `favlo://` image, video, or audio link names a file the listener already stored. `GET /media/{kind}/{filename}` streams that file. Kinds are `images`, `videos`, `voice`, and `music`. The app shows the file in the chat. The screen is [../web-spa/README.md](../web-spa/README.md).

## Gallery workspace

`GET /profile` returns `{ profile, folder_kinds }`. On `favlo web`, `profile` is `local`. `local` is enabled. `cloud` / `onedrive` and `cloud` / `dropbox` are listed. A cloud row can be saved while its provider stays disabled. Saving it does not sign in or list remote files.

A gallery workspace belongs to a channel and holds many folder rows. Creating one opens a workspace session. That session is not the home channel session. `GET /workspaces/{id}/folders` lists the rows at the workspace root. `parent_id` lists the rows inside a cloud folder. `POST /workspaces/{id}/folders` adds a row. `kind: local` needs an existing directory on this machine and stays at the workspace root. Its children are directories, not rows. `kind: cloud` stores a name, a provider, and an optional remote folder id. `POST /workspaces/{id}/folders/pick` opens the macOS folder dialog on this machine and returns that directory. It does not save the row. The browser does not read the disk. Image and directory routes are under `/workspaces/{id}/folders/{folder_id}/` and stay inside that local directory. A path that leaves it is rejected. `POST .../images/move` moves an image inside that directory. `DELETE .../images` deletes one. `DELETE .../directories` deletes a directory only when it is empty. `PATCH /workspaces/{id}/folders/{folder_id}` with `move: true` changes a cloud row's parent. A local row moved onto a cloud row is refused, and a cloud row moved onto a local row is refused. Deleting a workspace deletes its folder rows and its workspace session. It does not delete the files.

Creating a gallery workspace copies `manage-gallery` from `skills-workspace/gallery` onto that workspace. A turn on that workspace session adds a standing prompt for the workspace type. Gallery tells the agent it is operating in a gallery workspace. Coding tells the agent it is operating in a coding workspace. The home channel does not receive either. Skills are appended as markdown: `# Skills`, then each skill name as `##`, its description, and its definition. A skill with no definition injects the description alone. `GET /global-skills` lists catalog skills. A catalog skill is a name, a description, a definition, and `add_by_default`. It is not added to a prompt. Skills in `skills-global` are Agent Skills files (`SKILL.md` with `name` and `description` in the frontmatter, and the body as the definition). They are seeded with `add_by_default` on, and copied onto the Favlo agent. A new agent receives every catalog skill that still has that switch on. `POST /global-skills/{id}/copy` copies the name, description, and definition onto an agent or a workspace. The copy has its own revision. Editing or deleting the catalog skill leaves the copy. `GET /agent-skills` lists skills stored on agents. `POST /agent-skills` stores a name and a description for one agent. Those skills are on the home channel and on a workspace session for that agent. `POST /workspaces/{id}/skills` stores a name and a description on the workspace. A workspace session appends the agent skills, then the workspace skills. `PATCH` and `DELETE` edit or remove one skill. A name is unique in the catalog, on that agent, or on that workspace. `GET /workspaces` without `channel_id` lists every workspace. On a gallery session after a local folder is saved, the runtime adds `gallery_list_folder`, `gallery_write_image`, `gallery_create_folder`, `gallery_rename_folder`, `gallery_move_folder`, `gallery_move_image`, `gallery_delete_image`, and `gallery_delete_folder`. Paths are relative to that local folder. The user message on a gallery turn names the open folder and, when an image is selected, that image. A png or jpg named in the message is read from that local folder and sent as an image on the turn. A name that is not there is left as text. An image already sent on an earlier turn of the session is left as text. Those names stay on that message, so the standing instructions do not change when the folder or the image changes. The home channel does not get those tools. The tools cannot add or delete a folder row. `gallery_write_image` takes a relative path and either base64 `bytes` or a `favlo://images/{id}.png` source. The note is [../docs/ai-assistance/README.md](../docs/ai-assistance/README.md).

## Development

The importable package is `web_api`. Tests live in the repository `tests/` directory:

```bash
uv run pytest tests/test_web_api.py tests/test_web_folder_pick.py tests/test_web_gallery.py tests/test_web_heartbeat.py tests/test_web_host.py tests/test_web_mcp.py tests/test_web_media.py
```

## Problems

Report a problem in this repository's issue tracker.

## License

The license is the Apache License, Version 2.0. The terms are in [../LICENSE.md](../LICENSE.md).
