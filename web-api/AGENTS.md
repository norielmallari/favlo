# Web API

A feature a person can start from the app includes a `web-api/` change the app can trigger. The same change adds or extends a Playwright happy path. The procedure is the web-happy-path skill. A change a person cannot start, and that leaves the screen the same, adds no script.

Use the Microsoft Agent Framework and AG-UI paths that already perform the step. A tool call, a task result, a stream event, and a run lifecycle stay on those paths. Do not add a workaround until the operator approves that workaround. Name the native call and why it cannot be used. Approval of one workaround does not cover the next.

For MCP tasks, the native path is a caller-owned `mcp.Client` with `TasksClientExtension`. `Client.call_tool` resolves the task before the function returns, and AG-UI emits that result in the same run.

Favlo media generate tools are an approved workaround. `session.call_tool` with `allow_claimed` returns the file name and says the file is not ready. The host polls `tasks/get` on its own client. When a task reaches a terminal status, the host starts a silent follow-up through `begin_greeting` and `favloSilent`. The cue is `<media_completed>`, `<media_cancelled>`, or `<media_error>` around the file name. That fence is the new user message. The task result is not sent. A gallery write selects that file in the same `gallery.updated` event. A file cue does not include a link or a task id. It is not wrapped in `<system_reminder>`. List, set, memory, status, and cancel stay on `Client.call_tool`.

## Stable prefix

The AG-UI thread snapshot is the conversation record. Later turns append. They do not rewrite earlier rows.

Leave prior messages byte-stable. Host cues are a durable fenced `user` row, `<system_reminder>` around the note, marked `metadata.favlo.ui=hidden`. A finished generate task uses the media fence on that same hidden row and is not wrapped in `<system_reminder>`. The run persists that row. A gallery selection's sidecar text and file name ride on the user message the model receives. The path comes from that POST's `state` and is not written again. The transcript projection omits the fence. Do not use mid-chat `system` or `developer` for those cues. Do not inject context only for the current request. Hide host cues in the transcript projection, and keep them in the snapshot. The trusted server writes host-cue rows before the run. Clients must not supply them. A silent gallery greeting uses that same channel plus an empty user pad the transcript omits. Do not delete snapshot rows after the run to hide them.

Tool schemas already used in a thread stay available until clear. Persist `persist_loaded` and `restore_loaded`. Prefer eager gallery tools over a mid-turn load that changes the schema. Clear and archive mint a new thread. A chat turn sets `store` true. The next model call sends the latest response id as `previous_response_id` and only the new turn. The listener prepends the stored chain, including reasoning. A missing id starts a new chain.

A fact that must appear every turn without growing history uses a channel that is identical whenever that fact is unchanged. A turn-varying fact belongs in durable history when the model must remember it.

Catalog names in `web_api/catalog.py` and `web-spa/src/workspace/catalog.ts` stay paired. UI tokens and screen rules are `web-spa/AGENTS.md`.
