---
name: favlo-server
description: >-
  Start the Favlo listener and call its media MCP tools. Use when an agent
  needs favlo serve, the favlo MCP server, generate_image, generate_video,
  generate_music, list_media_models, media_status, or media_cancel. Do not
  use for a language-model smoke or for kernel work.
---

# Favlo server

The listener is `favlo serve` on `127.0.0.1:8129`. Cursor reads [`.cursor/mcp.json`](../../mcp.json) in this repo. That entry is HTTP only: name `favlo`, type `http`, URL `http://127.0.0.1:8129/mcp`. Do not add a stdio command. Do not copy the server into a user-level MCP config.

## Start

If `127.0.0.1:8129` already accepts a connection, leave that process alone. Do not kill it and do not start a second listener. One exception: `POST /mcp` returns `not found`. That process was started before the route existed. Stop that pid and start the daemon again.

A media harness uses that same listener. `python -m harness.server run -- <command>` reuses it, or starts one and stops that process when the command exits. Do not point a harness at a second port.

If nothing is listening, from this repo:

```text
uv run favlo serve --non-interactive
```

That starts the daemon and returns. A later `favlo serve` attaches to it. `/quit` from the terminal that owns it is how a person stops it. An agent does not stop it.

`POST /mcp` answers only after the daemon is up. If the `favlo` tools are missing after that POST succeeds, reload the `favlo` server in Cursor. Do not register a second server.

## Tools

`list_media_models`, `media_status`, and `media_cancel` run immediately.

`generate_image`, `generate_video`, and `generate_music` return a task when the client advertises `io.modelcontextprotocol/tasks`. The GPU starts when the one worker reaches that task. Poll `media_status` with the `taskId` every 2 seconds until `completed`, `failed`, or `cancelled`. Then `resources/read` the `favlo://` link in the result. A client that does not advertise the extension receives the finished tool result on that same call. Image and music bytes are inlined only when the caller asked and the file is at most 1 MiB. Video is always a `favlo://videos/{id}` link.

The still on `generate_video` is a stored image id or a PNG image block. A filesystem path is refused. `quality` is an argument of that call. Image `n` from 1 to 10 is one task.

`harness.mcp_media` advertises the extension, polls, and reads the link. Do not add an HTTP media route.

Chat is `POST /v1/chat/completions`. It is not an MCP tool. A chat preempts a running media task. Media does not preempt a chat.

Do not start a generate the user did not ask for. One media task runs at a time.
