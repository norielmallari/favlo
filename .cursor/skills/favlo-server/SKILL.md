---
name: favlo-server
description: >-
  Start the Favlo listener and call its media MCP tools. Use when an agent
  needs favlo serve, the favlo MCP server, media_status, media_cancel,
  or a published {pipeline}__{local name} tool. Do not
  use for a language-model smoke or for kernel work.
---

# Favlo server

The listener is `favlo serve` on `127.0.0.1:8129`. Cursor reads [`.cursor/mcp.json`](../../mcp.json) in this repo. That entry is HTTP only: name `favlo`, type `http`, URL `http://127.0.0.1:8129/mcp`. Do not add a stdio command. Do not copy the server into a user-level MCP config.

## Start

If `127.0.0.1:8129` already accepts a connection, leave that process alone. Do not kill it and do not start a second listener. Unload a smoke hold with `uv run python -m favlo.smoke_lock release`, not by stopping the daemon. One exception: `POST /mcp` returns `not found`. That process was started before the route existed. Stop that pid and start the daemon again.

A media harness and a language harness use that same listener. `python -m harness.server run -- <command>` and `python -m favlo.smoke_lock attach --pipeline <stem> --command <name> -- <command>` reuse it, or start one and stop that process when the command exits. `attach` closes the named pipeline first when that model is resident and its sources are newer than the worker. Do not point a harness at a second port.

If nothing is listening, from this repo:

```text
uv run favlo serve --non-interactive
```

That starts a detached daemon and returns. A later `favlo serve` stops the listener already on that address and any pipeline process, then listens. An agent does not stop it by hand.

`POST /mcp` answers only after the daemon is up. If the `favlo` tools are missing after that POST succeeds, reload the `favlo` server in Cursor. Do not register a second server.

## Tools

`media_status`, `media_cancel`, `remember`, `recall`, and `forget` run immediately. A registered module returns `mcp_tools()`. The host publishes `{pipeline}__{local name}`. A tool marked `task` returns a task when the client advertises `io.modelcontextprotocol/tasks`. The GPU starts when the one worker reaches that task. Poll `media_status` with the `taskId` every 2 seconds until `completed`, `failed`, or `cancelled`. The finished tool result is one text JSON block: `metadata` and a `file://` `mediaUri`. A client that does not advertise the extension receives that result on the same call. `resources/read` of a `favlo://` URI stays for a client that is not on this machine. MCP does not list models or publish a `model` argument. The published tool name is the pipeline. The host opens that pipeline's first registry folder. `harness.mcp_media` may still pass a folder id when it prefixes a local name.

An image argument is an id starting with `image_`, `{"type": "image", "data": "<base64 png>"}`, or an absolute `file://` URI. A bare path and a `favlo://` URI are refused. Optional `duration` is clip length in seconds. An omitted `duration` uses the model default. `size` is `WIDTHxHEIGHT` when the tool requires it.

`harness.mcp_media` advertises the extension, polls, and reads the link. Do not add an HTTP media route.

Chat is `POST /v1/responses`. `POST /v1/chat/completions` translates into that request and translates the Response back. Neither is an MCP tool. A chat preempts a running media task. An embedding does not. Media does not preempt a chat or an embedding. An embedding does not preempt a chat that already holds the GPU, and it does not unload the resident model. The embeddings process closes when that request returns.

Do not start a generate the user did not ask for. One media task runs at a time.
