---
name: favlo-server
description: >-
  Start the Favlo listener and call its media MCP tools. Use when an agent
  needs favlo serve, the favlo MCP server, generate_image,
  generate_video_i2v, generate_video_t2v, generate_music,
  list_image_models, list_video_models_i2v, list_video_models_t2v,
  list_music_models,
  list_voice_models, media_status, media_cancel, or read_media. Do not
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

That starts a detached daemon and returns. A later `favlo serve` exits when that address is already in use. An agent does not stop it.

`POST /mcp` answers only after the daemon is up. If the `favlo` tools are missing after that POST succeeds, reload the `favlo` server in Cursor. Do not register a second server.

## Tools

`list_image_models`, `list_video_models_i2v`, `list_video_models_t2v`, `list_music_models`, `list_voice_models`, `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, `set_voice_model`, `media_status`, `media_cancel`, and `read_media` run immediately. An omitted generate seed is a random integer. The generate result includes the parameters stored in `<id>.json`, and `read_media` returns that record. A video row includes `resolutions`. A pipeline that names both modes appears on both video lists. A generate call with no `session` uses the `media` folder id in `models/registry.json` for that kind. `set_*` returns a session id. Pass that id as `session` on a later generate call to use the session. `harness.mcp_media` turns a harness `model` argument into that `set_*` call.

`generate_image`, `generate_video_i2v`, `generate_video_t2v`, and `generate_music` return a task when the client advertises `io.modelcontextprotocol/tasks`. The GPU starts when the one worker reaches that task. Poll `media_status` with the `taskId` every 2 seconds until `completed`, `failed`, or `cancelled`. Then `resources/read` the `favlo://` link in the result. A client that does not advertise the extension receives the finished tool result on that same call. Image and music bytes are inlined only when the caller asked and the file is at most 1 MiB. Video is always a `favlo://videos/{id}.mp4` link.

`generate_video_i2v` requires `image`: an id starting with `image_` from `generate_image`, or `{"type": "image", "data": "<base64 png>"}`. A filesystem path and a `favlo://` URI are refused. `generate_video_t2v` takes no still. Optional `duration` is clip length in seconds. An omitted `duration` uses the model default. `size` is required and is a `WIDTHxHEIGHT` from that model's `resolutions` on the matching list tool. `quality` is `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `standard`, or `hd`. `negative_prompt` defaults to empty, which is Wan's published sample negative and is encoded only when a guidance scale is not 1. `guidance_scale` is one number or `[low, high]`. The default is 1, and 1 skips the uncond forward. `steps` from 1 to 40 rebuilds the shifted Euler grid. 0 uses 4. The result is `favlo://videos/{id}.mp4`, read with `resources/read`.

`harness.mcp_media` advertises the extension, polls, and reads the link. Do not add an HTTP media route.

Chat is `POST /v1/chat/completions`. It is not an MCP tool. A chat preempts a running media task. An embedding does not. Media does not preempt a chat or an embedding. An embedding does not preempt a chat that already holds the GPU, and it does not unload the resident model. The embeddings process closes when that request returns.

Do not start a generate the user did not ask for. One media task runs at a time.
