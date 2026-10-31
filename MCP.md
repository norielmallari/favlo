# MCP

Favlo serves media generation and session memory on the same process as `favlo serve`. Chat completion stays on `POST /v1/chat/completions` and is not an MCP tool.

Listener behavior, GPU preemption, logging, and client YAML are in [docs/serve.md](docs/serve.md). This file is the tool and resource catalog for `POST /mcp`.

## Endpoint and protocol

| Item | Value |
|------|--------|
| URL | `http://127.0.0.1:8129/mcp` (default port) |
| Protocol | MCP `2026-07-28`, Streamable HTTP |
| Transport | One JSON object per `POST`, no session, no GET event stream |
| Capability | `io.modelcontextprotocol/tasks` |

`server/discover` lists supported versions. `initialize` with any other protocol version returns JSON-RPC `-32022`. Each request repeats the negotiated version and client capabilities in `params._meta`, and sends headers `mcp-protocol-version`, `mcp-method`, and for `tools/call` / `resources/read` also `mcp-name`.

Example client entry:

```yaml
mcp_servers:
  favlo:
    url: "http://127.0.0.1:8129/mcp"
```

There is no `/v1/images`, `/v1/videos`, or `/v1/audio` HTTP API. Stored bytes are read with `resources/read` on a `favlo://` URI.

## Tasks

Five tools are optional tasks: `generate_image`, `generate_video_i2v`, `generate_video_t2v`, `generate_music`, and `generate_voice`. Poll interval is 2000 ms.

| Client | `tools/call` behavior |
|--------|------------------------|
| Advertises `io.modelcontextprotocol/tasks` | Returns immediately with `resultType` `task`, `status` `working`, `taskId`, and `pollIntervalMs`. Work is queued; one job runs at a time in a pipeline process. |
| Does not advertise the extension | Blocks until the job finishes and returns the finished tool result on the same call. |

Completion and failure are observed with JSON-RPC `tasks/get` (part of the tasks extension) or the `media_status` tool. Cancel with `tasks/cancel` or `media_status`’s sibling `media_cancel`; both target the same task.

While `status` is `working`, `tasks/get` and `media_status` may include:

| Field | Meaning |
|-------|---------|
| `statusMessage` | `task is ongoing` when queued or paused for foreground work |
| `position` | Queue index when the task has not started yet |
| `stage` | Pipeline stage name, or `paused` when a chat preempted media |
| `progress` | Percent while the clip is generating |
| `eta` | Estimated seconds remaining |

## Tools

### Registry lists

These run in the request and do not enter the media task queue.

#### `list_image_models`

Returns registry rows whose pipeline generates an image.

#### `list_video_models_i2v`

Returns image-to-video rows. Each row includes `resolutions` when the pipeline defines `list_resolutions`. A pipeline that declares both modes appears on the I2V and T2V lists.

#### `list_video_models_t2v`

Returns text-to-video rows, with the same `resolutions` shape as I2V.

#### `list_music_models`

Returns registry rows whose pipeline generates music.

#### `list_voice_models`

Returns registry rows whose pipeline speaks. A row includes `voices` when the pipeline defines `list_voices`, and `emotions` when it defines `list_emotions`.

### Defaults and sessions

`models/registry.json` stores a `media` object. Its keys are `image`, `i2v`, `t2v`, `music`, and `voice`. Each value is a registry folder id. The first registered row for a kind fills an empty key. A later read fills a key that is still empty and saves it.

A generate call with no `session` uses that folder. `set_image_model`, `set_video_i2v_model`, `set_video_t2v_model`, `set_music_model`, and `set_voice_model` copy those defaults into one in-memory session, replace the kind named by the tool, and return the session id. A later `set_*` or generate call passes that same id. An omitted `session` on `set_*` starts a new id. An unknown id is an error. The session is not kept across a listener restart.

The caller repeats the returned id in its reply so a later turn can pass it.

#### `set_image_model`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Id from `list_image_models` |
| `session` | string | `""` | Omit to start a session. Pass an id to change that session |

**Result:** The session id, a token starting with `media_`.

#### `set_video_i2v_model`

Same parameters. `model` is an id from `list_video_models_i2v`.

#### `set_video_t2v_model`

Same parameters. `model` is an id from `list_video_models_t2v`.

#### `set_music_model`

Same parameters. `model` is an id from `list_music_models`.

#### `set_voice_model`

Same parameters. `model` is an id from `list_voice_models`.

### Task control

#### `media_status`

| Parameter | Type | Description |
|-----------|------|-------------|
| `taskId` | string | Task id from a claimed generate call |

Returns the queue record: `taskId`, `tool`, `arguments`, `status`, and working fields above. Unknown id is an error.

#### `media_cancel`

| Parameter | Type | Description |
|-----------|------|-------------|
| `taskId` | string | Task to cancel |

Cancels the task and drops any stored file for that job. A second call returns the same terminal status.

### Session memory

Stored in the profile memory database (`~/.favlo` by default). Embedding uses the listener’s embed hook.

#### `remember`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | One durable sentence: fact or preference |
| `client` | string | `""` | Optional client label |

Returns a confirmation string or an id usable with `forget`.

#### `recall`

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural-language search over stored sentences |

Returns matching rows (id and text) or an error string.

#### `forget`

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Id from `remember` or `recall` |

Removes one stored sentence.

### `generate_image`

Optional task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Image prompt |
| `size` | string | `720x1280` | `WIDTHxHEIGHT`; `auto` is the same 9:16 frame |
| `quality` | string | `high` | Quality preset for the pipeline |
| `seed` | integer | omitted | Omitted draws a random seed |
| `include_image` | boolean | `false` | Inline PNG in the result when decoded size ≤ 1 MiB |
| `session` | string | `""` | Session id from a `set_*` call. Omitted uses the registry default |

**Result:** Text content with the `image_*` id, a second text block with the stored parameters (the seed, the prompt, and the other fields used), a `resource_link` `favlo://images/{id}.png`, and optionally an inline `image` block. The same JSON is `<id>.json` beside the PNG.

### `generate_video_i2v`

Optional task. Produces one MP4.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Motion prompt; the still is the first frame |
| `size` | string | required | `WIDTHxHEIGHT` from `list_video_models_i2v` |
| `image` | string or object | required | `image_*` id from `generate_image`, or `{"type":"image","data":"<base64 png>"}` |
| `duration` | number | omitted | Clip length in seconds; omitted uses the model default |
| `seed` | integer | omitted | Omitted draws a random seed |
| `quality` | string | `high` | `auto`, `low`, `medium`, `high`, `xhigh`, `max`, `standard`, or `hd` |
| `negative_prompt` | string | `""` | Empty uses Wan’s published sample negative where applicable |
| `guidance_scale` | number or array | omitted | One number or `[low, high]`; default effective scale is 1 |
| `steps` | integer | `0` | Euler steps 1–40; `0` uses 4 |
| `session` | string | `""` | Session id from a `set_*` call. Omitted uses the registry default |

`image` refuses a filesystem path and a `favlo://` URI.

**Result:** Text with `video_*` id (and optional `size` / `seconds` fields), a text block with the stored parameters, plus `resource_link` `favlo://videos/{id}.mp4`. Video is never inlined. The same JSON is `<id>.json` beside the MP4. The still is not stored in that file.

### `generate_video_t2v`

Optional task. Same parameters and result shape as I2V except there is no `image`. `size` comes from `list_video_models_t2v`.

### `generate_music`

Optional task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Caption / style prompt |
| `response_format` | string | `mp3` | `mp3`, `wav`, or `flac` |
| `lyrics` | string | `""` | Lyrics when the pipeline uses them |
| `seconds` | number | omitted | Target duration when supported |
| `seed` | integer | omitted | Omitted draws a random seed |
| `vocal_language` | string | `""` | Vocal language when supported |
| `bpm` | integer | omitted | Tempo when supported |
| `keyscale` | string | `""` | Key when supported |
| `timesignature` | string | `""` | Meter when supported |
| `include_audio` | boolean | `false` | Inline audio when decoded size ≤ 1 MiB |
| `session` | string | `""` | Session id from a `set_*` call. Omitted uses the registry default |

**Result:** `music_*` id, a text block with the stored parameters (including the prompt, lyrics, and seed), `resource_link` `favlo://music/{id}.{ext}`, optional inline `audio`. The same JSON is `<id>.json` beside the clip.

### `generate_voice`

Optional task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text to speak; `[emotion]` tags use ids from `list_voice_models` when defined |
| `voice` | string | `""` | Speaker id from that row; empty uses the zero speaker |
| `instructions` | string | `""` | Extra instructions when the pipeline supports them |
| `response_format` | string | `mp3` | `mp3`, `wav`, or `flac` |
| `include_audio` | boolean | `false` | Inline audio when decoded size ≤ 1 MiB |
| `session` | string | `""` | Session id from a `set_*` call. Omitted uses the registry default |

**Result:** `voice_*` id, a text block with the stored parameters (the spoken line and the other fields used), `resource_link` `favlo://voice/{id}.{ext}`, optional inline `audio`. The same JSON is `<id>.json` beside the clip. Voice has no seed.

### `read_media`

Immediate. Returns the JSON stored for one generated file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Id starting with `image_`, `video_`, `music_`, or `voice_` |

An unknown id is an error. The record is `name`, `kind`, `model` when the job named one, `seed` when the pipeline uses one, and the other resolved parameters, including the prompts.

## Resources

Read with MCP `resources/read`. URIs may include a file suffix; the suffix selects the encoding where noted.

| URI template | MIME | Content |
|--------------|------|---------|
| `favlo://images/{image_id}` | `image/png` | Stored PNG; optional `.png` suffix |
| `favlo://videos/{video_id}` | `video/mp4` | Stored MP4; optional `.mp4` suffix |
| `favlo://music/{music_id}` | from suffix | `mp3`, `wav`, or `flac`; bare id uses the job format in memory, else WAV on disk |
| `favlo://voice/{voice_id}` | from suffix | Same rules as music |

Implementation: `src/favlo/mcp_server.py` (`build_mcp`). Memory tool text: `src/favlo/memory_mcp.py`.
