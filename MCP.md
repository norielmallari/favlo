# MCP

Favlo serves media generation and session memory on the same process as `favlo serve`. Language chat is `POST /v1/responses`. `POST /v1/chat/completions` translates into that request and translates the Response back. Neither is an MCP tool.

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

Each Favlo tool `description` from `tools/list` is at most 256 characters (`TOOL_DESCRIPTION_MAX_CHARS`). That is stricter than the OpenAI Chat Completions limit of 1024 on `tools[].function.description`. The MCP specification does not set a description length cap. Each parameter property `description` in the tool input schema is at most 48 characters (`TOOL_PARAM_DESCRIPTION_MAX_CHARS`); an empty property description is allowed when the name and type carry the meaning. `tests/test_mcp_tool_descriptions.py` checks both bounds on the mounted listener tools and on every published plugin tool.

## Tasks

A catalog row with `task` true is an optional task on `io.modelcontextprotocol/tasks`. Poll interval is 2000 ms. The listener publishes listener tools first, then one tool per `mcp_tools()` row, as `{pipeline_id}__{local name}`. A registry row that fails to import, fails when calling `mcp_tools()`, or declares an invalid tool is omitted; stderr notes the skip and `favlo serve` keeps listening. Hyphens in the pipeline id become underscores. The local name does not contain `__`. `tools/list` is that full set, in that order, for every connection. Every plugin tool name includes the pipeline prefix. A client must not keep a forever cache of those names; this response sets `ttlMs` 0 so a later list can replace the set. `tools/call` splits on the first `__`. The left side is the pipeline id and the right side is the local name. An unknown name is an error. One wrapper checks the schema, writes the arguments into the body, and sets the pipeline and the folder from the catalog. MCP does not list models. The pipeline id in the tool name is the model. The host opens the first registry folder on that pipeline. A task tool takes a media ticket. The child calls the handler attribute the record names and returns the saved file id. The host packs that file as one JSON text block: `metadata` and `mediaUri` (`file://`). Several files are an array of that object. The result does not include media bytes. Still fields are `image`, `source`, `target`, `subject`, and `background`: an `image_` id, a PNG object, or an absolute `file://` URI. A bare path and a `favlo://` input are refused. Other strings, including a prompt with a slash, stay text. Reload rebuilds the catalog. The next `tools/list` is that catalog.

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

Listener tools are listed here. A pipeline tool's schema lives on that module's `mcp_tools()` and is published with the pipeline prefix. This catalog does not copy those schemas.

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
| `client` | string | `""` | Owner. A web agent turn sets this to that agent. |

Returns a confirmation string or an id usable with `forget`. The same sentence can be stored once for each owner.

#### `recall`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural-language search over stored sentences |
| `client` | string | `""` | Owner. When set, only that owner's sentences are returned. |

Returns matching rows (id and text) or an error string.

#### `forget`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | string | required | Id from `remember` or `recall` |
| `client` | string | `""` | Owner. When set, another owner's sentence is left in place. |

Removes one stored sentence.

## Resources

Read with MCP `resources/read`. URIs may include a file suffix; the suffix selects the encoding where noted.

| URI template | MIME | Content |
|--------------|------|---------|
| `favlo://images/{image_id}` | `image/png` (+ JSON) | Stored PNG, then the standardized metadata JSON when present; optional `.png` suffix |
| `favlo://videos/{video_id}` | `video/mp4` | Stored MP4; optional `.mp4` suffix |
| `favlo://music/{music_id}` | from suffix | `mp3`, `wav`, or `flac`; bare id uses the job format in memory, else WAV on disk |
| `favlo://voice/{voice_id}` | from suffix | Same rules as music |

Implementation: `src/favlo/mcp_server.py` (`build_mcp`). Memory tool text: `src/favlo/memory_mcp.py`.
