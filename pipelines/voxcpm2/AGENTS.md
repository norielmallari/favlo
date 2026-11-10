# This package

This package is a media pipeline. It inherits `MediaPipeline`. Kind is `voice`. `id_prefix` and `object_name` are set on the class. The smoke tool local name is `generate_voice`. The as-built note is the README under `docs/` whose folder name matches this package.

A change to `parse`, `generate`, `pack`, pause, or the job dict loads the media-preemption skill. Do not paste those commands here.

`MediaPipeline` owns the shared job, pause, pack, and memory-gate shell. The class does not run a forward, name stages, or acquire a scheduler ticket. It does call `runtime.ensure_open` and `runtime.close`. `table_name`, `seeds`, and `id_hex` stay on the class when they differ from the shell defaults.

Bind one instance. `__call__` is `handle`. Implement `parse`, `generate`, and `pack`. The checkpoint type stays in this package. Return `mcp_tools()`. Binding `handle_*` publishes nothing. The host publishes `{pipeline}__{local name}`.

`generate` calls `stop()` at a stage boundary after the submitted command finishes. A safe point calls `checkpoint(stop, hold)` after that command. Cancel raises `MediaPaused("cancel", None)`. Preempt raises `MediaPaused("preempt", hold)` with the host values the next stage needs. Device buffers do not go in the hold and do not survive `close`. Resume calls `generate` again with that hold and the same job id.

Clear a hold with `state[key] = None`. Do not pop the key. The keys are `image_hold`, `video_hold`, `music_hold`, and `voice_hold`, the same tuple in `pipeline_worker.py` and `pipelines.py`. A new kind adds its key to both tuples in that change.

`parse` or `generate` raises `MediaRequestError` for a bad body. That is a 400. Any other exception fails the job with `error` `{"message", "type"}`. `describe` copies the job keys MCP already reads (`size`, `seconds`, `response_format`). `reply.json` runs once with the job before `generate`. Call it again only when `pack` or `working` returns a different dict.

A new abstract method on `MediaPipeline` updates every served media package in that same change. This package's note records this package's checkpoint. Do not describe another package there.

## Memory

The runtime implements `ensure_open(request)` and `close()`. The shell calls `ensure_open` after parse and always calls `close` before `on_finish`, including on pause, fail, and cancel. Helpers live in `src/favlo/media_memory.py`: `stamp_owner`, `open_gate`, `begin_release`, `confirm_release`, and `finish_release`. Do not keep `_memory_cap`, a private `parse_memory`, or a private reclaim wait.

`src/favlo/memory.py` is the only allocator. `open_runtime` accepts `owner` and calls `memory.set_max_memory`. Stamp with `stamp_owner` before the first allocate. The example shape is `example-model:1`. `working_set_bytes(request)` returns weights plus scratch for the named shape. `ensure_open` runs `open_gate` and reloads when closed. The request keeps its resolution, frame count, and duration. Shrink only the scratch knob.
