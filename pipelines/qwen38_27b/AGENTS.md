# This package

This package is a language pipeline. It inherits `LanguagePipeline`. The class is the disconnect shell around one response. It does not encode a prompt, sample a token, or parse a tool call. The as-built note is the README under `docs/` whose folder name matches this package. Read that note before editing kernels.

A shell change loads the seam-checks skill. A GPU smoke loads the smoke-testing skill. Do not paste those commands here.

## Responses

One concrete class: `class Pipeline(LanguagePipeline)`. Implement `handle_responses`. That method reads a Responses body and writes a Response, including its sampler and its stream. Bind `handle_responses` to the one `Pipeline` instance. The daemon calls that instance. Do not keep a `_Hooks` middle class. Do not bind `handle_chat`. `RequestClosed` subclasses `ClientClosed`. `handle_responses` raises `RequestClosed`. The shell clears `runtime._stop`.

This package implements the Responses API only. Do not translate Chat Completions bodies, write `chat.completion` or `chat.completion.chunk` objects, or wrap the reply in `ChatWire`. The listener owns that wrapper once. `POST /v1/chat/completions` runs `chat_to_responses`, dispatches this instance, and `ChatWire` or `ResponseWire` lives in `favlo.responses`. `POST /v1/responses` is the generation route. Another package's chat, stream, or markup parser is not a reference.

`content` is the reply. It is null when the model only calls a tool. `tool_calls` are the calls, in the OpenAI function shape. `reasoning` is the trace when the model wrote one. `reasoning_content` is that same text. An empty thought is not that field. `finish_reason` is `stop`, `length`, or `tool_calls`. The reply fills `usage` with the OpenAI token fields and `max_prompt_tokens` before the body is encoded. This package may set `cached_tokens`, `context_length`, and `max_prompt_tokens` on the Response, including `_chat_usage` for the wire.

`chat_to_responses` maps the chat body onto a Responses request. `reasoning_content` becomes a reasoning item. Effort and `preserve_thinking` pass through. Markers stay out of `content`. A call becomes `tool_calls`. An open trace stays `reasoning_content` and does not become the reply. A call that starts before the trace closes ends the trace. The marker strings belong to this package. Do not drop `reasoning_content` on the way in. A missing reasoning item breaks the stored prefix when `preserve_thinking` is on.

The chat client in `src/favlo/chat_client.py` passes those fields through. It does not scan for markup, close a trace, or copy reasoning into the reply. A harness does not add its own parser for markup. There is no shared sampler on `LanguagePipeline`.

## Store

`store` defaults to on. `store: false` leaves the body unchanged and drops the response. The chain is process memory in `src/favlo/response_chain.py`. Restarting `favlo serve` drops the chain. `LanguagePipeline.handle` calls `present` before `handle_responses`.

`present` rebuilds one transcript and removes `store` and `previous_response_id` before this package reads the body. This package encodes `input`. It does not call `remember`. `finish_turn` keeps the Response this package wrote.

A client that resends user and assistant messages has those copies dropped by `present`. A new user message and a `function_call_output` stay. An image already in the chain stays once. An unknown `previous_response_id` starts a new chain. An unknown `conversation` fails the turn. `conversation` and `previous_response_id` are not set together. Instructions ride on the current body. A gallery sidecar caption stays `store: false`.

`on_open` is the work that already sits outside the try. The base method assigns `runtime._stop` from `reply.stopped`. A subclass that adds open work calls `super`. `prepare` is empty. No served language class overrides it. `on_finish` always runs after `prepare` and clears `runtime._stop`. Hooks are called directly. Do not swallow a hook exception.

A new abstract method on `LanguagePipeline` updates every served language package in that same change.

## Variants and effort

Registration copies `VARIANTS` and `DEFAULT_VARIANT` from this package's `sampling.py`. The public model id is the row's alias when it stores one, otherwise the folder. `GET /v1/models` and `resolve` use that public id only. Listed ids are the public id and each registered suffix. The packed folder name is not a request id when the row has an alias.

Each card sets `temperature`, `top_p`, `top_k`, and `presence_penalty`. It may set `reasoning_effort`. Choosing a suffix loads that card onto `Selection.sampling` and, when present, that card's `reasoning_effort`. A bare id uses `DEFAULT_VARIANT`. The request body still overrides sampling fields one by one. Changing effort does not swap the sampling card. Do not infer thinking from the suffix spelling. Read the stored `reasoning_effort`, or the request.

`reasoning.effort` on the Responses body wins when the request sets it. Chat Completions `reasoning_effort` and `enable_thinking` map onto it. Accept every listed OpenAI effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. Map an unsupported name to the next higher supported value, then to the next lower.

When the request omits effort, resolve in this order: the selected variant's `reasoning_effort`, then an explicit row-level `reasoning_effort`, then row `enable_thinking`, then this package's default. Keep the mapping in `sampling.py` and document it in the package note. Pass the effort string into that resolve step. Do not collapse it to a bool before this package maps it.

A prompt up to the configured maximum is accepted. That maximum is the served context window. A prompt past that maximum is rejected. `truncation` `auto` may drop the oldest non-system messages until the prompt fits. An omitted `truncation` does not. When the prompt fits the maximum and does not fit the current working set, grow that working set from the ledger and then run prefill. The grow stays in this package.

## Memory

`src/favlo/memory.py` is the only allocator. Do not call `newBufferWithLength` or keep a second allocator. Heaps are per owner, kind (`resident`, `scratch`, `private`), and storage (`shared` or `private`).

`open_runtime` accepts `owner`. The listener calls `memory.set_max_memory` before that open. Stamp `_owner` before the first allocate. The example shape is `example-model:1`. Every `allocate` and `upload` in that opening uses that string. `sweep` matches the whole string. `close` sweeps it in `finally`, including when `open` fails after a span exists. A missing record raises `FileNotFoundError` before `Gpu()`. The listener passes `mint_owner` when the signature includes `owner`.

`budget()` asks the serve process and waits. `memory.set_max_memory` caps that answer. `room()` is the budget minus live and pending bytes. Read those two. Do not parse `--max-memory`, and do not keep `_memory_cap` or a private `parse_memory`. An in-process `complete` calls `memory.set_max_memory` before the first `budget()`.

The one knob is the context working set. Weights plus that working set must fit in `room()` before the first commit. The prompt keeps its length. Read the budget at open and again before growing the cache, including before a prefill chunk. A larger answer may raise the working set. Do not shrink it while a request is in flight. At open, shrink the working set down to the minimum that still runs the prompt. When that minimum does not fit, raise `RuntimeError` and name the prompt length. Do not drop layers, stand in a host GEMV, or change the packed quant. 16 GB is the product floor. It is not free memory and it is not `budget()`.

Spans go through `device_buffer(owner, nbytes)`. `release` returns a span the caller has dropped. Leave a buffer a later command or a live local still names. `close` sweeps the owner. The release is done when that owner has no live span and no command still holds a freed span. `memory.wait_released` returns on that check. The daemon does not admit the next call while an allocation or a release is still open.

Decode streaming uses a queue. The decode loop enqueues the token id and returns. It does not decode text, format an event, or write the socket. A stream thread writes the event. The enqueue does not wait for that write. The decode thread joins the stream thread after generation returns. A dead client stops the decode loop on the next enqueue.

## Progress

Progress uses `favlo.console` only. Store the listener console on the runtime at open. Do not `print` a tok/s line. Phases are `load`, `prefill`, and `decode` on the serve path, and `download` and `quantize` on setup. Required notes, in order: `prompt`, then `prefix_load` when a stored prefix was loaded, then `cache`. Optional notes are `prefix_store`, `prefix_store_failed`, `mtp_depth`, `mtp`, `incoming`, `output`, and `context`. A zero-duration phase still closes with `format_rate`.

No progress write blocks the hot path. Opening a phase, closing a phase, and a diagnostic note are enqueued. The console thread writes the line.

Load begins with `begin("load", total=bytes, unit="bytes")` and advances `done` by each sequential read. Prefill begins after prefix align when the uncached suffix is greater than zero, with `begin("prefill", total=uncached_tok, unit="tok")`. Do not open a prefill row for a full cache hit. Advance `done` after each chunk and as each finished layer lands. The prefill clock starts at the uncached forward and ends when the first output token is sampled. Close with `format_rate("prefill", uncached_tok, seconds)`. Decode begins after that first sample when `console.live_decode` is true. Set `progress.done` to the completion length after each kept token. Close with `format_rate("decode", completion_tok, seconds)`. On cancel or disconnect, end every open row.

NumPy stays off the prefill, decode, routing, expert grouping, and sampling path. The sampled token id may come back as bytes. Checkpoint packing, prefix packing, and the kernel self-checks are not that path. The route result stays on device.

Quantize from BF16 uses one worker per CPU. The pool size is `os.process_cpu_count()`, or `os.cpu_count()` when that count is missing. Each worker sets `VECLIB_MAXIMUM_THREADS`, `OMP_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1.

The packed weight file is read from SSD into one resident Metal buffer, in sequential pieces of at least 8 MiB. Each matrix stays at its byte offset. A launch binds the buffer there. A layer file that stays on SSD is read the same way into the slot that runs it.

## This package

`on_open` calls `super` and clears `_halted`.

Supported efforts are `none`, `low`, `medium`, and `xhigh`.
