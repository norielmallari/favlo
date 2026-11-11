# This package

This package is an embeddings pipeline. It inherits `EmbeddingsPipeline`. The class writes the OpenAI list object. It does not normalize a vector or choose a hidden size. A language model and a media package do not inherit it. The as-built note is the README under `docs/` whose folder name matches this package.

A change to `embed` or the list object loads the seam-checks skill. Do not paste those commands here.

Admission is `embed`. An embeddings request sits ahead of chats that have not started and does not set `preempt`. `resident` is `beside`. `waits_for_memory` is false: the request does not wait on a memory hold or on `wait_released`, and its process does not take that hold while it loads or unloads. A lazy `open_for` keeps the second process in `_open_embed_for` when this model is not already resident, and parks the resident when it is. `embeddings_ttl_seconds` stays in settings.

Bind `handle_embed` to the one `Pipeline` instance. `embed` returns the vector and the token count. The class writes `usage.prompt_tokens`. Do not copy the list object into this package. Do not catch `ValueError` around the reply builder.

`parse` accepts a non-empty string or a non-empty list of strings. Anything else raises `input must be a non-empty string or array of strings` with `param` `input`. Do not say `input must be a string`. A `ValueError` from `embed` is a 400 with `param` `None`. The `model` field is `selection.requested_id`, or `""` when the selection is missing. It does not read `body["model"]`.

A new abstract method on `EmbeddingsPipeline` updates every served embedding package in that same change.

## Memory

`src/favlo/memory.py` is the only allocator. Do not call `newBufferWithLength` or keep a second allocator. Heaps are per owner, kind (`resident`, `scratch`, `private`), and storage (`shared` or `private`).

`open_runtime` accepts `owner`. The listener calls `memory.set_max_memory` before that open. Stamp `_owner` before the first allocate. The example shape is `example-model:1`. Every `allocate` and `upload` in that opening uses that string. `close` sweeps it in `finally`, including when `open` fails after a span exists. A missing record raises `FileNotFoundError` before `Gpu()`.

`budget()` asks the serve process and waits. `room()` is the budget minus live and pending bytes. Read those two. Do not parse `--max-memory`, and do not keep a private cap. An in-process command calls `memory.set_max_memory` before the first `budget()`.

The one knob is the passage length. Weights plus one passage at the length the call named must fit in `room()` before the first commit. Read the budget at open and again before a later call that can use a longer passage. A larger answer may raise the limit. Do not shrink it while a request is in flight. When one passage at the named length does not fit, raise `RuntimeError` and name that length. Do not truncate the input, drop layers, or change the packed quant. 16 GB is the product floor. It is not free memory and it is not `budget()`.

Spans go through `device_buffer(owner, nbytes)`. `close` sweeps the owner. A chat or media call still waits on a memory hold. This embeddings process does not.
