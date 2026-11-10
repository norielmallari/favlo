# Pipelines

One package is one write set. Leave every other package's files untouched. The official list of packages is the root README. This file does not name one.

The as-built note for the package under edit is the README under `docs/` whose folder name matches that package. A process change updates `ARCHITECTURE.md`.

## Layout

A package name is an identifier. Layout is `pipelines/<model>/<runtime>/`. Import it as `{model}.{runtime}`. The model id, the directory name, and the CLI flag stem are the same underscore token. Runtimes are `metal41`, `cuda`, and `portable`.

Re-export `main` and `PIPELINE` from the package `__init__.py`. Use a regular package. A shader is a text file the module reads. Do not load a hyphenated entry file by path.

One module, one responsibility. Line count is not the trigger. Imports point one way: `formats` to `convert` or `session`, then `runtime`, then `commands`. When two modules need each other, move the shared names into a third module or leave the pieces together. Keep one class in one file. Do not split a class into mixins. Move a large source string, such as a Metal shader, into a text file the module reads. A namespace package is for one name split across distributions. A new model does not edit another package. Do not edit `src/favlo/cli.py` to register a runtime. Update that package's note in the same change that moves one of its modules.

`favlo serve` is the inference entry. `GET /v1/models` lists the registry. The `model` field on `POST /v1/responses` loads that row. `POST /v1/chat/completions` translates into that request and translates the Response back. Quantize and the other non-serve commands use `favlo --<model>`.

The write set for one backend is `pipelines/<model>/<runtime>/`, `docs/<model>/`, and tests that import only that package. `PIPELINE` in the package is the model id. Registration loads `sampling.py` from the package that owns the row. Do not pass a sampling card into the registry.

A served media package inherits `MediaPipeline`. A served language model inherits `LanguagePipeline`. A served embedding model inherits `EmbeddingsPipeline`. All three extend `Pipeline`. Those classes do not run a forward, name stages, or branch on the pipeline. A kernel does not branch on the pipeline. A new abstract method on `Pipeline` updates every subclass of every seam. A new abstract method on one seam updates every subclass of that seam. A seam migration may edit every served subclass of that seam in one change.

A pipeline calls `begin`, `count`, `end`, and `note` on the console. It does not add a phase, a unit, or a note topic for one package. Progress reporting is `docs/progress-reporting.md`. Synthetic `matmul2d` dtype rates are `docs/synthetic-matmul2d.md`.

A seam task may edit `language_pipeline.py`, `embeddings_pipeline.py`, `media_pipeline.py`, `media_memory.py`, `memory.py`, `mcp_server.py`, `response_chain.py`, `schedule.py`, and `pipelines.py`, and the listener, the registry, and the console. It does not edit another package to do that.

## Algorithms

Every equation comes from a public paper, spec, or vendor document, then maps onto a Metal op an M5 Pro with 48 GB runs. Another repository's source, tests, comments, and design notes are not a reference. Score a formula on this package's own forward. An int4 argmax is not evidence the equation is right. Check the BF16 shader dot of the checkpoint values. Record a package citation in that package's note, and a process citation in `REFERENCES.md`. The as-built form goes in that note, or in `ARCHITECTURE.md` when the change is the listener, the registry, or the console.

Allocate the input path and the output path first. The input path is the key/value cache and the prefill workspace. The output path is decode. An expert atlas, a staging buffer, and a scratch buffer the forward can rebuild receive what remains. When a formula needs more device memory, leave those three allocated, name another module to drop, and say which bytes it returns. A missing kernel stays missing. A host GEMV or a CPU matmul is not a stand-in. A language-model package does not fuse a whole layer. An image, video, music, or voice package may place several kernels of one generation in one command buffer.

## Model switch

One chat or media package is resident. A request that names a different folder or package closes that runtime before the next one opens. The same folder and the same package stay resident. An embeddings request does not close the resident runtime. It opens its own process for that request and closes it when the request returns.

`close` returns only after that package's device memory is released. That includes its key/value cache, prefill workspace, decode path, command queue, and open record. A prefix flush may finish first. After `close` returns, the previous runtime holds no Metal buffer.

Spans for one runtime share one owner string. The string is the flag stem, a colon, and `id()` of the object `open_runtime` returns, taken once when that runtime opens. The example shape is `example-model:1`. `allocate`, `upload`, and `sweep` use that string. A cache, a codec, or a probe in that package passes the same string. `open_runtime` receives `owner` when its signature includes that parameter. A package that omits the parameter keeps the `id()` string. `mint_owner` supplies the listener's string. The listener records every owner that gained a span during `open_runtime`, and `runtime._owner` when that attribute is a string. It sweeps those strings after `close`. An `open_runtime` that raises sweeps every owner that appeared during the call. A command outside `favlo serve` uses the same format, and that session's `close` sweeps the string before the command returns.

## Media preemption

An image, video, or music package reads preempt at each stage boundary, after the Metal command already submitted has finished. An embeddings request does not set `preempt` and does not close that device. On preempt, save the finished stage in a host checkpoint, leave the job in progress with the same id, and return. `close` then releases the device so the chat model can allocate. The listener calls the same handler again after those chats. The handler loads that checkpoint and continues. Cancel drops the checkpoint and does not resume. Device buffers do not survive `close`. A value the next stage needs is in the host checkpoint, kept in that package.

## Checks

A command that uses the GPU in its own process takes the machine lock. The procedure is the smoke-testing skill. Do not copy those commands here. A listener harness uses `attach` and does not take the lock.
