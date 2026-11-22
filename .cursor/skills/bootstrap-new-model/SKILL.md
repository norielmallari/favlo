---
name: bootstrap-new-model
description: Add one Favlo inference pipeline without editing any other model. Use when creating, bootstrapping, or scaffolding a new model or pipeline, when the user asks to isolate a model from the others, or when checking that a new model works with its pipeline smokes and agent smokes.
---

# Bootstrap a new model

One model is one pipeline. Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) first. Edit only the new write set, plus the runtime tables when the pipeline is open-core.

## Write set

Add these, and nothing of another pipeline:

| Piece | Open-core path | Extra plugin root |
|---|---|---|
| Package | `pipelines/<pipeline>/<runtime>/` | `custom-pipelines/<pipeline>/<runtime>/` under a directory listed in `pipeline_roots` or `FAVLO_PIPELINE_ROOTS` |
| Note | `docs/<pipeline>/` | that plugin tree's `docs/<pipeline>/` |
| Tests | `tests/` files that import only `<pipeline>.<runtime>` | that plugin tree's tests that import only that package |

`<pipeline>` is the flag stem, the model directory, and a Python identifier, such as `example-model`. `<runtime>` is `metal41`, `cuda`, or `portable`. The import is `<pipeline>.<runtime>`. Discovery walks plugin roots for `<pipeline>/<runtime>/__init__.py`. Do not edit `src/favlo/cli.py`.

An open-core pipeline lives in this Favlo checkout. A pipeline outside this tree lives under an extra plugin root and loads only when that directory is listed (`pipeline_roots` or `FAVLO_PIPELINE_ROOTS`). It uses the same base class, CLI, and MCP contract. It does not add a row to the open runtime table. This skill does not name another checkout.

The runtime package exports a typed `pipeline` instance. `favlo --<pipeline>` calls `pipeline.main`. `open_runtime` is that instance's method. `main` on the package may delegate to it.

## Base classes

`Pipeline` in `src/favlo/pipeline.py` is the root. A concrete class implements `main` and `open_runtime`. The seam class implements `handle`.

| Seam | Base class | What the subclass implements |
|---|---|---|
| Language | `LanguagePipeline` | `handle_responses` |
| Embeddings | `EmbeddingsPipeline` | `embed` |
| Media | `MediaPipeline` | `parse`, `generate`, `pack`, `mcp_tools` |
| Video | `BulkMediaPipeline` | the same methods. A video package does not stop at `MediaPipeline` |

Bind one instance: `pipeline = Pipeline()`. A media module sets `handle_image = pipeline` (or the handler name `mcp_tools` declares). A language module binds `handle_responses` to the instance. An embeddings module binds `handle_embed` to the instance.

`MediaPipeline` sets `kind` (`image`, `video`, `music`, or `voice`), `id_prefix`, and `object_name`. `generate` calls `checkpoint(stop, hold)` at a stage boundary after the submitted command finishes. Cancel carries no hold. Preempt carries host values only. Device buffers do not go in the hold. `open_runtime` accepts `owner`, calls `memory.set_max_memory`, and stamps one owner for every allocate. `ensure_open` and `close` use `favlo.media_memory`.

Read the sibling package `AGENTS.md` for that seam before writing the class. Write `pipelines/<pipeline>/AGENTS.md` from that sibling wording, speaking as "this package", and do not add a glob rule. Do not add another base class, a shared runner, or a kernel that branches on the pipeline. A seam task may edit `language_pipeline.py`, `embeddings_pipeline.py`, `media_pipeline.py`, `media_memory.py`, `memory.py`, `mcp_server.py`, `response_chain.py`, `schedule.py`, and `pipelines.py`, and the listener, the registry, and the console. Do not change other modules under `src/favlo/`.

## Tokenizer

`Tokenizer.from_file` runs once per checkpoint in the process. Encode and decode use the stored object.

A Jinja template is read and compiled once per checkpoint. `encode_chat` renders that template.

Marker ids that do not depend on the request are computed once.

A media or embeddings runtime holds the tokenizer for the open. `encode` does not call `from_file`.

A child process loads once inside that process.

The cache is per package. The miss path takes a lock.

A quantize command and a harness may load the file once for that command.

## CLI

`pipeline.main` is `favlo --<pipeline>`. Media commands on that flag are `setup`, `quantize`, and `kernels`. `setup` downloads the official checkpoint. `quantize` writes the packed record and registers it. `kernels` compiles the shader library and checks a dispatch. Generation is `favlo media`.

`favlo media --pipeline <pipeline>` is the listener path. It does not start `serve`. The body fields are the ones `parse` accepts. Do not add a route under `/v1/images`. The listener pack is `metadata` and `mediaUri`. A command that copies the stored file reads `mediaUri`.

## Video

A served video package subclasses `BulkMediaPipeline`. Image, music, and voice stay on `MediaPipeline`. Yield to a language or embeddings call at each diffusion boundary the sibling video package uses, after the submitted command finishes. The hold is host values. Device buffers do not survive `close`.

Read [serving-media](../../rules/serving-media.mdc). The stored clip is an uncompressed AVI. The served MP4 is encoded from that file. Do not pipe frames into an encoder and store only the MP4.

The serve path does not import torch. NumPy stops at quantization and tests. A layer product is `matmul2d` on one packed span. A missing kernel stays missing. A host matmul is not a stand-in.

Weights are one kernel-order file and a plane index. Open allocates one shared span and reads the file into it. The forward binds that span.

A full clip is a listener job. `print` does not paint `favlo serve`. Progress is `begin`, `count`, and `end` on the console the worker forwards.

The package `AGENTS.md` copies the sibling video package note, including bulk, yield, the lock, and the AVI. Do not leave it as a stub.

## MCP

A media package implements `mcp_tools` and returns `MediaTool` rows. The host publishes `{pipeline}__{local name}`. Binding `handle_*` publishes nothing. The tool `handler` is the bound method name, and `result` is `image`, `video`, `audio`, or `json`. Descriptions stay within the listener caps (256 for the tool, 48 for each parameter). The tool list loads no weights.

One preemption harness discovers registered pipelines. Do not add a second one or a probe row that names this pipeline.

## Sampling card

Add `<runtime>/sampling.py` with no imports from the package. Registration loads it by path.

- `VARIANTS`: a non-empty dict. Each name is one token and contains no colon. Each row sets `temperature`, `top_p`, `top_k`, and `presence_penalty`.
- `DEFAULT_VARIANT`: one of those names.

`register_quantized` copies that module onto the row and stores `runtime` and `pipeline_root`. Do not pass a card into the registry. A `:suffix` id is one of those names. A bare id selects no row; this pipeline applies `DEFAULT_VARIANT`, or the row the request selects. Numbers come from this model's card, not from another pipeline's `sampling.py`.

## Docs

`docs/<pipeline>/` is the as-built note: what this pipeline runs, how to invoke it, and where its papers are. An open-core pipeline adds one row to the table in [ARCHITECTURE.md](../../../ARCHITECTURE.md) and one row in [docs/README.md](../../../docs/README.md). A pipeline on an extra plugin root updates that tree's CLI catalog and harness note in the same change. It does not add a row to this checkout's Supported Models table.

The README under `docs/<pipeline>/` is a summary. `ARCHITECTURE.md` in that folder holds kernels, the packed record, the forward, and registry fields. Do not add a `SETUP.md`.

Citations for this model go in `docs/<pipeline>/`. Process citations stay in [REFERENCES.md](../../../REFERENCES.md). Do not add this model's kernels, cache, or sampling numbers to another pipeline's note, and do not add them to the root `ARCHITECTURE.md`.

No personal data: no names, home directories, hostnames, or volume paths. A claim by the author is in the third person.

## Isolation

Leave every other pipeline's package, shaders, tests, and docs untouched. Do not import another pipeline. Re-implement the math from this model's papers, model card, and the vendor documents. Another pipeline's source, tests, comments, and design notes are not a reference. Another application's tree is not a reference.

Prefill, decode, and calibration are Metal shaders. A missing kernel stays missing. A weight plane is a format whose op compiles and whose dispatch completes. Do not invent routing hits, energies, or token rates.

## Smoke

Run this package's checks. Stop if one fails. Do not start `favlo serve` yourself, and do not leave another model resident.

Every GPU command, for every pipeline, uses [smoke-testing](../smoke-testing/SKILL.md). No exceptions. `run` when this process compiles, dispatches, loads weights, or quantizes. `attach` when the command only calls the listener. Do not launch pytest, `kernels`, or a generate except as the command after `--`.

1. The tests that import only this package, under `run` when they touch the GPU.
2. `kernels`, under `run`, when this pipeline has that command. It compiles the shader library and runs that package's dispatch checks.
3. A language model then follows that skill: pipeline smoke, then the file agent, then the coding agent.
4. A media model runs its recorded smoke through `favlo media` against a listener already up, under `attach`, then the picture harness for that kind. The harness calls the published tool. Exit 0 is the pass. An in-process generate is not that smoke, and its temp file is not the stored clip.

Record a printed measurement in this pipeline's note. Do not invent one, and do not write it into another pipeline's note.

## Done when

- `uv run favlo --<pipeline> --help` lists this pipeline's commands.
- The package exports a typed `pipeline` instance of the seam base class.
- A media pipeline's `mcp_tools` lists every tool the listener should publish.
- The pipeline smoke and the picture harness, or the language smokes, passed on this model.
- An open-core pipeline has one runtime-table row, one `docs/README.md` row, and `pipelines/<pipeline>/AGENTS.md`.
- A pipeline on an extra plugin root is in that tree's CLI catalog and harness note, and is not listed in this checkout's Supported Models table.
- `git diff` does not touch another pipeline.
