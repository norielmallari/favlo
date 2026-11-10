---
name: bootstrap-new-model
description: Add one Favlo inference pipeline without editing any other model. Use when creating, bootstrapping, or scaffolding a new model or pipeline, when the user asks to isolate a model from the others, or when checking that a new model works with its pipeline smokes and agent smokes.
---

# Bootstrap a new model

One model is one pipeline. Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) first. Edit only the new write set, plus the runtime tables when the pipeline is open-core.

## Write set

Add these, and nothing of another pipeline:

| Piece | Open-core path | Cloud custom path |
|---|---|---|
| Package | `pipelines/<pipeline>/<runtime>/` | `custom-pipelines/<pipeline>/<runtime>/` in the hosted checkout |
| Note | `docs/<pipeline>/` | that checkout's `docs/<pipeline>/` |
| Tests | `tests/` files that import only `<pipeline>.<runtime>` | `custom-pipelines/tests/` files that import only that package |

`<pipeline>` is the flag stem, the model directory, and a Python identifier, such as `qwen36_35b_a3b`. `<runtime>` is `metal41`, `cuda`, or `portable`. The import is `<pipeline>.<runtime>`. Discovery walks plugin roots for `<pipeline>/<runtime>/__init__.py`. Do not edit `src/favlo/cli.py`.

An open-core pipeline lives in the Favlo checkout. A cloud custom pipeline lives under the hosted checkout's `custom-pipelines/` and is loaded only when that directory is a plugin root (`pipeline_roots` or `FAVLO_PIPELINE_ROOTS`). It uses the same base class, CLI, and MCP contract. It does not add a row to the open runtime table.

The runtime package exports a typed `pipeline` instance. `favlo --<pipeline>` calls `pipeline.main`. `open_runtime` is that instance's method. `main` on the package may delegate to it.

## Base classes

`Pipeline` in `src/favlo/pipeline.py` is the root. A concrete class implements `main` and `open_runtime`. The seam class implements `handle`.

| Seam | Base class | What the subclass implements |
|---|---|---|
| Language | `LanguagePipeline` | `handle_responses` |
| Embeddings | `EmbeddingsPipeline` | `embed` |
| Media | `MediaPipeline` | `parse`, `generate`, `pack`, `mcp_tools` |

Bind one instance: `pipeline = Pipeline()`. A media module sets `handle_image = pipeline` (or the handler name `mcp_tools` declares). A language module binds `handle_responses` to the instance. An embeddings module binds `handle_embed` to the instance.

`MediaPipeline` sets `kind` (`image`, `video`, `music`, or `voice`), `id_prefix`, and `object_name`. `generate` calls `checkpoint(stop, hold)` at a stage boundary after the submitted command finishes. Cancel carries no hold. Preempt carries host values only. Device buffers do not go in the hold. `open_runtime` accepts `owner`, calls `memory.set_max_memory`, and stamps one owner for every allocate. `ensure_open` and `close` use `favlo.media_memory`.

Read the seam rule before writing the class, and add the new package and `docs/<pipeline>/**` to its globs when the pipeline is open-core: [language-pipeline.mdc](../../rules/language-pipeline.mdc), [embeddings-pipeline.mdc](../../rules/embeddings-pipeline.mdc), or [media-pipeline.mdc](../../rules/media-pipeline.mdc). Do not add another base class, a shared runner, or a kernel that branches on the pipeline. Do not change `src/favlo/` unless the task is the listener, the registry, the console, or the prefix store.

## CLI

`pipeline.main` is `favlo --<pipeline>`. Media commands are `setup`, `quantize`, `kernels`, and `smoke`, plus an offline `generate` when the pipeline has one. `setup` downloads the official checkpoint. `quantize` writes the packed record and registers it. `kernels` compiles the shader library and checks a dispatch. `smoke` runs one recorded generation. A GPU command takes `favlo.smoke_lock`.

`favlo media --pipeline <pipeline>` is the listener path. It does not start `serve`. The body fields are the ones `parse` accepts. Do not add a route under `/v1/images`.

## MCP

A media package implements `mcp_tools` and returns `MediaTool` rows. The host publishes `{pipeline}__{local name}`. Binding `handle_*` publishes nothing. The tool `handler` is the bound method name, and `result` is `image`, `video`, `audio`, or `json`. Descriptions stay within the listener caps (256 for the tool, 48 for each parameter). The tool list loads no weights.

One preemption harness discovers registered pipelines. Do not add a second one or a probe row that names this pipeline.

## Sampling card

Add `<runtime>/sampling.py` with no imports from the package. Registration loads it by path.

- `VARIANTS`: a non-empty dict. Each name is one token and contains no colon. Each row sets `temperature`, `top_p`, `top_k`, and `presence_penalty`.
- `DEFAULT_VARIANT`: one of those names.

`register_quantized` copies that module onto the row and stores `runtime` and `pipeline_root`. Do not pass a card into the registry. A `:suffix` id is one of those names. A bare id selects no row; this pipeline applies `DEFAULT_VARIANT`, or the row the request selects. Numbers come from this model's card, not from another pipeline's `sampling.py`.

## Docs

`docs/<pipeline>/` is the as-built note: what this pipeline runs, how to invoke it, and where its papers are. An open-core pipeline adds one row to the table in [ARCHITECTURE.md](../../../ARCHITECTURE.md) and one row in [docs/README.md](../../../docs/README.md). A cloud custom pipeline updates the hosted CLI catalog and the hosted harness note in the same change.

Citations for this model go in `docs/<pipeline>/`. Process citations stay in [REFERENCES.md](../../../REFERENCES.md). Do not add this model's kernels, cache, or sampling numbers to another pipeline's note, and do not add them to `ARCHITECTURE.md`.

No personal data: no names, home directories, hostnames, or volume paths. A claim by the author is in the third person.

## Isolation

Leave every other pipeline's package, shaders, tests, and docs untouched. Do not import another pipeline. Re-implement the math from this model's papers, model card, and the vendor documents. Another pipeline's source, tests, comments, and design notes are not a reference. Another application's tree is not a reference.

Prefill, decode, and calibration are Metal shaders. A missing kernel stays missing. A weight plane is a format whose op compiles and whose dispatch completes. Do not invent routing hits, energies, or token rates.

## Smoke

Run this package's checks. Stop if one fails. Do not start `favlo serve` yourself, and do not leave another model resident.

1. `uv run pytest` on the tests that import only this package.
2. `uv run favlo --<pipeline> kernels` when this pipeline has that command. It compiles the shader library and runs that package's dispatch checks.
3. A language model then follows [smoke-testing](../smoke-testing/SKILL.md): pipeline smoke, then the file agent, then the coding agent.
4. A media model runs its recorded smoke, then the picture harness for that kind. The harness calls the published tool. Exit 0 is the pass.

Record a printed measurement in this pipeline's note. Do not invent one, and do not write it into another pipeline's note.

## Done when

- `uv run favlo --<pipeline> --help` lists this pipeline's commands.
- The package exports a typed `pipeline` instance of the seam base class.
- A media pipeline's `mcp_tools` lists every tool the listener should publish.
- The pipeline smoke and the picture harness, or the language smokes, passed on this model.
- An open-core pipeline has one runtime-table row, one `docs/README.md` row, and its paths in the seam-rule globs.
- A cloud custom pipeline is in the hosted CLI catalog and harness note.
- `git diff` does not touch another pipeline.
