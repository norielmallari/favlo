---
name: bootstrap-new-model
description: Add one MADMetal inference pipeline without editing any other model. Use when creating, bootstrapping, or scaffolding a new model or pipeline, when the user asks to isolate a model from the others, or when checking that a new model works with its smoke tests and long prompt.
---

# Bootstrap a new model

One model is one pipeline. Read [ARCHITECTURE.md](../../../ARCHITECTURE.md) first. Edit only the new write set, plus the two tables that list runtimes.

## Write set

Add these, and nothing of another pipeline:

| Piece | Path |
|---|---|
| Entry | `<pipeline>.py` at the repo root |
| Package | `<package>/` where `<package>` is `<pipeline>` with hyphens replaced by underscores |
| Note | `docs/<pipeline>/` |
| Tests | `tests/` files that import only `<package>` |

`<pipeline>` is the flag stem, such as `qwen36-35b-a3b`. The entry assigns `PIPELINE` to that stem. `src/madmetal/cli.py` discovers the file by that assignment. Do not edit `cli.py`.

The entry loads the package by path and does not import another pipeline. The package exports `main`, `open_runtime`, and `handle_chat`. `madmetal --<pipeline>` calls `main`. `madmetal serve` calls `open_runtime` and `handle_chat` after the registry row names this pipeline.

## Sampling card

Add `<package>/sampling.py` with no imports from the package. Registration loads it by path.

- `VARIANTS`: a non-empty dict. Each name is one token and contains no colon. Each row sets `temperature`, `top_p`, `top_k`, and `presence_penalty`.
- `DEFAULT_VARIANT`: one of those names.

`add_model` and `register_quantized` copy that module onto the row. Do not pass a card into the registry. A `:suffix` id is one of those names. A bare id selects no row; this pipeline applies `DEFAULT_VARIANT`, or the row the request selects. Numbers come from this model's card, not from another pipeline's `sampling.py`.

## Docs

`docs/<pipeline>/` is the as-built note: what this pipeline runs, how to invoke it, and where its papers are. Add one row to the table in [ARCHITECTURE.md](../../../ARCHITECTURE.md) and one row in [docs/README.md](../../../docs/README.md).

Citations for this model go in `docs/<pipeline>/`. Process citations stay in [REFERENCES.md](../../../REFERENCES.md). Do not add this model's kernels, cache, or sampling numbers to another pipeline's note, and do not add them to `ARCHITECTURE.md`.

No personal data: no names, home directories, hostnames, or volume paths. A claim by the author is in the third person.

## Isolation

Leave every other pipeline's entry, package, shaders, tests, and docs untouched. Do not import another pipeline. Do not add a base class, a shared runner, or a kernel that branches on the pipeline. Do not change `src/madmetal/` unless the task is the listener, the registry, or the console.

Re-implement the math from this model's papers, model card, and the vendor documents. Another pipeline's source, tests, comments, and design notes are not a reference. Another application's tree is not a reference.

Prefill, decode, and calibration are Metal shaders. A missing kernel stays missing. A weight plane is a format whose op compiles and whose dispatch completes. Do not invent routing hits, energies, or token rates.

## Smoke, then the long prompt

Run the smoke checks before any long generation. Stop if one fails. Do not start `madmetal serve` yourself, and do not leave another model resident. The long-prompt harness starts its own server.

1. `uv run pytest` on the tests that import only `<package>`.
2. `uv run madmetal --<pipeline> kernels` when this pipeline has that command. It compiles the shader library and runs that package's dispatch checks.
3. After both have passed, run the long prompt against this model:

```text
uv run python -m harness.long_prompt --model <pipeline>
```

The default target is 65536 tokens. Pass `--tokens` when this pipeline's prompt limit is different. The command prints prefill and decode rates. Record those measured rates in this pipeline's note. Do not invent them, and do not write them into another pipeline's note.

## Done when

- `uv run madmetal --<pipeline> --help` lists this pipeline's commands.
- The smoke checks above passed, and the long prompt then finished on this model.
- The runtime table and `docs/README.md` each have one new row.
- `git diff` does not touch another pipeline.
