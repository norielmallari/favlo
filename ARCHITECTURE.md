# MADMetal architecture

As-built contract for the process. The name expands to MADMetal Accelerated Decoding on Metal.

One model is one pipeline. `madmetal serve` is the entry point. It listens with nothing loaded. `GET /v1/models` lists `models/registry.json`. The `model` field on `POST /v1/chat/completions` selects one row, and that row's `pipeline` names the entry file that loads. The resident pipeline releases its record, device buffers, cache, and live prefix before a different pipeline loads. Two records are not resident together. A pipeline does not import another pipeline, and it does not share that pipeline's forward, records, or kernels. Quantize and the other non-serve commands require `madmetal --<pipeline>`. Each pipeline keeps that flag.

This file wins for the listener, the registry, the console, and the write set below. A pipeline's own note wins for that pipeline's records, cache, and kernels. Implementation plans stay under `artifacts/implementation-plan/` (gitignored).

## Runtimes

Each runtime is a separate write set. An agent developing one runtime edits that set and leaves the others untouched.

| Pipeline | Entry | Package | As-built note |
|---|---|---|---|
| `qwen38-flash-next` | `qwen38-flash-next.py` | `qwen38_flash_next/` | [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md) |
| `qwen38-27b` | `qwen38-27b.py` | `qwen38_27b/` | [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md) |

The entry file sets `PIPELINE` to its filename stem. The command discovers those files. Adding a runtime does not edit `src/madmetal/cli.py`.

A runtime's write set is the entry file, the package, `docs/<pipeline>/`, and tests that import only that package. A new runtime adds a row to the table above and its own doc directory. It does not edit another row's package, shaders, tests, or docs.

Shared code is `src/madmetal/`: the process entry, the registry, and the listener. There is no model base class and no kernel that branches on the pipeline. `src/madmetal/` changes when the listener, the registry, or the console changes. A pipeline task does not change it to reach into another runtime.

The registry stores a sampling card on each row. The pipeline that registers the row supplies that card. The card in `src/madmetal/registry.py` is the Flash Next model card, used when a caller omits one. Another runtime passes its own card and does not replace that default.

Tests for the listener, the registry, and the console live in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_registry.py`, and `tests/test_console.py`. A pipeline's tests do not import another pipeline.

## Listener

`madmetal serve` listens on `127.0.0.1:8129` with nothing loaded. `GET /v1/models` lists every registered row. `POST /v1/chat/completions` loads the pipeline that row names, one request at a time. A connection that arrives while a request is open waits until that request has returned. A different folder or a different pipeline unloads the previous record first. An unknown id is 404. `serve` does not register a record it finds on disk.

`models/registry.json` is gitignored. It stores folder names, the pipeline entry, an alias, the BF16 source path, the key/value mode, the resident percent, and the thinking and instruct sampling rows. A fresh tree has no rows.

## Console

Status goes through `src/madmetal/console.py`. The module imports nothing from a pipeline. The listener does not import a pipeline until a request names one. That row's entry file provides `open_runtime` and `handle_chat`. `--console auto` is pretty when stderr is a terminal, and plain when it is a pipe, when `NO_COLOR` is set, or when `TERM` is `dumb`. `plain` prints the lines the harness parses. `pretty` draws one row per open phase at 10 Hz. `json` writes one NDJSON object per event, schema version 1. How a pipeline binds those phases is that pipeline's note.
