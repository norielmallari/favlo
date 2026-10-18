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
| `qwen36-35b-a3b` | `qwen36-35b-a3b.py` | `qwen36_35b_a3b/` | [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md) |

The entry file sets `PIPELINE` to its filename stem. The command discovers those files. Adding a runtime does not edit `src/madmetal/cli.py`.

A runtime's write set is the entry file, the package, `docs/<pipeline>/`, and tests that import only that package. A new runtime adds a row to the table above and its own doc directory. It does not edit another row's package, shaders, tests, or docs.

Shared code is `src/madmetal/`: the process entry, the registry, the listener, and the SSD prefix store. There is no model base class and no kernel that branches on the pipeline. `src/madmetal/` changes when the listener, the registry, the console, or the prefix store changes. A pipeline task does not change it to reach into another runtime. A pipeline imports the prefix store and adapts its own buffers. It does not import another pipeline.

The registry stores a sampling card on each row. Registration loads `VARIANTS` and `DEFAULT_VARIANT` from that pipeline's `sampling.py` and writes them on the row. The stored names are the `:suffix` ids. A bare model id selects no row. The pipeline applies its default variant, or the row the request selects.

Tests for the listener, the registry, and the console live in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_registry.py`, and `tests/test_console.py`. A pipeline's tests do not import another pipeline.

## Listener

`madmetal serve` listens on `127.0.0.1:8129` with nothing loaded. `--max-memory` is passed to a pipeline whose `open_runtime` accepts it, as the maximum bytes that runtime may allocate. `GET /v1/models` lists every registered row. `POST /v1/chat/completions` loads the pipeline that row names, one request at a time. A connection that arrives while a request is open waits until that request has returned. A different folder or a different pipeline unloads the previous record first. An unknown id is 404. `serve` does not register a record it finds on disk.

`models/registry.json` is gitignored. It stores folder names, the pipeline entry, an alias, the BF16 source path, the key/value mode, the resident percent, and the sampling rows copied from that pipeline. A fresh tree has no rows.

## Console

Status goes through `src/madmetal/console.py`. The module imports nothing from a pipeline. The listener does not import a pipeline until a request names one. That row's entry file provides `open_runtime` and `handle_chat`. `--console auto` is pretty when stderr is a terminal, and plain when it is a pipe, when `NO_COLOR` is set, or when `TERM` is `dumb`. `plain` prints the lines the harness parses. `pretty` draws one row per open phase at 10 Hz. `json` writes one NDJSON object per event, schema version 1. How a pipeline binds those phases is that pipeline's note.

`madmetal serve` also appends those durable lines, phase summaries, and tracebacks to `~/.madmetal/logs/madmetal.log`. The live row is not written there. `~/.madmetal/settings.json` key `log_max_bytes` is the most bytes that file may hold. The default is 1048576. Keys `max_tokens` and `thinking_budget` apply only when present. An omitted `max_tokens` on the request and in that file does not cap the completion. An omitted `thinking_budget` does not close a think block on a token count. The first serve creates that file when it is missing. A line that would pass the cap leaves the newest bytes that still fit, starting at the next line when one fits.

## Prefix cache

`src/madmetal/prefix.py` stores one sequence as token ids plus named regions under `~/.madmetal/prefix/<model>/<checkpoint>/<hash>.bin`. The checkpoint id is the hex SHA-256 of `manifest.json`, or of `config.json` when the record has no manifest. The file name is the hex SHA-256 of the format, the model id, that digest, the cache tag, and the token ids. The store does not know a key/value layout. Each pipeline packs its own buffers into regions and copies a matching file back.

A prompt that extends the live tokens keeps the device. Any other prompt writes the live sequence, clears the device, and loads the longest stored prefix that pipeline accepts. Regions start on a 16 KB page. The write replaces the destination after the header hash is in place. A region whose stored bytes are a prefix of the new blob is copied forward. One model and one checkpoint stay within 32 GiB. Past that, the oldest files are deleted and the file just written stays. What a pipeline puts in the regions, and whether an exact-length file can be sampled, is that pipeline's note.

## Harness

`python -m harness.coding --model <id>` is a client of `serve`. It loads the thinking sampling card for that id and sends `enable_thinking` true. The model writes `ledger.py` in a throwaway workspace. The suite that decides the run lives outside that workspace. Exit 0 is a pass. The file-agent check in `pytest` is a separate three-step tool check.
