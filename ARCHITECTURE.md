# Favlo architecture

One model is one pipeline. `favlo serve` listens with nothing loaded. The `model` field on `POST /v1/chat/completions` loads one registry row. Two records are not resident together. A pipeline does not import another pipeline, and it does not share that pipeline's forward, records, or kernels. Quantize and the other non-serve commands require `favlo --<pipeline>`.

This file is the map and the write set. It wins for the listener, the registry, the console, and which files an agent may edit. A pipeline note wins for that pipeline's records, cache, and kernels. Implementation plans stay under `artifacts/implementation-plan/` (gitignored).

## Map

| Area | Code | Note |
|---|---|---|
| Command discovery | `src/favlo/cli.py` | This file. The command finds each `*-*.py` entry. Adding a runtime does not edit `cli.py`. |
| Listener and daemon | `src/favlo/listener.py`, `src/favlo/daemon.py` | [docs/serve.md](docs/serve.md) |
| Terminal chat | `src/favlo/talk.py`, `src/favlo/persona.py` | [docs/ai-assistance](docs/ai-assistance/README.md) |
| Console | `src/favlo/console.py` | [docs/console.md](docs/console.md) |
| Registry | `src/favlo/registry.py`, `models/registry.json` | This file |
| Prefix store | `src/favlo/prefix.py` | [docs/prefix.md](docs/prefix.md) |
| First-run setup | `src/favlo/setup.py` | [docs/guides/models.md](docs/guides/models.md) |
| Harness | `harness/` | [docs/harness.md](docs/harness.md) |

Shared code is `src/favlo/`. There is no model base class and no kernel that branches on the pipeline. A pipeline task does not change `src/favlo/` to reach into another runtime. A pipeline imports the prefix store and adapts its own buffers.

The daemon is a request server. It does not own chat history, tools, or the terminal session. The CLI owns those. Telemetry is one JSON event per line. The terminal paints it as it arrives, and prints the reply after that turn's requests have closed.

## Runtimes

Each runtime is a separate write set. An agent developing one runtime edits that set and leaves the others untouched.

| Pipeline | Entry | Package | As-built note |
|---|---|---|---|
| `qwen38-flash-next` | `qwen38-flash-next.py` | `qwen38_flash_next/` | [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md) |
| `qwen38-27b` | `qwen38-27b.py` | `qwen38_27b/` | [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md) |
| `qwen36-35b-a3b` | `qwen36-35b-a3b.py` | `qwen36_35b_a3b/` | [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md) |

The entry filename may contain hyphens. The command loads it by path. It is not an importable module name. The package beside it uses underscores. The entry sets `PIPELINE` to that filename stem. A runtime's write set is the entry file, the package, `docs/<pipeline>/`, and tests that import only that package. A new runtime adds a row to the table above and its own doc directory. It does not edit another row's package, shaders, tests, or docs.

Tests for the listener, the registry, and the console live in `tests/test_cli.py`, `tests/test_server.py`, `tests/test_registry.py`, and `tests/test_console.py`. A pipeline's tests do not import another pipeline.

## Registry

`models/registry.json` is gitignored. A row stores the folder, the pipeline entry, an alias, the BF16 source path, the key/value mode, the resident percent, and the sampling card copied from that pipeline's `sampling.py`. The stored names are the `:suffix` ids. A bare model id selects no row. The pipeline applies its default variant, or the row the request selects. `serve` does not register a record it finds on disk. A fresh tree has no rows. `GET /v1/models` lists every registered row, including each sampling suffix that row stores.

## Layout

```text
qwen38-flash-next.py    command entry; loads the package by path
qwen38_flash_next/      this model's pipeline; see its note
qwen38-27b.py           command entry; loads the package by path
qwen38_27b/             Qwen3.8-27B. It does not import Flash Next.
qwen36-35b-a3b.py       command entry; loads the package by path
qwen36_35b_a3b/         Qwen3.6-35B-A3B. It does not import another pipeline.
src/favlo/              process entry; the listener loads the pipeline a request names
  cli.py                discovers entry files; a new runtime does not edit this
  registry.py           folders, aliases, pipelines, and sampling variants
  listener.py           HTTP daemon; a terminal starts it and keeps the CLI
  daemon.py             spawn and stop the listener; no pipeline imports
  console.py            status row; no pipeline imports
  talk.py               terminal chat
  persona.py            voice and saved openings
  prefix.py             SSD prefix store
  setup.py              empty-registry download
  logs.py               rolling server log; size is settings.json log_max_bytes
harness/                file agent, coding agent, and the three benchmarks
docs/                   as-built notes; see docs/README.md
```

Probe, matmul, kernels, bf16, complete, and quantize stay on that pipeline's flag: `favlo --qwen38-flash-next quantize`, and the same flag for the other commands. The process does not batch and does not use the Neural Engine. A pipeline does not fuse a whole layer into one kernel. Prefill, decode, and the calibration prefill are Metal shaders. A missing kernel stays missing.

## Commands

```text
uv sync --extra dev
uv run pytest
uv run favlo serve
uv run favlo --qwen38-flash-next probe
uv run favlo --qwen38-flash-next matmul
uv run favlo --qwen38-flash-next kernels
uv run favlo --qwen38-flash-next quantize
uv run favlo --qwen38-27b quantize
uv run favlo --qwen36-35b-a3b kernels
uv run favlo --qwen36-35b-a3b quantize
uv run python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness --model qwen38-27b
uv run python -m harness.benchmarks --model qwen38-27b
uv run python -m harness.coding --model qwen38-27b
```

What the harness scores is [docs/harness.md](docs/harness.md). A pipeline's quantize, probe, and matmul commands are that pipeline's note.
