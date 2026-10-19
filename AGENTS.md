# AGENTS.md

Starting point for coding agents in **Favlo**.

Read this file first. Changes stay in this repository. Edit another repository only when the user asks for that repository. Then read [ARCHITECTURE.md](ARCHITECTURE.md) for the process and the write set. The pipeline you are editing has its own note: [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md) for Flash Next, [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md) for 27B, [docs/qwen36-35b-a3b/README.md](docs/qwen36-35b-a3b/README.md) for Qwen3.6-35B-A3B. The papers and specifications the math is taken from are [REFERENCES.md](REFERENCES.md). A pipeline's own citations live in its note. Product name and the one-pipeline rule are [README.md](README.md). Root [ARCHITECTURE.md](ARCHITECTURE.md) wins for the listener and for which files an agent may edit. A pipeline note wins for that pipeline's kernels. AI-assisted features live in [docs/ai-assistance](docs/ai-assistance/README.md). An update to one of those features updates that folder in the same change.

## What this repo is

Each model is its own inference pipeline, written for that model's shapes and for the Metal tensor types on Apple Silicon.

`favlo serve` is the entry point for inference. It listens with nothing loaded. A client selects a model through the OpenAI endpoints: `GET /v1/models` lists the registry, and the `model` field on `POST /v1/chat/completions` names one row. That row's pipeline loads. The resident pipeline releases itself before a different pipeline loads. Quantize, probe, matmul, kernels, bf16, and complete stay on the model flag: `favlo --qwen38-flash-next quantize`, `favlo --qwen38-27b quantize`, and the same flag for that pipeline's other commands. A new pipeline keeps its own flag for those commands.

## In scope

- `favlo serve` loads the pipeline a request's `model` field names. One record is resident. A different pipeline closes the resident record before it loads. Quantize and the other non-serve commands require that pipeline's model flag.
- OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, one request at a time. `GET /v1/models` lists every registered row, including each sampling suffix that row stores.
- A pipeline's record, cache, kernels, and sampling card are that pipeline's note. Flash Next is [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md).

## Out of scope

- A shared model runner, a base class that every model subclasses, or a kernel that branches on model type.
- Both records resident, or a second listener.
- Batching and the Neural Engine.
- Fusing a whole layer into one kernel. A pipeline's note describes its own loop.
- Using another application's source, tests, comments, or design notes as the reference for a profiler, quantizer, kernel, or forward.
- A host GEMV, a CPU matmul of activations, or any stand-in forward that runs because a Metal kernel is missing. Prefill, decode, and the calibration prefill are Metal shaders on Apple Silicon.

## Layout

```text
qwen38-flash-next.py       command entry; loads the package by path
qwen38_flash_next/         this model's pipeline; see its note
qwen38-27b.py              command entry; loads the package by path
qwen38_27b/                Qwen3.8-27B. It does not import Flash Next.
qwen36-35b-a3b.py          command entry; loads the package by path
qwen36_35b_a3b/            Qwen3.6-35B-A3B. It does not import another pipeline.
src/favlo/              process entry; the listener loads the pipeline a request names
  registry.py              flavor folders, aliases, pipelines, and sampling variants
  listener.py              HTTP daemon; a terminal starts it and keeps the CLI
  daemon.py                spawn and stop the listener; no pipeline imports
  console.py               status row; no pipeline imports
  logs.py                  rolling server log; size is settings.json log_max_bytes
harness/                   file-agent client, coherence, and the three benchmarks
docs/                      pipeline notes, guides, and AI-assisted features; see docs/README.md
docs/ai-assistance/        terminal chat and the other AI-assisted features
docs/qwen38-flash-next/    Flash Next record, n-gram, schedule, probe
docs/qwen38-27b/           27B as-built note
docs/qwen36-35b-a3b/       Qwen3.6-35B-A3B as-built note
pyproject.toml
```

A model entry filename may contain hyphens. The command loads it by path. It is not an importable module name. The package beside it uses underscores. The entry sets `PIPELINE` to that filename stem. The command discovers those files.

Adding a model means adding an entry file, a package, a `docs/<pipeline>/` note, and one row in the runtime table in [ARCHITECTURE.md](ARCHITECTURE.md). It does not mean editing another pipeline's package, shaders, tests, or docs, and it does not mean editing `src/favlo/cli.py`. Load [bootstrap-new-model](.cursor/skills/bootstrap-new-model/SKILL.md) before creating one. Load [prefix-caching](.cursor/skills/prefix-caching/SKILL.md) before wiring that pipeline to the shared prefix store.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders.

A pipeline's calibration strings are data, not an algorithm. The bucket for Flash Next is in its references note.

## Invariants

1. One listener. The daemon started by `favlo serve --non-interactive` is that listener. `favlo serve` on a terminal is the CLI and attaches to it. The OpenAI `model` field selects the registry row, and that row loads one pipeline through its entry file. Quantize and the other non-serve commands require that pipeline's model flag.
2. The pipeline package owns the math. Shared code is the process entry, the registry, and the listener. There is no model base class.
3. Weights on disk for a pipeline are the Metal records that pipeline's note describes.
4. A pipeline's key/value layout is that pipeline's note.
5. Do not invent routing hits, calibration energies, or hidden-state errors. A measurement belongs in that pipeline's note.
6. A pipeline change updates that pipeline's `docs/<pipeline>/` note in the same change. A process change (listener, registry, console, or the write set) updates [ARCHITECTURE.md](ARCHITECTURE.md). An AI-assisted feature change updates [docs/ai-assistance](docs/ai-assistance/README.md) in the same change. Add a page under that folder when the feature needs its own note, and list the page from that folder's README. Do not edit another pipeline's note to describe this one. `docs/` is as-built only. Implementation plans live under `artifacts/implementation-plan/` (gitignored). Do not commit plan or option-comparison notes under `docs/`.
7. This process is the runtime. Re-implement from public research documents. Do not use another application's source as the reference, and do not launch another application's environment.
8. Changes stay in this repository. Another repository stays untouched unless the user asks for that repository.
9. Prefill, decode, and the calibration prefill map onto Metal. The endpoint serves tokens those shaders produced. A missing kernel stays missing. It is not replaced by a slower forward. A weight plane is a Metal tensor format whose op compiles and whose dispatch completes. The formats a pipeline stores are that pipeline's note.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary` on the subject line. Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`. Scope is optional; prefer a short pipeline id such as `qwen38-flash-next` or `shared` when the change is localized.

```text
feat(qwen38): stream experts from SSD into the resident atlas
perf(qwen38): fuse scan prefix into MoE prefill shader
docs: describe parallel quantize worker layout
```

Optional hook (local only; do not commit git config):

```text
git config core.hooksPath .githooks
```

## Commands

```text
uv sync --extra dev
uv run pytest
uv run favlo --qwen38-flash-next probe
uv run favlo --qwen38-flash-next matmul
uv run favlo --qwen38-flash-next kernels
uv run favlo --qwen38-flash-next quantize
uv run favlo --qwen38-27b quantize
uv run favlo --qwen36-35b-a3b kernels
uv run favlo --qwen36-35b-a3b quantize
uv run favlo serve
uv run python -m harness.file_agent --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness --model qwen38-27b
uv run python -m harness.benchmarks --model qwen38-27b
uv run python -m harness.coding --model qwen38-27b
```

`python -m harness.benchmarks --model <id>` scores IFEval, GSM8K, and HumanEval against `serve`. The `model` field is that id. Thinking stays off. Omitted sampling fields are filled from `GET /v1/models` row `<id>:instruct`. A CLI flag overrides that field. The checker reads `message.content`. `--limit 30` scores the first 30 rows and writes the smoke section. `--limit 100` writes the sample of 100. A run with no limit scores the smoke, then the full sets. That run is not part of `pytest`. Each finished suite is written into `docs/<pipeline>/benchmarks.md`.

`pytest` includes the file agent: it starts `serve` and checks that create, edit, and confirm land in the workspace. `python -m harness.file_agent` is that client against an already running server. `python -m harness --model <id>` starts `serve`, reads repository files one turn at a time until the prompt reaches the token target (default 65536), and prints the server's prefill and decode rates. The model field is the pipeline name. The request sets `enable_thinking` false and leaves every other sampling field unset, so the listener fills the instruct card. The server stops after the first turn and starts another, so the second turn loads the prefix from disk. Later turns stay on that server. That run is not part of `pytest`. `python -m harness.coding --model <id>` starts `serve` and sets `enable_thinking` true on that same bare id. The model writes `ledger.py` through tools. A pytest file outside the workspace is the result. Exit 0 is a pass. The command prints each request's prefill and decode line. That series stays separate from the file agent's. That run is not part of `pytest`. A pipeline's quantize, probe, and matmul commands are that pipeline's note.
