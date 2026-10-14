# AGENTS.md

Starting point for coding agents in **MADMetal**.

Read this file first. Then read [ARCHITECTURE.md](ARCHITECTURE.md). The record, the n-gram hash, the kernel schedule, and the probe are in [docs/qwen38/](docs/qwen38/record.md). The papers and specifications the math is taken from are [REFERENCES.md](REFERENCES.md). Product name and the one-pipeline rule are [README.md](README.md). If another doc disagrees with the architecture, **ARCHITECTURE.md wins**.

## What this repo is

MADMetal Accelerated Decoding on Metal. Each model is its own inference pipeline, written for that model's shapes and for the Metal tensor types on Apple Silicon.

The first pipeline is Qwen3.8-Flash-Next. `madmetal --qwen38-flash-next` loads `qwen38-flash-next.py`, which loads the `qwen38_flash_next` package.

The runtime does not call into a multi-model server to run the forward.

## In scope

- `qwen38_flash_next`: BF16 calibration prefill, converter to native records, and the text runtime for that model. `qwen38-flash-next.py` is the entry the command loads.
- `madmetal` command that dispatches one model flag onto one pipeline entry.
- OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, one request at a time, with streaming and tools. While a function name is open, the sampler keeps only tokens that continue a name from the request. A `tool_choice` that names one function forces that call's opening as the tokenizer's encoding of that text.
- Prefix cache of that model's full state.
- Expert records streamed from SSD by Metal IO into a resident atlas. Consecutive ids are one sequential load of the live images. `--resident-percent` on `serve` and `complete` sets that atlas as a percent of the hot tier. Omitting it keeps the 24 GiB ladder. The prompt length is the tokens that fit after that budget, and at most 131,072. The indexer attends at most 512 blocks of 4, plus the unfinished tail.

## Out of scope

- A shared model runner, a base class that every model subclasses, or a kernel that branches on model type.
- A second model until that model is explicitly added.
- Affine group-64 packing and per-layer bit allocation.
- Batching, the Neural Engine, and the vision tower.
- Multi-token prediction in the serving loop. The MTP weights are packed. The endpoint samples one token at a time.
- Fusing a whole layer into one kernel. The runtime is a layered loop. Shaders share a command buffer until the host needs an expert file offset or the sampled token id.
- Using another application's source, tests, comments, or design notes as the reference for a profiler, quantizer, kernel, or forward.
- A host GEMV, a CPU matmul of activations, or any stand-in forward that runs because a Metal kernel is missing. Prefill, decode, and the calibration prefill are Metal shaders on Apple Silicon.

## Layout

```text
qwen38-flash-next.py       command entry; loads the package by path
qwen38_flash_next/         this model's pipeline
  shaders/                 Metal source, read as text
  formats.py               record math
  allocate.py              contiguous page-aligned files
  convert.py               BF16 shards to the record
  probe.py                 block and matmul checks
  session.py               device buffers and shaders
  checks.py                host formulas for the kernel checks
  record.py                the file the shaders read
  tools.py                 tool-call text parsed into OpenAI messages
  experts.py               resident expert atlas
  runtime.py               the forward
  commands.py              quantize, bf16, complete, serve
src/madmetal/              process entry; selects one pipeline and stops
harness/                   file-agent client, coherence, and the three benchmarks
docs/qwen38/               record, n-gram, schedule, and probe notes
pyproject.toml
```

A model entry filename may contain hyphens. The command loads it by path. It is not an importable module name. The package beside it uses underscores.

Adding a model means adding an entry file, a package, and a flag. It does not mean editing `qwen38_flash_next`.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders.

The calibration prompt strings are data in a bucket JSON: `tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`. `tool_calling` and `chat` are already chat-templated. They are not an algorithm.

## Invariants

1. One process, one model flag, one pipeline package. The command loads that package through one hyphenated entry file.
2. The pipeline package owns the math. Shared code is limited to process entry and pieces that are not the model, and only when a second pipeline exists to share them.
3. Weights on disk for this model are the Metal records in [docs/qwen38/record.md](docs/qwen38/record.md). The tier rules are in ARCHITECTURE.md.
4. The key/value cache defaults to E4M3 for completed blocks of four. The unfinished tail stays BF16. `--kv-cache bf16` stores every K and V token as BF16. The indexer key stays BF16.
5. Do not invent routing hits, calibration energies, or hidden-state errors. The device roundtrip in [docs/qwen38/probe.md](docs/qwen38/probe.md) measures nibble order and int4 endpoints before the expert table is packed.
6. Keep ARCHITECTURE.md current in the same change that alters the pipeline, the records, the cache, the endpoint, or the scope. Update the note in `docs/qwen38/` that owns a layout, the n-gram hash, the kernel schedule, or the probe. ARCHITECTURE.md wins if they disagree.
7. This process is the runtime. Re-implement from public research documents. Do not use another application's source as the reference, and do not launch another application's environment.
8. Prefill, decode, and the calibration prefill map onto Metal. The endpoint serves tokens those shaders produced. A missing kernel stays missing. It is not replaced by a slower forward.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): imperative summary` on the subject line. Common types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `build`, `ci`. Scope is optional; prefer a short pipeline id such as `qwen38` or `madmetal` when the change is localized.

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
uv run madmetal --qwen38-flash-next probe
uv run madmetal --qwen38-flash-next matmul
uv run madmetal --qwen38-flash-next kernels
uv run madmetal --qwen38-flash-next quantize
uv run madmetal --qwen38-flash-next serve
uv run python -m harness --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness.long_prompt --tokens 65536
uv run python -m harness.benchmarks
```

`python -m harness.benchmarks` scores IFEval, GSM8K, and HumanEval. The intelligence smoke is the first 30 rows of each, then the run continues through the full sets. `--sample-only` stops after those 30. That run is not part of `pytest`.

`quantize` reads the Hugging Face BF16 shards once and writes the record. `--hot` selects the int4 keep set. `--super` lifts listed ids out of that keep set into resident int8. The complement of the keep set is int2. Running it again on the same output continues from `quantize_progress.json`. After the manifest is written it stores one greedy grove completion in `grove.json`. `serve` reads that record. A request does not quantize a weight. `pytest` includes the file agent: it starts `serve` and checks that create, edit, and confirm land in the workspace. `python -m harness` is that client against an already running server. `python -m harness.long_prompt` starts `serve`, reads repository files one turn at a time until the prompt reaches the token target, and prints the server's prefill and decode rates. That run is not part of `pytest`.

`probe` writes one block of 32 and reads it back on the GPU. It is the measurement in [docs/qwen38/probe.md](docs/qwen38/probe.md). It does not pack an expert.

`matmul` runs one int4 E8M0 tile through `matmul2d` and the decode shader dot. Those are the prefill and decode primitives. They do not load the checkpoint.
