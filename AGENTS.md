# AGENTS.md

Starting point for coding agents in **MADMetal**.

Read this file first. Then read [ARCHITECTURE.md](ARCHITECTURE.md). The papers and specifications the math is taken from are [REFERENCES.md](REFERENCES.md). Product name and the one-pipeline rule are [README.md](README.md). If another doc disagrees with the architecture, **ARCHITECTURE.md wins**.

## What this repo is

MADMetal Accelerated Decoding on Metal. Each model is its own inference pipeline, written for that model's shapes and for the Metal tensor types on Apple Silicon.

The first pipeline is Qwen3.8-Flash-Next. `madmetal --qwen38-flash-next` loads `qwen38-flash-next.py`, which loads the `qwen38_flash_next` package.

The runtime does not call into a multi-model server to run the forward.

## In scope

- `qwen38_flash_next`: BF16 calibration prefill, converter to native records, and the text runtime for that model. `qwen38-flash-next.py` is the entry the command loads.
- `madmetal` command that dispatches one model flag onto one pipeline entry.
- OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, one request at a time, with streaming, tools, and a grammar on the tool-call span.
- Prefix cache of that model's full state.
- Expert records streamed from SSD by Metal IO into a resident atlas. Consecutive ids are one sequential load of the live images.

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
  experts.py               resident expert atlas
  runtime.py               the forward
  commands.py              quantize, bf16, complete, serve
src/madmetal/              process entry; selects one pipeline and stops
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
3. Weights on disk for this model are the Metal records in ARCHITECTURE.md.
4. The key/value cache defaults to BF16 for every cached token. fp8 E4M3 and E5M2 are a switch.
5. Do not invent routing hits, calibration energies, or hidden-state errors. The device roundtrip in ARCHITECTURE.md measures nibble order and int4 endpoints before the expert table is packed.
6. Keep ARCHITECTURE.md current in the same change that alters the pipeline, the records, the cache, the endpoint, or the scope.
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
```

`quantize` reads the Hugging Face BF16 shards once and writes the record. `--hot` selects the int4 keep set. `--super` lifts listed ids out of that keep set into resident int8. The complement of the keep set is int2. Running it again on the same output continues from `quantize_progress.json`. After the manifest is written it stores one greedy grove completion in `grove.json`. `serve` reads that record. A request does not quantize a weight.

`probe` writes one block of 32 and reads it back on the GPU. It is the measurement in ARCHITECTURE.md. It does not pack an expert.

`matmul` runs one int4 E8M0 tile through `matmul2d` and the decode shader dot. Those are the prefill and decode primitives. They do not load the checkpoint.
