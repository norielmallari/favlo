# AGENTS.md

Starting point for coding agents in **MADMetal**.

Read this file first. Then read [ARCHITECTURE.md](ARCHITECTURE.md) for the process and the write set. The pipeline you are editing has its own note: [docs/qwen38-flash-next/architecture.md](docs/qwen38-flash-next/architecture.md) for Flash Next, [docs/qwen38-27b/README.md](docs/qwen38-27b/README.md) for 27B. The papers and specifications the math is taken from are [REFERENCES.md](REFERENCES.md). Product name and the one-pipeline rule are [README.md](README.md). Root [ARCHITECTURE.md](ARCHITECTURE.md) wins for the listener and for which files an agent may edit. A pipeline note wins for that pipeline's kernels.

## What this repo is

MADMetal Accelerated Decoding on Metal. Each model is its own inference pipeline, written for that model's shapes and for the Metal tensor types on Apple Silicon.

`madmetal serve` is the entry point for inference. It listens with nothing loaded. A client selects a model through the OpenAI endpoints: `GET /v1/models` lists the registry, and the `model` field on `POST /v1/chat/completions` names one row. That row's pipeline loads. The resident pipeline releases itself before a different pipeline loads. Quantize, probe, matmul, kernels, bf16, and complete stay on the model flag: `madmetal --qwen38-flash-next quantize`, `madmetal --qwen38-27b quantize`, and the same flag for that pipeline's other commands. A new pipeline keeps its own flag for those commands.

## In scope

- `qwen38_flash_next`: BF16 calibration prefill, converter to native records, and the text runtime for that model. `qwen38-flash-next.py` is the entry the command loads.
- `madmetal serve` loads the pipeline a request's `model` field names. One record is resident. A different pipeline closes the resident record before it loads. Quantize and the other non-serve commands require that pipeline's model flag.
- OpenAI-compatible `GET /v1/models` and `POST /v1/chat/completions`, one request at a time, with streaming and tools. `GET /v1/models` lists every registered pipeline. While a function name is open, the sampler keeps only tokens that continue a name from the request. A `tool_choice` that names one function forces that call's opening as the tokenizer's encoding of that text.
- Prefix cache of that model's full state. The live sequence stays in memory. A request that does not extend it writes the sequence into the prefix directory and loads the longest stored prefix.
- Expert records streamed from SSD by Metal IO into a resident atlas. Consecutive ids are one sequential load of the live images. `--resident-percent` on `serve` and `complete` sets that atlas as a percent of the hot tier. Omitting it sizes the atlas from the memory the live sequence leaves. A shorter sequence keeps a larger atlas. The indexer attends at most 512 blocks of 4, plus the unfinished tail.

## Out of scope

- A shared model runner, a base class that every model subclasses, or a kernel that branches on model type.
- Both records resident, or a second listener.
- Affine group-64 packing and per-layer bit allocation.
- Batching, the Neural Engine, and the vision tower.
- A second output head, or a hot/cold split of the MTP expert file. An int2 span kernel. Publishing an accept length.
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
  prefix_store.py          finished prefills on SSD
  tools.py                 tool-call text parsed into OpenAI messages
  experts.py               resident expert atlas
  runtime.py               the forward
  commands.py              quantize, bf16, complete, and this pipeline's chat completion
qwen38-27b.py              command entry; loads the package by path
qwen38_27b/                Qwen3.8-27B. It does not import Flash Next.
src/madmetal/              process entry; the listener loads the pipeline a request names
  registry.py              flavor folders, aliases, pipelines, and sampling variants
  listener.py              the HTTP listener; it does not import a pipeline
  console.py               status row; no pipeline imports
harness/                   file-agent client, coherence, and the three benchmarks
docs/                      one directory per pipeline; see docs/README.md
docs/qwen38-flash-next/    Flash Next record, n-gram, schedule, probe
docs/qwen38-27b/           27B as-built note
pyproject.toml
```

A model entry filename may contain hyphens. The command loads it by path. It is not an importable module name. The package beside it uses underscores. The entry sets `PIPELINE` to that filename stem. The command discovers those files.

Adding a model means adding an entry file, a package, a `docs/<pipeline>/` note, and one row in the runtime table in [ARCHITECTURE.md](ARCHITECTURE.md). It does not mean editing another pipeline's package, shaders, tests, or docs, and it does not mean editing `src/madmetal/cli.py`.

## Re-implementation

Write algorithms from public research documents: papers, published specifications, and vendor hardware documentation. Do not open another application's tree and port its implementation. That covers local checkouts of inference runtimes and of corpus builders.

The calibration prompt strings are data in a bucket JSON: `tool_calling`, `chat`, `mixed`, `reasoning`, `code`, `en`, `ko`, `zh`, `ja`. `tool_calling` and `chat` are already chat-templated. They are not an algorithm.

## Invariants

1. One process, one listener. `madmetal serve` is that listener. The OpenAI `model` field selects the registry row, and that row loads one pipeline through its entry file. Quantize and the other non-serve commands require that pipeline's model flag.
2. The pipeline package owns the math. Shared code is the process entry, the registry, and the listener. There is no model base class.
3. Weights on disk for a pipeline are the Metal records that pipeline's note describes. Flash Next's record is [docs/qwen38-flash-next/record.md](docs/qwen38-flash-next/record.md).
4. The key/value cache defaults to E4M3 for completed blocks of four. The unfinished tail stays BF16. `--kv-cache bf16` stores every K and V token as BF16. The indexer key stays BF16.
5. Do not invent routing hits, calibration energies, or hidden-state errors. The device roundtrip in [docs/qwen38-flash-next/probe.md](docs/qwen38-flash-next/probe.md) measures nibble order and int4 endpoints before the expert table is packed.
6. A pipeline change updates that pipeline's `docs/<pipeline>/` note in the same change. A process change (listener, registry, console, or the write set) updates [ARCHITECTURE.md](ARCHITECTURE.md). Do not edit another pipeline's note to describe this one. `docs/` is as-built only. Implementation plans live under `artifacts/implementation-plan/` (gitignored). Do not commit plan or option-comparison notes under `docs/`.
7. This process is the runtime. Re-implement from public research documents. Do not use another application's source as the reference, and do not launch another application's environment.
8. Prefill, decode, and the calibration prefill map onto Metal. The endpoint serves tokens those shaders produced. A missing kernel stays missing. It is not replaced by a slower forward. A weight plane is a Metal tensor format whose op compiles and whose dispatch completes. half × int8 does not complete, so the hot plane is fp8 E4M3 and the cold plane is fp4 E2M1.

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
uv run madmetal --qwen38-27b quantize
uv run madmetal serve
uv run python -m harness --base-url http://127.0.0.1:8129/v1 --workspace ./note-workspace
uv run python -m harness.long_prompt --tokens 65536
uv run python -m harness.benchmarks
```

`python -m harness.benchmarks` scores IFEval, GSM8K, and HumanEval. The intelligence smoke is the first 30 rows of each, then the run continues through the full sets. `--sample-only` stops after those 30. That run is not part of `pytest`.

`quantize` reads the Hugging Face BF16 shards once and writes the record from `quant_map.json`. The shipped map is fp4 for every expert, and its `super` list is the pin. Those images stay in the file for their quant. Running it again on the same output continues from `quantize_progress.json`. After the manifest is written it stores one greedy grove completion in `grove.json`. `serve` reads that record. A request does not quantize a weight. `pytest` includes the file agent: it starts `serve` and checks that create, edit, and confirm land in the workspace. `python -m harness` is that client against an already running server. `python -m harness.long_prompt` starts `serve`, reads repository files one turn at a time until the prompt reaches the token target, and prints the server's prefill and decode rates. That run is not part of `pytest`.

`probe` writes one block of 32 and reads it back on the GPU. It is the measurement in [docs/qwen38-flash-next/probe.md](docs/qwen38-flash-next/probe.md). It does not pack an expert.

`matmul` runs one fp4 E2M1 tile and one fp8 E4M3 tile through `matmul2d`, and the fp4 tile through the decode shader dot. Those are the prefill and decode primitives. They do not load the checkpoint.
